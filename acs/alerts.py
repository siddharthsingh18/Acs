"""
acs.alerts
Enterprise notification alert system for emergency dispatch services.
Supports:
- Web Push (PWA) browser and phone notifications via VAPID / pywebpush
- Telegram Bot HTTP API (sendPhoto / sendMessage)
- SMTP Email (HTML + Plaintext MIMEMultipart with embedded keyframe image)
- Console logging
- Storm Rate-Limiting per camera / location during collision pile-up storms
- De-duplication per incident and per channel tier
- Hard Interlock: DEMO incidents are mathematically barred from real external channels
- Escalation: Multi-tier notifications with one-click acknowledgment
- Mandatory disclaimer on all dispatches
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from email.mime.image import MIMEImage
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
import json
import logging
import os
import secrets
import smtplib
import threading
import time
from typing import Any, Dict, List, Optional
from urllib.parse import quote_plus
from dotenv import load_dotenv
import requests

from acs.config import config

load_dotenv()
logger = logging.getLogger("acs.alerts")

DISCLAIMER_TEXT = (
    "⚠️ DISCLAIMER: Alerts support human decisions and must not trigger emergency dispatch automatically."
)


def is_demo_incident(incident_id: str, metadata: Optional[Dict[str, Any]] = None) -> bool:
    """Returns True if the incident is marked demo, sample, or test.

    Hard Interlock Guarantee:
    DEMO incidents are mathematically barred from dispatching through real external networks.
    """
    if not incident_id:
        return True
    upper_id = str(incident_id).strip().upper()
    if upper_id.startswith(("DEMO-", "SAMPLE-", "TEST-", "INC-DEMO")):
        return True
    if metadata:
        if metadata.get("is_demo") is True:
            return True
        source = str(metadata.get("source", "")).lower()
        if "sample" in source or "demo" in source or "test" in source:
            return True
        if str(metadata.get("video_source", "")).lower() in ("sample", "demo"):
            return True
    return False


class StormRateLimiter:
    """Sliding-window rate limiter per location or camera to prevent alert storms."""

    def __init__(self, window_seconds: float = 60.0, max_alerts: int = 2):
        self.window_seconds = window_seconds
        self.max_alerts = max_alerts
        self._history: Dict[str, List[float]] = {}
        self._lock = threading.Lock()

    def should_rate_limit(self, key: str) -> bool:
        """Returns True if key has exceeded alert quota within window."""
        now = time.time()
        with self._lock:
            timestamps = self._history.get(key, [])
            cutoff = now - self.window_seconds
            valid = [ts for ts in timestamps if ts > cutoff]
            if len(valid) >= self.max_alerts:
                self._history[key] = valid
                return True
            valid.append(now)
            self._history[key] = valid
            return False

    def reset(self) -> None:
        """Clears rate limiter state."""
        with self._lock:
            self._history.clear()


storm_rate_limiter = StormRateLimiter(
    window_seconds=getattr(config, "storm_rate_limit_seconds", 60.0),
    max_alerts=getattr(config, "storm_max_alerts_per_window", 2),
)


def get_or_generate_vapid_keys() -> Dict[str, str]:
    """Retrieves existing VAPID keys or creates an EC P-256 pair saved to reports/vapid_keys.json."""
    pub = os.getenv("VAPID_PUBLIC_KEY") or getattr(config, "vapid_public_key", None)
    priv = os.getenv("VAPID_PRIVATE_KEY") or getattr(config, "vapid_private_key", None)
    claims_sub = os.getenv("VAPID_CLAIMS_SUB") or getattr(config, "vapid_claims_sub", "mailto:alerts@acs-surveillance.local")

    if pub and priv:
        return {"public_key": pub, "private_key": priv, "claims_sub": claims_sub}

    keys_file = getattr(config, "vapid_keys_file", "reports/vapid_keys.json")
    if os.path.exists(keys_file):
        try:
            with open(keys_file, "r") as f:
                data = json.load(f)
                if data.get("public_key") and data.get("private_key"):
                    return {
                        "public_key": data["public_key"],
                        "private_key": data["private_key"],
                        "claims_sub": data.get("claims_sub", claims_sub),
                    }
        except Exception as e:
            logger.warning("Failed to read VAPID keys file %s: %s", keys_file, e)

    # Generate new EC P-256 VAPID keys using py_vapid
    try:
        from py_vapid import Vapid
        from py_vapid.utils import b64urlencode
        from cryptography.hazmat.primitives import serialization

        vapid = Vapid()
        vapid.generate_keys()
        raw_pub = vapid.public_key.public_bytes(
            encoding=serialization.Encoding.X962,
            format=serialization.PublicFormat.UncompressedPoint,
        )
        pub_b64 = b64urlencode(raw_pub)
        priv_pem = vapid.private_key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption(),
        ).decode("utf-8")

        os.makedirs(os.path.dirname(os.path.abspath(keys_file)), exist_ok=True)
        key_data = {
            "public_key": pub_b64,
            "private_key": priv_pem,
            "claims_sub": claims_sub,
        }
        with open(keys_file, "w") as f:
            json.dump(key_data, f, indent=2)
        logger.info("Generated new VAPID keys and saved to %s", keys_file)
        return key_data
    except Exception as err:
        logger.error("Error generating VAPID keys: %s", err)
        return {}


def record_alert_delivery(
    incident_id: str,
    channel: str,
    tier: int,
    status: str,
    details: Optional[str] = None,
) -> None:
    """Persists alert delivery status record into SQLite database."""
    try:
        from sqlmodel import Session
        from backend.app.db.database import engine
        from backend.app.db.models import AlertDelivery

        with Session(engine) as session:
            delivery = AlertDelivery(
                incident_id=incident_id,
                channel=channel,
                tier=tier,
                status=status,
                details=details,
                created_at=time.time(),
            )
            session.add(delivery)
            session.commit()
    except Exception as exc:
        logger.debug("Could not record alert delivery in database: %s", exc)

    try:
        from backend.app.core.metrics import record_alert_delivery as record_alert_delivery_metric
        record_alert_delivery_metric(channel=channel, status=status)
    except Exception:
        pass


def check_existing_delivery(incident_id: str, channel: str, tier: int) -> bool:
    """Checks if an alert for this incident, channel, and tier has already been dispatched (de-duplication)."""
    try:
        from sqlmodel import Session, select
        from backend.app.db.database import engine
        from backend.app.db.models import AlertDelivery

        with Session(engine) as session:
            stmt = select(AlertDelivery).where(
                AlertDelivery.incident_id == incident_id,
                AlertDelivery.channel == channel,
                AlertDelivery.tier == tier,
                AlertDelivery.status.in_(["sent", "dry_run", "acknowledged"]),
            )
            existing = session.exec(stmt).first()
            return existing is not None
    except Exception:
        return False


@dataclass
class AlertPayload:
    """Encapsulates incident details, keyframe attachments, and links for dispatch alerts."""

    incident_id: str
    class_name: str
    timestamp: str
    confidence: float
    report_text: str
    image_path: Optional[str] = None
    report_filename: Optional[str] = None
    dry_run: bool = False
    severity: str = "medium"
    severity_score: int = 50
    camera_id: Optional[str] = None
    location_name: Optional[str] = None
    location_lat: Optional[float] = None
    location_lng: Optional[float] = None
    ack_token: Optional[str] = None
    ack_url: Optional[str] = None
    map_url: Optional[str] = None
    keyframe_url: Optional[str] = None
    tier: int = 1
    is_escalation: bool = False
    is_demo: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)


class BaseNotifier(ABC):
    """Abstract base notifier interface for pluggable alert backends."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique identifier name for this notification channel."""
        pass

    @abstractmethod
    def is_configured(self) -> bool:
        """Returns True if required credentials are present in the environment."""
        pass

    @abstractmethod
    def send(self, payload: AlertPayload) -> bool:
        """Sends an emergency alert payload through this notification channel.

        Args:
            payload: Incident details and dispatch assessment.

        Returns:
            bool: True if alert dispatch succeeded, False otherwise.
        """
        pass


class WebPushNotifier(BaseNotifier):
    """Dispatches Web Push (PWA) notifications to subscribed operator browsers and mobile devices."""

    def __init__(self):
        self._keys = None

    @property
    def name(self) -> str:
        return "webpush"

    def is_configured(self) -> bool:
        keys = get_or_generate_vapid_keys()
        return bool(keys and keys.get("private_key") and keys.get("public_key"))

    def send(self, payload: AlertPayload) -> bool:
        """Sends Web Push notifications to all active subscribers with rich actions and disclaimer."""
        is_demo = is_demo_incident(payload.incident_id, payload.metadata) or payload.is_demo
        if payload.dry_run or is_demo:
            logger.info(
                "[WebPushNotifier - DRY RUN/DEMO] Simulated push dispatch for Incident %s (%s)",
                payload.incident_id,
                payload.class_name,
            )
            return True

        if not self.is_configured():
            logger.warning("[WebPushNotifier] Skipped: VAPID keys not configured.")
            return False

        keys = get_or_generate_vapid_keys()
        private_key = keys["private_key"]
        claims = {"sub": keys["claims_sub"]}

        # Retrieve active push subscriptions from SQLite
        subscriptions = []
        try:
            from sqlmodel import Session, select
            from backend.app.db.database import engine
            from backend.app.db.models import WebPushSubscription

            with Session(engine) as session:
                subscriptions = session.exec(
                    select(WebPushSubscription).where(WebPushSubscription.is_active == True)
                ).all()
        except Exception as db_err:
            logger.warning("[WebPushNotifier] Could not fetch subscriptions from DB: %s", db_err)

        if not subscriptions:
            logger.info("[WebPushNotifier] No active push subscriptions registered. Notification recorded.")
            return True

        import pywebpush

        tier_prefix = f"[TIER {payload.tier} ESCALATION] " if payload.is_escalation else ""
        notification_title = f"🚨 {tier_prefix}{payload.class_name.upper()} Detected"
        notification_body = (
            f"Severity: {payload.severity.upper()} ({payload.severity_score}/100)\n"
            f"{payload.report_text[:120]}...\n\n"
            f"{DISCLAIMER_TEXT}"
        )

        app_url = getattr(config, "app_base_url", "http://localhost:8000")
        push_data = json.dumps({
            "title": notification_title,
            "body": notification_body,
            "icon": "/vite.svg",
            "image": payload.keyframe_url or (f"{app_url}/media/keyframes/{os.path.basename(payload.image_path)}" if payload.image_path else None),
            "tag": f"incident-{payload.incident_id}",
            "data": {
                "incident_id": payload.incident_id,
                "ack_url": payload.ack_url,
                "map_url": payload.map_url,
                "severity": payload.severity,
                "severity_score": payload.severity_score,
                "disclaimer": DISCLAIMER_TEXT,
            },
        }, ensure_ascii=False)

        success_count = 0
        for sub in subscriptions:
            sub_info = {
                "endpoint": sub.endpoint,
                "keys": {
                    "p256dh": sub.p256dh,
                    "auth": sub.auth,
                },
            }
            try:
                pywebpush.webpush(
                    subscription_info=sub_info,
                    data=push_data,
                    vapid_private_key=private_key,
                    vapid_claims=claims,
                    ttl=3600,
                )
                success_count += 1
            except pywebpush.WebPushException as wp_err:
                status_code = getattr(getattr(wp_err, "response", None), "status_code", None)
                if status_code in (404, 410):
                    # Subscription expired or unsubscribed
                    logger.info("Web push subscription expired (%d), deactivating: %s", status_code, sub.endpoint[:30])
                    try:
                        from sqlmodel import Session
                        from backend.app.db.database import engine
                        from backend.app.db.models import WebPushSubscription
                        with Session(engine) as session:
                            db_sub = session.get(WebPushSubscription, sub.id)
                            if db_sub:
                                db_sub.is_active = False
                                session.add(db_sub)
                                session.commit()
                    except Exception:
                        pass
                else:
                    logger.warning("WebPush dispatch error for %s: %s", sub.endpoint[:30], wp_err)
            except Exception as e:
                logger.warning("Unexpected WebPush dispatch error: %s", e)

        logger.info("WebPush alert dispatched to %d / %d subscribers for %s", success_count, len(subscriptions), payload.incident_id)
        return True


class TelegramNotifier(BaseNotifier):
    """Dispatches keyframe images and rich dispatch reports via Telegram Bot HTTP API."""

    def __init__(
        self,
        bot_token: Optional[str] = None,
        chat_id: Optional[str] = None,
        timeout: float = 10.0,
    ):
        self.bot_token = bot_token or os.getenv("TELEGRAM_BOT_TOKEN") or getattr(config, "telegram_bot_token", None)
        self.chat_id = chat_id or os.getenv("TELEGRAM_CHAT_ID") or getattr(config, "telegram_chat_id", None)
        self.timeout = timeout

    @property
    def name(self) -> str:
        return "telegram"

    def is_configured(self) -> bool:
        return bool(self.bot_token and self.chat_id)

    def send(self, payload: AlertPayload) -> bool:
        """Sends an incident photo and formatted Markdown caption to Telegram."""
        is_demo = is_demo_incident(payload.incident_id, payload.metadata) or payload.is_demo
        if payload.dry_run or is_demo:
            logger.info(
                "[TelegramNotifier - DRY RUN/DEMO] Simulated dispatch to chat %s: Incident %s (%s)",
                self.chat_id or "DEMO_CHAT",
                payload.incident_id,
                payload.class_name,
            )
            return True

        if not self.is_configured():
            logger.warning(
                "[TelegramNotifier] Skipped: TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID not configured in .env."
            )
            return False

        base_url = f"https://api.telegram.org/bot{self.bot_token}"

        escalation_hdr = "⚡ *[ESCALATION TIER 2 - UNACKNOWLEDGED]*\n" if payload.is_escalation else ""
        caption_lines = [
            f"🚨 *EMERGENCY DISPATCH: {payload.class_name.upper()}*",
            escalation_hdr,
            f"📊 *Severity:* {payload.severity.upper()} (Score: {payload.severity_score}/100)",
            f"🕒 *Time:* {payload.timestamp}",
        ]
        if payload.camera_id or payload.location_name:
            caption_lines.append(f"📍 *Location:* {payload.location_name or payload.camera_id}")
        if payload.map_url:
            caption_lines.append(f"🗺️ [Open Google Maps]({payload.map_url})")
        if payload.ack_url:
            caption_lines.append(f"✅ [Acknowledge Incident (One-Click)]({payload.ack_url})")

        caption_lines.append(f"\n{payload.report_text[:400]}")
        caption_lines.append(f"\n{DISCLAIMER_TEXT}")

        caption_text = "\n".join([line for line in caption_lines if line])
        # Telegram captions max 1024 characters
        if len(caption_text) > 1024:
            caption_text = caption_text[:1020] + "..."

        try:
            if payload.image_path and os.path.exists(payload.image_path):
                send_photo_url = f"{base_url}/sendPhoto"
                with open(payload.image_path, "rb") as img_file:
                    files = {"photo": img_file}
                    data = {
                        "chat_id": self.chat_id,
                        "caption": caption_text,
                        "parse_mode": "Markdown",
                    }
                    resp = requests.post(send_photo_url, data=data, files=files, timeout=self.timeout)
                    if resp.status_code == 200:
                        logger.info("Telegram alert sent successfully with photo for %s", payload.incident_id)
                        return True
                    else:
                        logger.warning("Telegram sendPhoto failed (status %d): %s", resp.status_code, resp.text)

            # Fallback to standard text message if photo send failed or no photo present
            send_msg_url = f"{base_url}/sendMessage"
            data = {
                "chat_id": self.chat_id,
                "text": caption_text,
                "parse_mode": "Markdown",
            }
            resp = requests.post(send_msg_url, data=data, timeout=self.timeout)
            if resp.status_code == 200:
                logger.info("Telegram text alert sent successfully for %s", payload.incident_id)
                return True
            else:
                logger.error("Telegram sendMessage failed (status %d): %s", resp.status_code, resp.text)
                return False

        except Exception as exc:
            logger.error("Telegram alert network failure for %s: %s", payload.incident_id, exc)
            return False


class SMTPNotifier(BaseNotifier):
    """Dispatches emergency dispatch reports and keyframes via standard SMTP email."""

    def __init__(
        self,
        host: Optional[str] = None,
        port: Optional[int] = None,
        user: Optional[str] = None,
        password: Optional[str] = None,
        sender: Optional[str] = None,
        recipients: Optional[List[str]] = None,
        use_tls: Optional[bool] = None,
        timeout: float = 10.0,
    ):
        self.host = host or os.getenv("SMTP_HOST") or getattr(config, "smtp_host", None)
        self.port = port or int(os.getenv("SMTP_PORT", str(getattr(config, "smtp_port", 587))))
        self.user = user or os.getenv("SMTP_USER") or getattr(config, "smtp_user", None)
        self.password = password or os.getenv("SMTP_PASSWORD") or getattr(config, "smtp_password", None)
        self.sender = sender or os.getenv("SMTP_FROM") or self.user or "alerts@acs-surveillance.local"

        to_val = os.getenv("SMTP_TO") or getattr(config, "smtp_to", None) or ""
        raw_recipients = recipients if recipients is not None else ([r.strip() for r in to_val.split(",") if r.strip()])
        self.recipients = [r.strip() for r in raw_recipients if r.strip()]
        self.use_tls = (
            use_tls
            if use_tls is not None
            else (os.getenv("SMTP_USE_TLS", "true").lower() in ("true", "1", "yes"))
        )
        self.timeout = timeout

    @property
    def name(self) -> str:
        return "email"

    def is_configured(self) -> bool:
        return bool(self.host and self.recipients)

    def send(self, payload: AlertPayload) -> bool:
        """Sends an HTML/Plaintext email with attached keyframe, map link, one-click ack, and disclaimer."""
        is_demo = is_demo_incident(payload.incident_id, payload.metadata) or payload.is_demo
        if payload.dry_run or is_demo:
            logger.info(
                "[SMTPNotifier - DRY RUN/DEMO] Simulated email to %s: Incident %s (%s)",
                ", ".join(self.recipients) or "DEMO_RECIPIENT",
                payload.incident_id,
                payload.class_name,
            )
            return True

        if not self.is_configured():
            logger.warning("[SMTPNotifier] Skipped: SMTP_HOST or SMTP_TO not configured in .env.")
            return False

        try:
            msg = MIMEMultipart("alternative")
            esc_prefix = "[ESCALATION TIER 2] " if payload.is_escalation else ""
            msg["Subject"] = f"{esc_prefix}[ACS DISPATCH ALERT] {payload.class_name.upper()} - {payload.incident_id} ({payload.severity.upper()})"
            msg["From"] = self.sender
            msg["To"] = ", ".join(self.recipients)

            # Plain text body
            text_lines = [
                f"ACS Automated Surveillance Alert - {payload.incident_id}",
                f"Class: {payload.class_name.upper()}",
                f"Severity: {payload.severity.upper()} ({payload.severity_score}/100)",
                f"Timestamp: {payload.timestamp}",
            ]
            if payload.location_name or payload.camera_id:
                text_lines.append(f"Location: {payload.location_name or payload.camera_id}")
            if payload.map_url:
                text_lines.append(f"Map: {payload.map_url}")
            if payload.ack_url:
                text_lines.append(f"Acknowledge Incident: {payload.ack_url}")
            text_lines.append("\nReport Summary:")
            text_lines.append(payload.report_text)
            text_lines.append(f"\n{DISCLAIMER_TEXT}")
            plain_body = "\n".join(text_lines)

            # Rich HTML body
            map_button = f'<a href="{payload.map_url}" style="display:inline-block;padding:10px 18px;background:#1e293b;color:#38bdf8;text-decoration:none;border-radius:6px;font-weight:bold;margin-right:10px;">🗺️ View Location on Map</a>' if payload.map_url else ""
            ack_button = f'<a href="{payload.ack_url}" style="display:inline-block;padding:10px 18px;background:#059669;color:#ffffff;text-decoration:none;border-radius:6px;font-weight:bold;">✅ Acknowledge Incident</a>' if payload.ack_url else ""

            html_body = f"""<!DOCTYPE html>
<html>
<body style="font-family:-apple-system,BlinkMacSystemFont,Segoe UI,Roboto,Helvetica,Arial,sans-serif;background-color:#0f172a;color:#f8fafc;padding:24px;margin:0;">
  <div style="max-width:620px;margin:0 auto;background:#1e293b;border-radius:12px;overflow:hidden;border:1px solid #334155;">
    <div style="background:#dc2626;padding:16px 20px;">
      <h2 style="margin:0;color:#ffffff;font-size:18px;">🚨 ACS EMERGENCY DISPATCH ALERT</h2>
      <p style="margin:4px 0 0;color:#fee2e2;font-size:13px;">Incident: {payload.incident_id} | Class: {payload.class_name.upper()}</p>
    </div>
    <div style="padding:20px;">
      <div style="display:flex;justify-content:space-between;margin-bottom:16px;">
        <span style="background:#e11d48;color:#fff;padding:4px 10px;border-radius:4px;font-weight:bold;font-size:12px;">Severity: {payload.severity.upper()} ({payload.severity_score}/100)</span>
        <span style="color:#94a3b8;font-size:12px;">{payload.timestamp}</span>
      </div>
      <div style="background:#0f172a;padding:14px;border-radius:8px;margin-bottom:16px;border:1px solid #334155;">
        <pre style="white-space:pre-wrap;color:#e2e8f0;font-size:13px;line-height:1.5;margin:0;font-family:inherit;">{payload.report_text}</pre>
      </div>
      <div style="margin:20px 0;">
        {map_button}
        {ack_button}
      </div>
      <div style="background:#451a03;border:1px solid #b45309;padding:12px;border-radius:8px;margin-top:20px;">
        <p style="margin:0;color:#fef3c7;font-size:12px;line-height:1.4;">{DISCLAIMER_TEXT}</p>
      </div>
    </div>
  </div>
</body>
</html>"""

            msg.attach(MIMEText(plain_body, "plain", "utf-8"))
            msg.attach(MIMEText(html_body, "html", "utf-8"))

            # Keyframe attachment
            if payload.image_path and os.path.exists(payload.image_path):
                try:
                    with open(payload.image_path, "rb") as img_file:
                        img_data = img_file.read()
                    img_part = MIMEImage(img_data)
                    filename = os.path.basename(payload.image_path)
                    img_part.add_header("Content-Disposition", "attachment", filename=filename)
                    msg.attach(img_part)
                except Exception as img_err:
                    logger.warning("Could not attach keyframe to email: %s", img_err)

            with smtplib.SMTP(self.host, self.port, timeout=self.timeout) as server:
                if self.use_tls:
                    server.starttls()
                if self.user and self.password:
                    server.login(self.user, self.password)
                server.send_message(msg)

            logger.info("SMTP email alert sent successfully to %s for %s", self.recipients, payload.incident_id)
            return True

        except Exception as exc:
            logger.error("SMTP alert delivery failure for %s: %s", payload.incident_id, exc)
            return False


class ConsoleNotifier(BaseNotifier):
    """Outputs incident alert summaries to application standard logger."""

    @property
    def name(self) -> str:
        return "console"

    def is_configured(self) -> bool:
        return True

    def send(self, payload: AlertPayload) -> bool:
        dry_prefix = "[DRY RUN] " if payload.dry_run else ""
        logger.warning(
            "%s[EMERGENCY DISPATCH ALERT] Incident %s: %s (Severity: %s, Score: %s) at %s | Map: %s | Ack: %s | Image: %s\n%s",
            dry_prefix,
            payload.incident_id,
            payload.class_name,
            payload.severity,
            payload.severity_score,
            payload.timestamp,
            payload.map_url or "N/A",
            payload.ack_url or "N/A",
            payload.image_path or "N/A",
            DISCLAIMER_TEXT,
        )
        return True


class AlertManager:
    """Manages pluggable notification channels, rate-limiting, de-duplication, and safety interlocks."""

    def __init__(self, dry_run: bool = False):
        self.dry_run = dry_run
        self.storm_limiter = storm_rate_limiter
        self.notifiers: Dict[str, BaseNotifier] = {
            "console": ConsoleNotifier(),
            "telegram": TelegramNotifier(),
            "email": SMTPNotifier(),
            "webpush": WebPushNotifier(),
        }

    def register_notifier(self, notifier: BaseNotifier) -> None:
        """Registers a custom notifier backend."""
        self.notifiers[notifier.name] = notifier

    def send(
        self,
        payload: AlertPayload,
        channels: Optional[List[str]] = None,
    ) -> Dict[str, bool]:
        """Dispatches an alert payload across selected channels with safety guarantees."""
        selected_channels = channels or list(self.notifiers.keys())
        results: Dict[str, bool] = {}

        # 1. HARD DEMO & DRY-RUN SAFETY INTERLOCK:
        # DEMO incidents are mathematically barred from real external channels
        is_demo = is_demo_incident(payload.incident_id, payload.metadata) or payload.is_demo
        effective_dry_run = self.dry_run or payload.dry_run or is_demo

        # Ensure URLs are prepared
        app_url = getattr(config, "app_base_url", "http://localhost:8000")
        ack_token = payload.ack_token or secrets.token_urlsafe(24)
        ack_url = payload.ack_url or f"{app_url}/api/alerts/ack/{ack_token}"
        map_url = payload.map_url
        if not map_url and payload.location_lat is not None and payload.location_lng is not None:
            map_url = f"https://maps.google.com/?q={payload.location_lat},{payload.location_lng}"
        elif not map_url and payload.location_name:
            map_url = f"https://maps.google.com/?q={quote_plus(payload.location_name)}"

        effective_payload = AlertPayload(
            incident_id=payload.incident_id,
            class_name=payload.class_name,
            timestamp=payload.timestamp,
            confidence=payload.confidence,
            report_text=payload.report_text,
            image_path=payload.image_path,
            report_filename=payload.report_filename,
            dry_run=effective_dry_run,
            severity=payload.severity,
            severity_score=payload.severity_score,
            camera_id=payload.camera_id,
            location_name=payload.location_name,
            location_lat=payload.location_lat,
            location_lng=payload.location_lng,
            ack_token=ack_token,
            ack_url=ack_url,
            map_url=map_url,
            keyframe_url=payload.keyframe_url,
            tier=payload.tier,
            is_escalation=payload.is_escalation,
            is_demo=is_demo,
            metadata=payload.metadata,
        )

        # 2. STORM RATE LIMITING:
        # Suppress alert bursts per camera or geographic location during storm conditions
        rate_key = (
            payload.camera_id
            or payload.location_name
            or (
                f"{payload.location_lat:.3f},{payload.location_lng:.3f}"
                if (payload.location_lat is not None and payload.location_lng is not None)
                else None
            )
        )
        if rate_key and not payload.is_escalation and self.storm_limiter.should_rate_limit(rate_key):
            logger.warning(
                "Storm rate-limiting activated for location '%s'. Suppressing rapid alert burst for %s.",
                rate_key,
                payload.incident_id,
            )
            for channel_name in selected_channels:
                results[channel_name] = False
                record_alert_delivery(
                    incident_id=payload.incident_id,
                    channel=channel_name,
                    tier=payload.tier,
                    status="suppressed",
                    details=f"Suppressed by storm rate-limiter for location {rate_key}",
                )
            return results

        # 3. CHANNEL DISPATCH & DE-DUPLICATION:
        for channel_name in selected_channels:
            notifier = self.notifiers.get(channel_name)
            if not notifier:
                logger.warning("Requested alert channel '%s' is not registered.", channel_name)
                results[channel_name] = False
                continue

            # De-duplication per incident, channel, and tier
            if check_existing_delivery(payload.incident_id, channel_name, payload.tier):
                logger.info("Alert already delivered to channel '%s' for incident %s (Tier %d). Skipping duplicate.", channel_name, payload.incident_id, payload.tier)
                results[channel_name] = True
                continue

            try:
                success = notifier.send(effective_payload)
                results[channel_name] = success
                delivery_status = "dry_run" if effective_dry_run else ("sent" if success else "failed")
                details_msg = "Dispatched in dry-run mode (Demo interlock)" if (is_demo and not self.dry_run) else ("Dry-run mode active" if effective_dry_run else ("Delivery successful" if success else "Channel delivery failed"))
                record_alert_delivery(
                    incident_id=payload.incident_id,
                    channel=channel_name,
                    tier=payload.tier,
                    status=delivery_status,
                    details=details_msg,
                )
            except Exception as exc:
                logger.error("Unhandled error dispatching via channel '%s': %s", channel_name, exc)
                results[channel_name] = False
                record_alert_delivery(
                    incident_id=payload.incident_id,
                    channel=channel_name,
                    tier=payload.tier,
                    status="failed",
                    details=str(exc),
                )

        return results


# Global alert manager instance
alert_manager = AlertManager()


def dispatch_alert(
    payload: AlertPayload,
    channels: Optional[List[str]] = None,
    dry_run: Optional[bool] = None,
) -> Dict[str, bool]:
    """Top-level functional dispatcher conforming to original interface."""
    if dry_run is not None:
        payload.dry_run = dry_run
    return alert_manager.send(payload, channels=channels)


def send_incident_alert(
    incident: Any,
    report_text: str,
    image_path: Optional[str] = None,
    channels: Optional[List[str]] = None,
    dry_run: bool = False,
    tier: int = 1,
    is_escalation: bool = False,
) -> Dict[str, bool]:
    """Dispatches an incident alert with keyframe, map link, severity, and one-click ack."""
    inc_id = getattr(incident, "incident_id", "INC-DEMO")
    cls_name = getattr(incident, "class_name", "traffic_accident")
    ts = getattr(incident, "start_timestamp", "2026-10-08 12:00:00 UTC")
    conf = getattr(incident, "max_confidence", 1.0)
    img = image_path or getattr(incident, "image_path", None)
    cam_id = getattr(incident, "camera_id", None)
    sev = getattr(incident, "severity", "medium") or "medium"
    sev_score = getattr(incident, "severity_score", 50) or 50
    is_demo = getattr(incident, "is_demo", False)

    v_status = getattr(incident, "verification_status", None)
    if v_status == "suppressed":
        logger.info("Incident %s is marked 'suppressed'. Alert suppressed.", inc_id)
        return {"suppressed": True}
    if v_status == "needs_review":
        logger.info("Incident %s requires human review. Alert withheld.", inc_id)
        return {"held_for_review": True}

    final_text = report_text
    if v_status == "unverified" and "[UNVERIFIED" not in final_text:
        final_text = f"[UNVERIFIED - YOLO ONLY: PENDING VERIFICATION]\n{report_text}"

    # Lookup camera coordinates and ack token from DB if available
    loc_lat = None
    loc_lng = None
    loc_name = None
    ack_token = getattr(incident, "ack_token", None)

    try:
        from sqlmodel import Session
        from backend.app.db.database import engine
        from backend.app.db.models import Camera, Incident

        with Session(engine) as session:
            db_inc = session.get(Incident, inc_id)
            if db_inc:
                if not ack_token and not db_inc.ack_token:
                    db_inc.ack_token = secrets.token_urlsafe(24)
                    session.add(db_inc)
                    session.commit()
                ack_token = db_inc.ack_token or ack_token
                is_demo = is_demo or getattr(db_inc, "is_demo", False)

            if cam_id:
                db_cam = session.get(Camera, cam_id)
                if db_cam:
                    loc_name = db_cam.name
                    loc_lat = db_cam.location_lat
                    loc_lng = db_cam.location_lng
    except Exception as db_e:
        logger.debug("Could not query DB for camera/incident info: %s", db_e)

    app_url = getattr(config, "app_base_url", "http://localhost:8000")
    if not ack_token:
        ack_token = secrets.token_urlsafe(24)
    ack_url = f"{app_url}/api/alerts/ack/{ack_token}"

    map_url = None
    if loc_lat is not None and loc_lng is not None:
        map_url = f"https://maps.google.com/?q={loc_lat},{loc_lng}"
    elif loc_name:
        map_url = f"https://maps.google.com/?q={quote_plus(loc_name)}"

    keyframe_url = getattr(incident, "keyframe_url", None)
    if not keyframe_url and img:
        keyframe_url = f"/media/keyframes/{os.path.basename(img)}"

    payload = AlertPayload(
        incident_id=inc_id,
        class_name=cls_name,
        timestamp=ts,
        confidence=conf,
        report_text=final_text,
        image_path=img,
        dry_run=dry_run,
        severity=sev,
        severity_score=sev_score,
        camera_id=cam_id,
        location_name=loc_name,
        location_lat=loc_lat,
        location_lng=loc_lng,
        ack_token=ack_token,
        ack_url=ack_url,
        map_url=map_url,
        keyframe_url=keyframe_url,
        tier=tier,
        is_escalation=is_escalation,
        is_demo=is_demo,
        metadata={"verification_status": v_status or "auto_confirmed", "is_demo": is_demo},
    )
    return alert_manager.send(payload, channels=channels)


def send_test_alert(
    dry_run: bool = False,
    lang: str = "en",
    channels: Optional[List[str]] = None,
) -> Dict[str, bool]:
    """Dispatches a synthetic test alert across channels (forced dry_run for demo safety)."""
    if lang == "ar":
        test_text = (
            "[تقرير إرسال طوارئ تجريبي - نظام ACS]\n"
            "الوقت: 2026-10-08 12:00:00 UTC | مستوى الخطورة: متوسطة (50/100)\n"
            "المركبات المعنية: سيارة ركاب (تجريبي)\n"
            "نقطة الاصطدام: كاميرا المراقبة رقم 04\n"
            "الأضرار والمخاطر: فحص اتصال قنوات التنبيه بنجاح\n"
            "الوحدات الموصى بإرسالها: شرطة (Police), إسعاف (EMS)"
        )
    else:
        test_text = (
            "[ACS TEST DISPATCH ALERT - CONNECTIVITY CHECK]\n"
            "Timestamp: 2026-10-08 12:00:00 UTC | Severity: MEDIUM (Score: 50/100)\n"
            "Vehicles Involved: Test Vehicle (Simulation)\n"
            "Collision Point: CCTV Camera Zone 04\n"
            "Observed Damage: Alert connectivity validation test\n"
            "Recommended Units: POLICE, EMS, TOW"
        )

    sample_keyframe = "report_images/INC-3-001_keyframe.jpg"
    img_path = sample_keyframe if os.path.exists(sample_keyframe) else None

    payload = AlertPayload(
        incident_id="TEST-ALERT-001",
        class_name="connectivity_test",
        timestamp="2026-10-08 12:00:00 UTC",
        confidence=1.0,
        report_text=test_text,
        image_path=img_path,
        dry_run=True,  # Safety interlock: Test alerts are always dry_run
        severity="medium",
        severity_score=50,
        camera_id="cam_test_04",
        location_name="Zone 04 Test Intersection",
        is_demo=True,
    )
    return alert_manager.send(payload, channels=channels)
