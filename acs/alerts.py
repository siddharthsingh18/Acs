"""
acs.alerts
Pluggable notification alert system for emergency dispatch services.
Supports:
- Telegram Bot HTTP API (sendPhoto / sendMessage)
- SMTP Email (MIMEMultipart with embedded keyframe image)
- Console logging
- Dry-run mode for demos and tests
- Automatic .env credential loading
- Robust non-blocking error handling that never crashes the detection pipeline
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from email.mime.image import MIMEImage
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
import logging
import os
import smtplib
from typing import Any, Dict, List, Optional
from dotenv import load_dotenv
import requests

from acs.config import config

load_dotenv()
logger = logging.getLogger("acs.alerts")


@dataclass
class AlertPayload:
    """Encapsulates incident details and keyframe attachments for dispatch alerts."""

    incident_id: str
    class_name: str
    timestamp: str
    confidence: float
    report_text: str
    image_path: Optional[str] = None
    report_filename: Optional[str] = None
    dry_run: bool = False
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


class TelegramNotifier(BaseNotifier):
    """Dispatches keyframe images and dispatch reports via Telegram Bot HTTP API."""

    def __init__(
        self,
        bot_token: Optional[str] = None,
        chat_id: Optional[str] = None,
        timeout: float = 10.0,
    ):
        """Initializes Telegram notifier with credentials.

        Args:
            bot_token: Optional override for TELEGRAM_BOT_TOKEN.
            chat_id: Optional override for TELEGRAM_CHAT_ID.
            timeout: HTTP request timeout in seconds.
        """
        self.bot_token = bot_token or os.getenv("TELEGRAM_BOT_TOKEN") or getattr(config, "telegram_bot_token", None)
        self.chat_id = chat_id or os.getenv("TELEGRAM_CHAT_ID") or getattr(config, "telegram_chat_id", None)
        self.timeout = timeout

    @property
    def name(self) -> str:
        return "telegram"

    def is_configured(self) -> bool:
        return bool(self.bot_token and self.chat_id)

    def send(self, payload: AlertPayload) -> bool:
        """Sends an incident photo and text caption to the configured Telegram chat."""
        if payload.dry_run:
            logger.info(
                "[TelegramNotifier - DRY RUN] Simulated dispatch to chat %s: Incident %s (%s)",
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
        caption_text = f"🚨 *EMERGENCY DISPATCH: {payload.class_name.upper()}*\n\n{payload.report_text}"
        # Telegram captions max 1024 characters
        if len(caption_text) > 1024:
            caption_text = caption_text[:1020] + "..."

        try:
            # 1. Attempt to send with keyframe photo if image path is valid
            if payload.image_path and os.path.exists(payload.image_path):
                send_photo_url = f"{base_url}/sendPhoto"
                with open(payload.image_path, "rb") as img_file:
                    files = {"photo": img_file}
                    data = {
                        "chat_id": self.chat_id,
                        "caption": caption_text,
                    }
                    resp = requests.post(send_photo_url, data=data, files=files, timeout=self.timeout)
                    if resp.status_code == 200:
                        logger.info("Telegram alert sent successfully with photo for %s", payload.incident_id)
                        return True
                    else:
                        logger.warning("Telegram sendPhoto failed (status %d): %s", resp.status_code, resp.text)

            # 2. Fallback to standard text message if photo send failed or no photo present
            send_msg_url = f"{base_url}/sendMessage"
            data = {
                "chat_id": self.chat_id,
                "text": caption_text,
            }
            resp = requests.post(send_msg_url, data=data, timeout=self.timeout)
            if resp.status_code == 200:
                logger.info("Telegram text alert sent successfully for %s", payload.incident_id)
                return True
            else:
                logger.error("Telegram sendMessage failed (status %d): %s", resp.status_code, resp.text)
                return False

        except Exception as exc:
            # Pipeline integrity: Log failure and continue without raising
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
        """Initializes SMTP email notifier with credentials."""
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
        """Sends an HTML/Plaintext email with attached incident keyframe."""
        if payload.dry_run:
            logger.info(
                "[SMTPNotifier - DRY RUN] Simulated email to %s: Incident %s (%s)",
                ", ".join(self.recipients) or "DEMO_RECIPIENT",
                payload.incident_id,
                payload.class_name,
            )
            return True

        if not self.is_configured():
            logger.warning(
                "[SMTPNotifier] Skipped: SMTP_HOST or SMTP_TO not configured in .env."
            )
            return False

        try:
            msg = MIMEMultipart()
            msg["Subject"] = f"[ACS DISPATCH ALERT] {payload.class_name.upper()} - {payload.incident_id}"
            msg["From"] = self.sender
            msg["To"] = ", ".join(self.recipients)

            # Plain text body
            body_text = f"ACS Automated Surveillance Alert\n\n{payload.report_text}\n"
            msg.attach(MIMEText(body_text, "plain", "utf-8"))

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

            # SMTP Connection & Delivery
            with smtplib.SMTP(self.host, self.port, timeout=self.timeout) as server:
                if self.use_tls:
                    server.starttls()
                if self.user and self.password:
                    server.login(self.user, self.password)
                server.send_message(msg)

            logger.info("SMTP email alert sent successfully to %s for %s", self.recipients, payload.incident_id)
            return True

        except Exception as exc:
            # Pipeline integrity: Log failure and continue without raising
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
            "%s[EMERGENCY DISPATCH ALERT] Incident %s: %s (Confidence: %.2f) at %s | Image: %s",
            dry_prefix,
            payload.incident_id,
            payload.class_name,
            payload.confidence,
            payload.timestamp,
            payload.image_path or "N/A",
        )
        return True


# Default Alert Manager holding registered backends
class AlertManager:
    """Manages pluggable notification channels and dispatches emergency alerts."""

    def __init__(self, dry_run: bool = False):
        self.dry_run = dry_run
        self.notifiers: Dict[str, BaseNotifier] = {
            "console": ConsoleNotifier(),
            "telegram": TelegramNotifier(),
            "email": SMTPNotifier(),
        }

    def register_notifier(self, notifier: BaseNotifier) -> None:
        """Registers a custom notifier backend."""
        self.notifiers[notifier.name] = notifier

    def send(
        self,
        payload: AlertPayload,
        channels: Optional[List[str]] = None,
    ) -> Dict[str, bool]:
        """Dispatches an alert payload across selected channels without crashing."""
        selected_channels = channels or list(self.notifiers.keys())
        results: Dict[str, bool] = {}

        # Respect dry-run flag from manager or payload
        effective_payload = payload
        if self.dry_run and not payload.dry_run:
            effective_payload = AlertPayload(
                incident_id=payload.incident_id,
                class_name=payload.class_name,
                timestamp=payload.timestamp,
                confidence=payload.confidence,
                report_text=payload.report_text,
                image_path=payload.image_path,
                report_filename=payload.report_filename,
                dry_run=True,
                metadata=payload.metadata,
            )

        for channel_name in selected_channels:
            notifier = self.notifiers.get(channel_name)
            if not notifier:
                logger.warning("Requested alert channel '%s' is not registered.", channel_name)
                results[channel_name] = False
                continue

            try:
                success = notifier.send(effective_payload)
                results[channel_name] = success
            except Exception as exc:
                # Absolute safety: Never let an alert channel error crash the pipeline
                logger.error("Unhandled error dispatching via channel '%s': %s", channel_name, exc)
                results[channel_name] = False

        return results


# Global alert manager instance
alert_manager = AlertManager()


def dispatch_alert(
    payload: AlertPayload,
    channels: Optional[List[str]] = None,
    dry_run: Optional[bool] = None,
) -> Dict[str, bool]:
    """Top-level functional dispatcher conforming to original interface.

    Args:
        payload: AlertPayload to send.
        channels: Optional list of backend names ('telegram', 'email', 'console').
        dry_run: Optional override for simulation mode.

    Returns:
        Dict[str, bool]: Delivery success per channel.
    """
    if dry_run is not None:
        payload.dry_run = dry_run
    return alert_manager.send(payload, channels=channels)


def send_incident_alert(
    incident: Any,
    report_text: str,
    image_path: Optional[str] = None,
    channels: Optional[List[str]] = None,
    dry_run: bool = False,
) -> Dict[str, bool]:
    """Convenience helper to dispatch a closed incident and its dispatch report text.

    Args:
        incident: AccidentIncident instance or object with metadata attributes.
        report_text: Multimodal dispatch assessment text.
        image_path: Path to keyframe image.
        channels: Target channels (default: all configured).
        dry_run: Whether to simulate delivery.

    Returns:
        Dict[str, bool]: Channel delivery status mapping.
    """
    inc_id = getattr(incident, "incident_id", "INC-DEMO")
    cls_name = getattr(incident, "class_name", "traffic_accident")
    ts = getattr(incident, "start_timestamp", "2026-10-08 12:00:00 UTC")
    conf = getattr(incident, "max_confidence", 1.0)
    img = image_path or getattr(incident, "image_path", None)

    payload = AlertPayload(
        incident_id=inc_id,
        class_name=cls_name,
        timestamp=ts,
        confidence=conf,
        report_text=report_text,
        image_path=img,
        dry_run=dry_run,
    )
    return alert_manager.send(payload, channels=channels)


def send_test_alert(
    dry_run: bool = False,
    lang: str = "en",
    channels: Optional[List[str]] = None,
) -> Dict[str, bool]:
    """Dispatches a synthetic test alert to verify Telegram and SMTP connectivity.

    Args:
        dry_run: Whether to simulate sending.
        lang: Language for test report text ('en' or 'ar').
        channels: Target notification channels.

    Returns:
        Dict[str, bool]: Status mapping per channel.
    """
    if lang == "ar":
        test_text = (
            "[تقرير إرسال طوارئ تجريبي - نظام ACS]\n"
            "الوقت: 2026-10-08 12:00:00 UTC | مستوى الخطورة: متوسطة\n"
            "المركبات المعنية: سيارة ركاب (تجريبي)\n"
            "نقطة الاصطدام: كاميرا المراقبة رقم 04\n"
            "الأضرار والمخاطر: فحص اتصال قنوات التنبيه بنجاح\n"
            "الوحدات الموصى بإرسالها: شرطة (Police), إسعاف (EMS)"
        )
    else:
        test_text = (
            "[ACS TEST DISPATCH ALERT - CONNECTIVITY CHECK]\n"
            "Timestamp: 2026-10-08 12:00:00 UTC | Severity: MEDIUM\n"
            "Vehicles Involved: Test Vehicle (Simulation)\n"
            "Collision Point: CCTV Camera Zone 04\n"
            "Observed Damage: Alert connectivity validation test\n"
            "Recommended Units: POLICE, EMS, TOW"
        )

    # Use existing sample image if present
    sample_keyframe = "report_images/INC-3-001_keyframe.jpg"
    img_path = sample_keyframe if os.path.exists(sample_keyframe) else None

    payload = AlertPayload(
        incident_id="TEST-ALERT-001",
        class_name="connectivity_test",
        timestamp="2026-10-08 12:00:00 UTC",
        confidence=1.0,
        report_text=test_text,
        image_path=img_path,
        dry_run=dry_run,
    )
    return alert_manager.send(payload, channels=channels)
