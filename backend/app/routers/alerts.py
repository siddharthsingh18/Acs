"""
backend.app.routers.alerts
Route handlers for emergency alert dispatching, connectivity testing, Web Push, one-click ack, and settings.
"""
from datetime import datetime
import hashlib
import logging
import time
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, Request, Response, status
from fastapi.responses import HTMLResponse, JSONResponse
from sqlmodel import Session, select

from acs.alerts import (
    DISCLAIMER_TEXT,
    alert_manager,
    get_or_generate_vapid_keys,
    send_test_alert,
)
from acs.config import config
from backend.app.db.database import get_session
from backend.app.db.models import AlertDelivery, Incident, User, WebPushSubscription
from backend.app.deps import extract_client_ip, get_current_user_optional, require_admin, require_operator, require_viewer
from backend.app.services.alert_escalation_service import check_and_escalate_unacknowledged_incidents
from backend.app.services.audit_service import log_audit_event
from backend.app.schemas import (
    AlertSettingsResponse,
    AlertSettingsUpdateRequest,
    AlertTestRequest,
    AlertTestResponse,
    ChannelStatusInfo,
    WebPushSubscribeRequest,
    WebPushUnsubscribeRequest,
)

logger = logging.getLogger("acs.backend.routers.alerts")
router = APIRouter(prefix="/api/alerts", tags=["Alerts"])


@router.post("/test", response_model=AlertTestResponse)
def send_test_alert_endpoint(
    body: AlertTestRequest,
    request: Request,
    session: Session = Depends(get_session),
    admin_user: User = Depends(require_admin),
) -> AlertTestResponse:
    """Dispatches a test emergency alert to verify Telegram, SMTP, and WebPush connectivity (Admin only)."""
    results = send_test_alert(
        dry_run=bool(body.dry_run),
        lang=body.lang or "en",
        channels=body.channels,
    )
    status_str = ", ".join([f"{k}: {'OK' if v else 'FAIL'}" for k, v in results.items()])

    client_ip = extract_client_ip(request)
    log_audit_event(
        session=session,
        username=admin_user.username,
        user_role=admin_user.role,
        action="alert_test_sent",
        target_type="alert",
        target_id="test_alert",
        ip_address=client_ip,
        details={"channels": body.channels, "dry_run": body.dry_run, "results": results},
        user_id=admin_user.id,
    )

    return AlertTestResponse(
        results=results,
        message=f"Test alert dispatched: {status_str}",
    )


@router.get("/vapid-public-key")
def get_vapid_public_key_endpoint() -> dict:
    """Returns the base64url-encoded EC P-256 public key required for browser push subscriptions."""
    keys = get_or_generate_vapid_keys()
    public_key = keys.get("public_key")
    if not public_key:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="VAPID keys could not be generated on this server.",
        )
    return {"public_key": public_key}


@router.post("/push/subscribe")
def subscribe_push_endpoint(
    body: WebPushSubscribeRequest,
    request: Request,
    session: Session = Depends(get_session),
    current_user: Optional[User] = Depends(get_current_user_optional),
) -> dict:
    """Registers or re-activates a Web Push subscription endpoint for browser/phone alerts."""
    endpoint_hash = hashlib.sha256(body.endpoint.encode("utf-8")).hexdigest()[:24]
    sub_id = f"sub_{endpoint_hash}"

    p256dh = body.keys.get("p256dh")
    auth_secret = body.keys.get("auth")
    if not p256dh or not auth_secret:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid Web Push subscription keys (missing p256dh or auth).",
        )

    existing = session.exec(
        select(WebPushSubscription).where(WebPushSubscription.endpoint == body.endpoint)
    ).first()

    now = time.time()
    user_id = current_user.id if current_user else None
    user_agent = body.user_agent or request.headers.get("user-agent")

    if existing:
        existing.p256dh = p256dh
        existing.auth = auth_secret
        existing.is_active = True
        existing.user_id = user_id or existing.user_id
        existing.user_agent = user_agent
        session.add(existing)
    else:
        new_sub = WebPushSubscription(
            id=sub_id,
            endpoint=body.endpoint,
            p256dh=p256dh,
            auth=auth_secret,
            user_id=user_id,
            user_agent=user_agent,
            created_at=now,
            is_active=True,
        )
        session.add(new_sub)

    session.commit()
    logger.info("Web push subscription registered/refreshed for user %s (%s)", user_id or "anonymous", sub_id)
    return {"status": "subscribed", "subscription_id": sub_id}


@router.post("/push/unsubscribe")
def unsubscribe_push_endpoint(
    body: WebPushUnsubscribeRequest,
    session: Session = Depends(get_session),
) -> dict:
    """Deactivates a Web Push subscription endpoint."""
    existing = session.exec(
        select(WebPushSubscription).where(WebPushSubscription.endpoint == body.endpoint)
    ).first()
    if existing:
        existing.is_active = False
        session.add(existing)
        session.commit()
        logger.info("Web push subscription deactivated: %s", existing.id)
    return {"status": "unsubscribed"}


@router.get("/ack/{ack_token}")
def acknowledge_alert_by_token(
    ack_token: str,
    request: Request,
    session: Session = Depends(get_session),
) -> Response:
    """One-click acknowledgment endpoint for incident alerts via email, telegram, or phone notification.

    Renders a responsive, high-contrast HTML confirmation page or returns JSON if requested.
    """
    clean_token = str(ack_token).strip()
    inc = session.exec(select(Incident).where(Incident.ack_token == clean_token)).first()
    if not inc:
        # Check if accept header wants JSON
        if "application/json" in request.headers.get("accept", ""):
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Invalid or expired acknowledgment link.",
            )
        html_not_found = """<!DOCTYPE html>
<html>
<head><meta charset="utf-8"><title>ACS Alert - Not Found</title><meta name="viewport" content="width=device-width, initial-scale=1"></head>
<body style="background:#0f172a;color:#f8fafc;font-family:-apple-system,BlinkMacSystemFont,sans-serif;display:flex;align-items:center;justify-content:center;height:100vh;margin:0;">
  <div style="background:#1e293b;border:1px solid #334155;padding:32px;border-radius:12px;max-width:440px;text-align:center;">
    <h2 style="color:#ef4444;margin-top:0;">⚠️ Link Not Found</h2>
    <p style="color:#94a3b8;font-size:14px;">The specified acknowledgment token is invalid or has expired.</p>
  </div>
</body>
</html>"""
        return HTMLResponse(content=html_not_found, status_code=404)

    now_iso = datetime.utcnow().isoformat() + "Z"
    now_epoch = time.time()
    inc.is_acknowledged = True
    inc.acknowledged_at = now_iso
    inc.acknowledged_by = "One-Click Remote (Mobile/Email)"
    session.add(inc)

    # Update associated AlertDelivery records to acknowledged
    deliveries = session.exec(select(AlertDelivery).where(AlertDelivery.incident_id == inc.incident_id)).all()
    for deliv in deliveries:
        deliv.status = "acknowledged"
        deliv.acknowledged_at = now_epoch
        deliv.acknowledged_by = "One-Click Remote"
        session.add(deliv)

    session.commit()

    # Log audit event
    try:
        log_audit_event(
            session=session,
            username="remote_operator",
            user_role="operator",
            action="incident_acknowledged_one_click",
            target_type="incident",
            target_id=inc.incident_id,
            ip_address=extract_client_ip(request),
            details={"ack_token": clean_token, "timestamp": now_iso},
        )
    except Exception:
        pass

    if "application/json" in request.headers.get("accept", ""):
        return JSONResponse(
            content={
                "status": "acknowledged",
                "incident_id": inc.incident_id,
                "is_acknowledged": True,
                "acknowledged_at": now_iso,
                "acknowledged_by": inc.acknowledged_by,
            }
        )

    # Render sleek, dark-mode confirmation page
    html_success = f"""<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <title>ACS Alert Acknowledged - {inc.incident_id}</title>
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <style>
    body {{
      background: #090d16;
      color: #f8fafc;
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
      display: flex;
      align-items: center;
      justify-content: center;
      min-height: 100vh;
      margin: 0;
      padding: 16px;
      box-sizing: border-box;
    }}
    .card {{
      background: #111827;
      border: 1px solid #1f2937;
      border-radius: 14px;
      max-width: 480px;
      width: 100%;
      padding: 28px;
      box-shadow: 0 20px 25px -5px rgba(0,0,0,0.5);
    }}
    .badge {{
      display: inline-flex;
      align-items: center;
      gap: 6px;
      padding: 6px 14px;
      border-radius: 9999px;
      background: #064e3b;
      color: #34d399;
      font-size: 13px;
      font-weight: 600;
      margin-bottom: 16px;
    }}
    .meta-row {{
      display: flex;
      justify-content: space-between;
      border-bottom: 1px solid #1f2937;
      padding: 10px 0;
      font-size: 13px;
    }}
    .meta-row span:first-child {{ color: #94a3b8; }}
    .meta-row span:last-child {{ font-weight: 600; color: #f1f5f9; }}
    .btn {{
      display: block;
      text-align: center;
      padding: 12px;
      background: #2563eb;
      color: #ffffff;
      text-decoration: none;
      border-radius: 8px;
      font-weight: 600;
      font-size: 14px;
      margin-top: 24px;
      transition: background 0.2s;
    }}
    .btn:hover {{ background: #1d4ed8; }}
    .disclaimer {{
      background: #291b00;
      border: 1px solid #78350f;
      padding: 10px 12px;
      border-radius: 8px;
      font-size: 11px;
      color: #fde68a;
      margin-top: 18px;
      line-height: 1.4;
    }}
  </style>
</head>
<body>
  <div class="card">
    <div class="badge">
      <span>✓</span>
      <span>INCIDENT ACKNOWLEDGED</span>
    </div>
    <h2 style="margin: 0 0 6px 0; font-size: 20px;">Incident Alert Confirmed</h2>
    <p style="margin: 0 0 20px 0; color: #94a3b8; font-size: 13px;">
      Automated escalation has been halted. On-call response registered.
    </p>

    <div class="meta-row">
      <span>Incident ID</span>
      <span>{inc.incident_id}</span>
    </div>
    <div class="meta-row">
      <span>Classification</span>
      <span style="text-transform: uppercase;">{inc.class_name}</span>
    </div>
    <div class="meta-row">
      <span>Severity</span>
      <span style="color: #f43f5e; text-transform: uppercase;">{inc.severity} ({inc.severity_score}/100)</span>
    </div>
    <div class="meta-row">
      <span>Acknowledged At</span>
      <span>{now_iso}</span>
    </div>

    <a href="/" class="btn">Open ACS Control Center</a>

    <div class="disclaimer">
      {DISCLAIMER_TEXT}
    </div>
  </div>
</body>
</html>"""
    return HTMLResponse(content=html_success, status_code=200)


@router.post("/escalate-check")
def trigger_escalation_check_endpoint(
    session: Session = Depends(get_session),
    current_user: User = Depends(require_operator),
) -> dict:
    """Manually checks and executes escalation for any unacknowledged incidents (Operator/Admin)."""
    escalated = check_and_escalate_unacknowledged_incidents(session=session)
    return {
        "status": "completed",
        "escalated_count": len(escalated),
        "escalated_incidents": escalated,
    }


@router.get("/settings", response_model=AlertSettingsResponse)
def get_alert_settings_endpoint(
    viewer: User = Depends(require_viewer),
) -> AlertSettingsResponse:
    """Returns active emergency alert settings, delivery channels, VAPID key, and escalation state."""
    channels = list(alert_manager.notifiers.keys())
    channel_status = {
        name: ChannelStatusInfo(
            configured=notifier.is_configured(),
            description="Configured" if notifier.is_configured() else "Not configured",
        )
        for name, notifier in alert_manager.notifiers.items()
    }

    vapid_keys = get_or_generate_vapid_keys()
    pub_key = vapid_keys.get("public_key")

    return AlertSettingsResponse(
        enabled=config.alerts_enabled,
        dry_run=config.alerts_dry_run,
        channels=channels,
        channel_status=channel_status,
        vapid_public_key=pub_key,
        escalation_enabled=getattr(config, "escalation_enabled", True),
        escalation_delay_minutes=getattr(config, "escalation_delay_minutes", 5.0),
    )


@router.put("/settings", response_model=AlertSettingsResponse)
def update_alert_settings_endpoint(
    body: AlertSettingsUpdateRequest,
    request: Request,
    session: Session = Depends(get_session),
    admin_user: User = Depends(require_admin),
) -> AlertSettingsResponse:
    """Updates emergency alert configuration, escalation parameters, and detection thresholds (Admin only)."""
    changes = {}
    if body.enabled is not None:
        config.alerts_enabled = body.enabled
        changes["enabled"] = body.enabled
    if body.dry_run is not None:
        config.alerts_dry_run = body.dry_run
        alert_manager.dry_run = body.dry_run
        changes["dry_run"] = body.dry_run
    if body.model_path is not None:
        config.model_path = body.model_path
        changes["model_path"] = body.model_path
    if body.confidence_threshold is not None:
        config.confidence_threshold = body.confidence_threshold
        changes["confidence_threshold"] = body.confidence_threshold
    if body.cooldown_seconds is not None:
        config.cooldown_seconds = body.cooldown_seconds
        changes["cooldown_seconds"] = body.cooldown_seconds
    if body.escalation_enabled is not None:
        config.escalation_enabled = body.escalation_enabled
        changes["escalation_enabled"] = body.escalation_enabled
    if body.escalation_delay_minutes is not None:
        config.escalation_delay_minutes = body.escalation_delay_minutes
        changes["escalation_delay_minutes"] = body.escalation_delay_minutes

    client_ip = extract_client_ip(request)
    log_audit_event(
        session=session,
        username=admin_user.username,
        user_role=admin_user.role,
        action="alert_settings_updated",
        target_type="settings",
        target_id="alerts",
        ip_address=client_ip,
        details=changes,
        user_id=admin_user.id,
    )

    channels: List[str] = list(alert_manager.notifiers.keys())
    channel_status = {
        name: ChannelStatusInfo(
            configured=notifier.is_configured(),
            description="Configured" if notifier.is_configured() else "Not configured",
        )
        for name, notifier in alert_manager.notifiers.items()
    }

    vapid_keys = get_or_generate_vapid_keys()
    pub_key = vapid_keys.get("public_key")

    return AlertSettingsResponse(
        enabled=config.alerts_enabled,
        dry_run=config.alerts_dry_run,
        channels=channels,
        channel_status=channel_status,
        vapid_public_key=pub_key,
        escalation_enabled=getattr(config, "escalation_enabled", True),
        escalation_delay_minutes=getattr(config, "escalation_delay_minutes", 5.0),
    )
