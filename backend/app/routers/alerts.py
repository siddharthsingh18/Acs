"""
backend.app.routers.alerts
Route handlers for emergency alert dispatching, connectivity testing, and settings.
"""
import logging
from typing import List

from fastapi import APIRouter, Depends, Request
from sqlmodel import Session

from acs.alerts import alert_manager, send_test_alert
from acs.config import config
from backend.app.db.database import get_session
from backend.app.db.models import User
from backend.app.deps import extract_client_ip, require_admin, require_viewer
from backend.app.services.audit_service import log_audit_event
from backend.app.schemas import (
    AlertSettingsResponse,
    AlertSettingsUpdateRequest,
    AlertTestRequest,
    AlertTestResponse,
    ChannelStatusInfo,
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
    """Dispatches a test emergency alert to verify Telegram and SMTP connectivity (Admin only)."""
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


@router.get("/settings", response_model=AlertSettingsResponse)
def get_alert_settings_endpoint(
    viewer: User = Depends(require_viewer),
) -> AlertSettingsResponse:
    """Returns active emergency alert settings, delivery channels, and credential readiness."""
    channels = list(alert_manager.notifiers.keys())
    channel_status = {
        name: ChannelStatusInfo(
            configured=notifier.is_configured(),
            description="Configured" if notifier.is_configured() else "Not configured",
        )
        for name, notifier in alert_manager.notifiers.items()
    }
    return AlertSettingsResponse(
        enabled=config.alerts_enabled,
        dry_run=config.alerts_dry_run,
        channels=channels,
        channel_status=channel_status,
    )


@router.put("/settings", response_model=AlertSettingsResponse)
def update_alert_settings_endpoint(
    body: AlertSettingsUpdateRequest,
    request: Request,
    session: Session = Depends(get_session),
    admin_user: User = Depends(require_admin),
) -> AlertSettingsResponse:
    """Updates emergency alert configuration and detection parameters (Admin only)."""
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
    return AlertSettingsResponse(
        enabled=config.alerts_enabled,
        dry_run=config.alerts_dry_run,
        channels=channels,
        channel_status=channel_status,
    )

