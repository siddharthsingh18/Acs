"""
backend.app.routers.alerts
Route handlers for emergency alert dispatching, connectivity testing, and settings.
"""
import logging
from typing import List

from fastapi import APIRouter

from acs.alerts import alert_manager, send_test_alert
from acs.config import config
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
def send_test_alert_endpoint(body: AlertTestRequest) -> AlertTestResponse:
    """Dispatches a test emergency alert to verify Telegram and SMTP connectivity."""
    results = send_test_alert(
        dry_run=bool(body.dry_run),
        lang=body.lang or "en",
        channels=body.channels,
    )
    status_str = ", ".join([f"{k}: {'OK' if v else 'FAIL'}" for k, v in results.items()])
    return AlertTestResponse(
        results=results,
        message=f"Test alert dispatched: {status_str}",
    )


@router.get("/settings", response_model=AlertSettingsResponse)
def get_alert_settings_endpoint() -> AlertSettingsResponse:
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
def update_alert_settings_endpoint(body: AlertSettingsUpdateRequest) -> AlertSettingsResponse:
    """Updates emergency alert configuration and detection parameters."""
    if body.enabled is not None:
        config.alerts_enabled = body.enabled
    if body.dry_run is not None:
        config.alerts_dry_run = body.dry_run
        alert_manager.dry_run = body.dry_run
    if body.model_path is not None:
        config.model_path = body.model_path
    if body.confidence_threshold is not None:
        config.confidence_threshold = body.confidence_threshold
    if body.cooldown_seconds is not None:
        config.cooldown_seconds = body.cooldown_seconds

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
