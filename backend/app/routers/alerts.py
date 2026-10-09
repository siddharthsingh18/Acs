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
    """Returns active emergency alert settings and available delivery channels."""
    channels = list(alert_manager.notifiers.keys())
    return AlertSettingsResponse(
        enabled=config.alerts_enabled,
        dry_run=config.alerts_dry_run,
        channels=channels,
    )


@router.put("/settings", response_model=AlertSettingsResponse)
def update_alert_settings_endpoint(body: AlertSettingsUpdateRequest) -> AlertSettingsResponse:
    """Updates emergency alert configuration (enabled flag and dry_run mode)."""
    if body.enabled is not None:
        config.alerts_enabled = body.enabled
    if body.dry_run is not None:
        config.alerts_dry_run = body.dry_run
        alert_manager.dry_run = body.dry_run

    channels: List[str] = list(alert_manager.notifiers.keys())
    return AlertSettingsResponse(
        enabled=config.alerts_enabled,
        dry_run=config.alerts_dry_run,
        channels=channels,
    )
