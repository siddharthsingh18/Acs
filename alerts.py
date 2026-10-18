"""
alerts.py (Compatibility wrapper)
Re-exports alert notification classes and dispatch functions from acs.alerts.
"""
from acs.alerts import (
    AlertPayload,
    AlertManager,
    BaseNotifier,
    TelegramNotifier,
    SMTPNotifier,
    ConsoleNotifier,
    WebPushNotifier,
    DISCLAIMER_TEXT,
    is_demo_incident,
    StormRateLimiter,
    storm_rate_limiter,
    get_or_generate_vapid_keys,
    alert_manager,
    dispatch_alert,
    send_incident_alert,
    send_test_alert,
)

__all__ = [
    "AlertPayload",
    "AlertManager",
    "BaseNotifier",
    "TelegramNotifier",
    "SMTPNotifier",
    "ConsoleNotifier",
    "WebPushNotifier",
    "DISCLAIMER_TEXT",
    "is_demo_incident",
    "StormRateLimiter",
    "storm_rate_limiter",
    "get_or_generate_vapid_keys",
    "alert_manager",
    "dispatch_alert",
    "send_incident_alert",
    "send_test_alert",
]
