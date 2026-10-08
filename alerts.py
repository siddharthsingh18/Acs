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
    "alert_manager",
    "dispatch_alert",
    "send_incident_alert",
    "send_test_alert",
]
