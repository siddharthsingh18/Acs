"""
Unit tests for pluggable emergency alerts system in acs/alerts.py.
Covers:
- BaseNotifier interface compliance.
- TelegramNotifier (dry run, unconfigured, mocked sendPhoto & sendMessage, error handling).
- SMTPNotifier (dry run, unconfigured, mocked MIME message delivery, error handling).
- ConsoleNotifier output.
- AlertManager multi-channel dispatching.
- Pipeline safety: unhandled backend exceptions never crash the pipeline.
- send_test_alert and send_incident_alert helpers.
"""
import os
import shutil
import tempfile
import unittest
from unittest.mock import MagicMock, patch
import cv2
import numpy as np

from acs.alerts import (
    AlertManager,
    AlertPayload,
    BaseNotifier,
    ConsoleNotifier,
    SMTPNotifier,
    TelegramNotifier,
    send_incident_alert,
    send_test_alert,
)


class TestAlertsSystem(unittest.TestCase):
    """Test suite for the emergency notification system."""

    def setUp(self):
        """Create a temporary directory with a dummy keyframe."""
        self.test_dir = tempfile.mkdtemp()
        self.image_path = os.path.join(self.test_dir, "test_keyframe.jpg")
        cv2.imwrite(self.image_path, np.zeros((100, 100, 3), dtype=np.uint8))

        self.payload = AlertPayload(
            incident_id="INC-TEST-001",
            class_name="car_car_accident",
            timestamp="2026-10-08 12:00:00 UTC",
            confidence=0.95,
            report_text="[TEST DISPATCH REPORT]\nSeverity: HIGH\nVehicles: Car 1, Car 2",
            image_path=self.image_path,
            dry_run=False,
        )

    def tearDown(self):
        """Clean up temporary test files."""
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

    def test_telegram_dry_run(self):
        """Telegram notifier in dry-run mode must succeed without network calls."""
        notifier = TelegramNotifier(bot_token="fake_token", chat_id="12345")
        dry_payload = AlertPayload(
            incident_id="INC-DRY-01",
            class_name="car_car_accident",
            timestamp="2026-10-08 12:00:00 UTC",
            confidence=0.90,
            report_text="Dry run report",
            image_path=self.image_path,
            dry_run=True,
        )
        with patch("requests.post") as mock_post:
            result = notifier.send(dry_payload)
            self.assertTrue(result)
            mock_post.assert_not_called()

    def test_telegram_unconfigured(self):
        """Telegram notifier without credentials must safely return False and not crash."""
        notifier = TelegramNotifier(bot_token="", chat_id="")
        self.assertFalse(notifier.is_configured())
        result = notifier.send(self.payload)
        self.assertFalse(result)

    @patch("requests.post")
    def test_telegram_send_photo_success(self, mock_post):
        """Telegram notifier sends photo with caption when image exists."""
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_post.return_value = mock_resp

        notifier = TelegramNotifier(bot_token="test_token", chat_id="12345")
        result = notifier.send(self.payload)
        self.assertTrue(result)
        self.assertTrue(mock_post.called)
        args, kwargs = mock_post.call_args
        self.assertIn("sendPhoto", args[0])
        self.assertIn("photo", kwargs["files"])

    @patch("requests.post")
    def test_telegram_api_error_does_not_crash(self, mock_post):
        """Telegram network or HTTP exceptions must be caught and return False."""
        mock_post.side_effect = RuntimeError("Telegram connection timeout")
        notifier = TelegramNotifier(bot_token="test_token", chat_id="12345")
        result = notifier.send(self.payload)
        self.assertFalse(result)

    def test_smtp_dry_run(self):
        """SMTP notifier in dry-run mode must succeed without network calls."""
        notifier = SMTPNotifier(host="smtp.example.com", recipients=["dispatch@example.com"])
        dry_payload = AlertPayload(
            incident_id="INC-DRY-02",
            class_name="bike_person_accident",
            timestamp="2026-10-08 12:00:00 UTC",
            confidence=0.88,
            report_text="Dry run email report",
            image_path=self.image_path,
            dry_run=True,
        )
        with patch("smtplib.SMTP") as mock_smtp:
            result = notifier.send(dry_payload)
            self.assertTrue(result)
            mock_smtp.assert_not_called()

    def test_smtp_unconfigured(self):
        """SMTP notifier without host or recipients must safely return False."""
        notifier = SMTPNotifier(host="", recipients=[])
        self.assertFalse(notifier.is_configured())
        result = notifier.send(self.payload)
        self.assertFalse(result)

    @patch("smtplib.SMTP")
    def test_smtp_send_email_success(self, mock_smtp):
        """SMTP notifier connects, creates MIME message with attachment, and sends."""
        mock_server = MagicMock()
        mock_smtp.return_value.__enter__.return_value = mock_server

        notifier = SMTPNotifier(
            host="smtp.test.com",
            port=587,
            user="user@test.com",
            password="pwd",
            recipients=["dispatch@test.com"],
            use_tls=True,
        )
        result = notifier.send(self.payload)
        self.assertTrue(result)
        mock_server.starttls.assert_called_once()
        mock_server.login.assert_called_once_with("user@test.com", "pwd")
        mock_server.send_message.assert_called_once()

    @patch("smtplib.SMTP")
    def test_smtp_error_does_not_crash(self, mock_smtp):
        """SMTP delivery exceptions must be handled gracefully without crashing."""
        mock_smtp.side_effect = ConnectionRefusedError("SMTP server unreachable")
        notifier = SMTPNotifier(host="smtp.test.com", recipients=["dispatch@test.com"])
        result = notifier.send(self.payload)
        self.assertFalse(result)

    def test_console_notifier(self):
        """Console notifier always logs and succeeds."""
        notifier = ConsoleNotifier()
        self.assertTrue(notifier.is_configured())
        self.assertTrue(notifier.send(self.payload))

    def test_alert_manager_pipeline_safety(self):
        """An unhandled exception in a custom notifier backend must never crash AlertManager."""
        class FaultyNotifier(BaseNotifier):
            @property
            def name(self) -> str:
                return "faulty"
            def is_configured(self) -> bool:
                return True
            def send(self, payload: AlertPayload) -> bool:
                raise ValueError("Fatal crash inside third-party backend")

        manager = AlertManager(dry_run=True)
        manager.register_notifier(FaultyNotifier())

        results = manager.send(self.payload, channels=["console", "faulty"])
        self.assertIn("console", results)
        self.assertIn("faulty", results)
        self.assertTrue(results["console"])
        self.assertFalse(results["faulty"])

    def test_send_test_alert(self):
        """send_test_alert runs cleanly in dry-run mode."""
        results = send_test_alert(dry_run=True, lang="en")
        self.assertIsInstance(results, dict)
        self.assertTrue(results.get("console"))

    def test_send_incident_alert(self):
        """send_incident_alert helper dispatches incident metadata."""
        class DummyIncident:
            incident_id = "INC-MOCK-999"
            class_name = "car_car_accident"
            start_timestamp = "2026-10-08 15:00:00 UTC"
            max_confidence = 0.94
            image_path = self.image_path

        results = send_incident_alert(
            incident=DummyIncident(),
            report_text="Dispatch assessment test text",
            dry_run=True,
            channels=["console"],
        )
        self.assertTrue(results.get("console"))


if __name__ == "__main__":
    unittest.main()
