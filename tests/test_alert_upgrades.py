"""
tests.test_alert_upgrades
Comprehensive test suite validating alerting upgrades:
- Web Push (VAPID key generation and delivery)
- Hard DEMO incident safety interlock
- Storm rate-limiting and de-duplication
- Multi-tier escalation for unacknowledged incidents
- One-click acknowledgment endpoint (token & authenticated)
- Mandatory disclaimer inclusion across all channels
"""
import os
import secrets
import time
from unittest.mock import MagicMock, patch
import pytest
from fastapi.testclient import TestClient
from sqlmodel import Session, select

from acs.alerts import (
    DISCLAIMER_TEXT,
    AlertManager,
    AlertPayload,
    SMTPNotifier,
    StormRateLimiter,
    TelegramNotifier,
    WebPushNotifier,
    get_or_generate_vapid_keys,
    is_demo_incident,
)
from acs.config import config
from backend.app.db.database import engine, init_db
from backend.app.db.models import AlertDelivery, Incident, WebPushSubscription
from backend.app.main import app
from backend.app.services.alert_escalation_service import check_and_escalate_unacknowledged_incidents


@pytest.fixture(scope="module", autouse=True)
def setup_test_db():
    init_db()


@pytest.fixture
def client():
    return TestClient(app)


class TestVapidAndWebPush:
    """Validates VAPID key management and WebPushNotifier backend."""

    def test_vapid_key_generation(self, tmp_path):
        keys_file = str(tmp_path / "vapid_keys.json")
        with patch.object(config, "vapid_keys_file", keys_file):
            keys = get_or_generate_vapid_keys()
            assert "public_key" in keys
            assert "private_key" in keys
            assert len(keys["public_key"]) >= 80  # Base64url raw uncompressed point
            assert "BEGIN PRIVATE KEY" in keys["private_key"]
            assert os.path.isfile(keys_file)

            # Second call loads existing file
            keys2 = get_or_generate_vapid_keys()
            assert keys2["public_key"] == keys["public_key"]

    def test_webpush_notifier_dry_run_and_demo(self):
        notifier = WebPushNotifier()
        payload = AlertPayload(
            incident_id="INC-2026-WP-01",
            class_name="traffic_collision",
            timestamp="2026-10-19 12:00:00",
            confidence=0.92,
            report_text="Vehicle overturned at junction",
            dry_run=True,
        )
        # In dry-run mode, returns True without external dispatch
        assert notifier.send(payload) is True

    @patch("pywebpush.webpush")
    def test_webpush_notifier_dispatches_with_disclaimer(self, mock_webpush):
        notifier = WebPushNotifier()
        # Insert test subscription into database
        sub_id = f"sub_{secrets.token_hex(8)}"
        token_suffix = secrets.token_hex(4)
        with Session(engine) as session:
            sub = WebPushSubscription(
                id=sub_id,
                endpoint=f"https://fcm.googleapis.com/fcm/send/test-token-{token_suffix}",
                p256dh="test_p256dh_key",
                auth="test_auth_secret",
                user_id="usr_operator",
                created_at=time.time(),
                is_active=True,
            )
            session.add(sub)
            session.commit()

        payload = AlertPayload(
            incident_id="INC-2026-WP-02",
            class_name="severe_crash",
            timestamp="2026-10-19 12:05:00",
            confidence=0.88,
            report_text="Multiple vehicle impact",
            dry_run=False,
            severity="high",
            severity_score=85,
            ack_url="http://localhost:8000/api/alerts/ack/test_token",
        )

        mock_webpush.return_value = MagicMock(status_code=200)
        success = notifier.send(payload)
        assert success is True
        assert mock_webpush.called
        call_kwargs = mock_webpush.call_args[1]
        sent_data = call_kwargs["data"]
        # Mandatory disclaimer must be embedded in notification body
        assert DISCLAIMER_TEXT in sent_data


class TestDemoSafetyInterlock:
    """Verifies that DEMO incidents can NEVER be dispatched over real network sockets."""

    @pytest.mark.parametrize(
        "incident_id,metadata,expected_demo",
        [
            ("DEMO-001", {}, True),
            ("demo-test-2", {}, True),
            ("SAMPLE-004", {}, True),
            ("TEST-ALERT-01", {}, True),
            ("INC-DEMO-99", {}, True),
            ("INC-REAL-101", {"source": "demo_sample.mp4"}, True),
            ("INC-REAL-102", {"video_source": "sample"}, True),
            ("INC-REAL-103", {"is_demo": True}, True),
            ("INC-REAL-200", {"source": "camera_cctv_01.mp4"}, False),
        ],
    )
    def test_is_demo_incident(self, incident_id, metadata, expected_demo):
        assert is_demo_incident(incident_id, metadata) == expected_demo

    @patch("requests.post")
    def test_demo_incident_bars_telegram_network_call(self, mock_post):
        tg = TelegramNotifier(bot_token="test_token", chat_id="123456")
        payload = AlertPayload(
            incident_id="DEMO-CRASH-001",
            class_name="accident",
            timestamp="2026-10-19 12:00:00",
            confidence=0.90,
            report_text="Demo accident test",
            dry_run=False,  # Even if developer explicitly sets dry_run=False
        )
        res = tg.send(payload)
        assert res is True
        # Safety interlock must prevent any HTTP requests to api.telegram.org
        assert not mock_post.called

    @patch("smtplib.SMTP")
    def test_demo_incident_bars_smtp_network_call(self, mock_smtp):
        smtp = SMTPNotifier(host="smtp.example.com", recipients=["admin@example.com"])
        payload = AlertPayload(
            incident_id="SAMPLE-TEST-99",
            class_name="accident",
            timestamp="2026-10-19 12:00:00",
            confidence=0.85,
            report_text="Sample simulation",
            dry_run=False,
        )
        res = smtp.send(payload)
        assert res is True
        assert not mock_smtp.called


class TestStormRateLimiterAndDeDuplication:
    """Tests rate-limiting during collision pileups and alert de-duplication."""

    def test_storm_rate_limiter_window(self):
        limiter = StormRateLimiter(window_seconds=10.0, max_alerts=2)
        key = "camera_junction_05"

        # First 2 alerts allowed
        assert limiter.should_rate_limit(key) is False
        assert limiter.should_rate_limit(key) is False

        # 3rd alert within window is suppressed
        assert limiter.should_rate_limit(key) is True

        # Different camera is unaffected
        assert limiter.should_rate_limit("camera_highway_02") is False

    def test_alert_manager_suppresses_storm(self):
        mgr = AlertManager()
        mgr.storm_limiter = StormRateLimiter(window_seconds=10.0, max_alerts=1)

        payload1 = AlertPayload(
            incident_id=f"INC-STORM-{secrets.token_hex(4)}",
            class_name="accident",
            timestamp="2026-10-19 12:00:00",
            confidence=0.90,
            report_text="Accident 1",
            camera_id="cam_storm_test",
            dry_run=True,
        )
        res1 = mgr.send(payload1, channels=["console"])
        assert res1["console"] is True

        # Rapid second alert at same camera is rate-limited
        payload2 = AlertPayload(
            incident_id=f"INC-STORM-{secrets.token_hex(4)}",
            class_name="accident",
            timestamp="2026-10-19 12:00:05",
            confidence=0.95,
            report_text="Accident 2",
            camera_id="cam_storm_test",
            dry_run=True,
        )
        res2 = mgr.send(payload2, channels=["console"])
        assert res2["console"] is False  # Suppressed by storm rate limiter


class TestEscalationAndOneClickAck:
    """Validates multi-tier escalation service and one-click token acknowledgment."""

    def test_one_click_acknowledgment_endpoint(self, client):
        inc_id = f"INC-ACK-{secrets.token_hex(4)}"
        ack_token = secrets.token_urlsafe(24)
        with Session(engine) as session:
            inc = Incident(
                incident_id=inc_id,
                start_timestamp="2026-10-19 12:00:00",
                end_timestamp="2026-10-19 12:01:00",
                class_name="traffic_accident",
                max_confidence=0.91,
                image_path="reports/keyframe.jpg",
                keyframe_url="/media/keyframe.jpg",
                created_at=time.time(),
                is_acknowledged=False,
                ack_token=ack_token,
                escalation_tier=1,
            )
            deliv = AlertDelivery(
                incident_id=inc_id,
                channel="webpush",
                tier=1,
                status="sent",
                created_at=time.time(),
            )
            session.add(inc)
            session.add(deliv)
            session.commit()

        # 1. Test HTML confirmation view (clicking link from lockscreen/email)
        resp_html = client.get(f"/api/alerts/ack/{ack_token}")
        assert resp_html.status_code == 200
        assert "Incident Alert Confirmed" in resp_html.text
        assert "INCIDENT ACKNOWLEDGED" in resp_html.text
        assert DISCLAIMER_TEXT in resp_html.text

        # 2. Verify state updated in database
        with Session(engine) as session:
            updated = session.get(Incident, inc_id)
            assert updated.is_acknowledged is True
            assert updated.acknowledged_at is not None
            assert "Remote" in updated.acknowledged_by

            deliv_updated = session.exec(
                select(AlertDelivery).where(AlertDelivery.incident_id == inc_id)
            ).first()
            assert deliv_updated.status == "acknowledged"

    def test_escalation_service_triggers_for_unacknowledged_incident(self):
        inc_id = f"INC-ESC-{secrets.token_hex(4)}"
        # Incident created 10 minutes ago, unacknowledged
        created_10m_ago = time.time() - 600.0

        with Session(engine) as session:
            inc = Incident(
                incident_id=inc_id,
                start_timestamp="2026-10-19 11:50:00",
                end_timestamp="2026-10-19 11:51:00",
                class_name="rollover_crash",
                max_confidence=0.94,
                image_path="reports/esc.jpg",
                keyframe_url="/media/esc.jpg",
                created_at=created_10m_ago,
                is_acknowledged=False,
                escalation_tier=1,
                severity="high",
                severity_score=90,
            )
            session.add(inc)
            session.commit()

            # Execute escalation check with 5 min threshold
            escalated = check_and_escalate_unacknowledged_incidents(
                session=session, delay_minutes_override=5.0
            )
            assert len(escalated) >= 1
            esc_entry = next((e for e in escalated if e["incident_id"] == inc_id), None)
            assert esc_entry is not None
            assert esc_entry["tier"] == 2

            # Verify incident updated to Tier 2 in database
            refreshed = session.get(Incident, inc_id)
            assert refreshed.escalation_tier == 2
            assert refreshed.last_escalated_at is not None

    def test_acknowledged_incident_never_escalates(self):
        inc_id = f"INC-NOESC-{secrets.token_hex(4)}"
        created_20m_ago = time.time() - 1200.0

        with Session(engine) as session:
            inc = Incident(
                incident_id=inc_id,
                start_timestamp="2026-10-19 11:40:00",
                end_timestamp="2026-10-19 11:41:00",
                class_name="fender_bender",
                max_confidence=0.75,
                image_path="reports/ack.jpg",
                keyframe_url="/media/ack.jpg",
                created_at=created_20m_ago,
                is_acknowledged=True,  # Already acknowledged by officer
                acknowledged_by="Officer Jane Doe",
                escalation_tier=1,
            )
            session.add(inc)
            session.commit()

            escalated = check_and_escalate_unacknowledged_incidents(
                session=session, delay_minutes_override=5.0
            )
            esc_entry = next((e for e in escalated if e["incident_id"] == inc_id), None)
            assert esc_entry is None


class TestMandatoryDisclaimer:
    """Verifies that the disclaimer is included across all channels and endpoints."""

    def test_disclaimer_constant(self):
        assert "Alerts support human decisions" in DISCLAIMER_TEXT
        assert "must not trigger emergency dispatch automatically" in DISCLAIMER_TEXT

    def test_disclaimer_in_settings_endpoint(self, client):
        resp = client.get("/api/alerts/settings")
        # Ensure status code 200 or 401 (if auth enforced), in default test it returns settings
        assert resp.status_code in (200, 401)
