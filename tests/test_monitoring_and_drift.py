"""
tests.test_monitoring_and_drift
Comprehensive test suite validating:
1. Prometheus metrics registry & /metrics endpoint.
2. Structured logging & correlation Request IDs.
3. Sentry error handling & data scrubbing.
4. Model drift distribution, PSI calculation, and anomaly alerting.
5. MLflow experiment tracking integration.
"""
import json
import logging
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from acs.tracking import ExperimentTracker, compute_file_hash
from backend.app.core.logging_config import (
    JSONStructuredFormatter,
    RequestIDFilter,
    set_request_id,
)
from backend.app.core.metrics import (
    record_alert_delivery,
    record_camera_fps,
    record_incident,
    record_inference_latency,
    record_model_drift,
    record_queue_depth,
    record_vlm_call,
)
from backend.app.core.sentry import _scrub_sensitive_data, init_sentry
from backend.app.main import app
from backend.app.services.drift_service import (
    compute_distribution,
    compute_psi,
    drift_service,
)


@pytest.fixture
def client():
    return TestClient(app)


# -----------------------------------------------------------------------------
# 1. Prometheus Metrics Tests
# -----------------------------------------------------------------------------
def test_prometheus_metrics_recordings_and_endpoint(client):
    """Verifies that all required Prometheus metrics are recorded and scraped via /metrics."""
    # Emit sample recordings
    record_camera_fps("cam-test-1", "Northbound Cam", 24.5)
    record_inference_latency("cam-test-1", 0.018, model="best.pt")
    record_queue_depth("review_queue", 7)
    record_incident("cam-test-1", "high", "auto_confirmed")
    record_vlm_call("success", duration_seconds=1.2, model="gemini-2.5")
    record_vlm_call("timeout", duration_seconds=5.0, model="gemini-2.5")
    record_alert_delivery("telegram", "sent", duration_seconds=0.35)
    record_model_drift("cam-test-1", "confidence", False)

    # Scrape endpoint
    response = client.get("/metrics")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/plain")
    body = response.text

    # Verify all required metrics exist in output
    assert "acs_camera_fps" in body
    assert 'camera_id="cam-test-1"' in body
    assert "acs_inference_duration_seconds_bucket" in body
    assert "acs_queue_depth" in body
    assert 'queue_name="review_queue"' in body
    assert "acs_incidents_total" in body
    assert 'severity="high"' in body
    assert "acs_vlm_calls_total" in body
    assert 'status="timeout"' in body
    assert "acs_alert_delivery_total" in body
    assert 'channel="telegram"' in body
    assert "acs_model_drift_detected" in body


# -----------------------------------------------------------------------------
# 2. Structured Logging & Request IDs Tests
# -----------------------------------------------------------------------------
def test_request_id_generation_and_propagation(client):
    """Verifies that incoming requests receive or generate an X-Request-ID header."""
    # 1. Custom Request ID preserved
    custom_id = "req_custom_trace_98765"
    res1 = client.get("/api/health", headers={"X-Request-ID": custom_id})
    assert res1.status_code == 200
    assert res1.headers.get("X-Request-ID") == custom_id

    # 2. Auto-generated Request ID when omitted
    res2 = client.get("/api/health")
    assert res2.status_code == 200
    gen_id = res2.headers.get("X-Request-ID")
    assert gen_id is not None
    assert gen_id.startswith("req_")


def test_structured_log_formatter():
    """Validates that JSONStructuredFormatter outputs JSON with request_id and metadata."""
    set_request_id("req_test_abc123")
    formatter = JSONStructuredFormatter()
    record = logging.LogRecord(
        name="acs.test",
        level=logging.INFO,
        pathname="test.py",
        lineno=42,
        msg="Test accident event",
        args=(),
        exc_info=None,
    )
    RequestIDFilter().filter(record)
    output = formatter.format(record)

    parsed = json.loads(output)
    assert parsed["level"] == "INFO"
    assert parsed["message"] == "Test accident event"
    assert parsed["request_id"] == "req_test_abc123"
    assert "timestamp" in parsed


# -----------------------------------------------------------------------------
# 3. Sentry Error Monitoring & Scrubbing Tests
# -----------------------------------------------------------------------------
def test_sentry_data_scrubbing():
    """Verifies that credentials and tokens are scrubbed from Sentry events."""
    event = {
        "request": {
            "headers": {
                "authorization": "Bearer secret_jwt_token_12345",
                "cookie": "session=xyz987",
                "content-type": "application/json",
            },
            "data": {
                "password": "super_secret_admin_pw",
                "username": "dispatcher_1",
            },
        }
    }
    scrubbed = _scrub_sensitive_data(event, {})
    assert scrubbed["request"]["headers"]["authorization"] == "[FILTERED]"
    assert scrubbed["request"]["headers"]["cookie"] == "[FILTERED]"
    assert scrubbed["request"]["headers"]["content-type"] == "application/json"
    assert scrubbed["request"]["data"]["password"] == "[FILTERED]"
    assert scrubbed["request"]["data"]["username"] == "dispatcher_1"


def test_sentry_init_fallback():
    """Verifies that init_sentry handles missing DSN cleanly without crashing."""
    with patch.dict("os.environ", {"SENTRY_DSN": ""}, clear=False):
        assert init_sentry() is False


# -----------------------------------------------------------------------------
# 4. Model Drift & PSI Tests
# -----------------------------------------------------------------------------
def test_compute_distribution_and_psi():
    """Verifies 10-decile histogram calculation and Population Stability Index logic."""
    # 1. Identical distributions should have near-zero PSI
    baseline_confs = [0.45, 0.55, 0.65, 0.75, 0.85, 0.95] * 20
    current_confs = [0.46, 0.54, 0.66, 0.74, 0.86, 0.94] * 20

    base_dist = compute_distribution(baseline_confs, total_detections=120)
    curr_dist = compute_distribution(current_confs, total_detections=120)

    psi_stable = compute_psi(base_dist.confidence_bins, curr_dist.confidence_bins)
    assert psi_stable < 0.10, f"Expected stable PSI < 0.10, got {psi_stable}"

    # 2. Sharply shifted distribution (e.g. lens dirty causing confidence drop to low bins)
    shifted_confs = [0.05, 0.12, 0.18, 0.22, 0.25] * 20
    dirty_dist = compute_distribution(shifted_confs, total_detections=100)

    psi_drifted = compute_psi(base_dist.confidence_bins, dirty_dist.confidence_bins)
    assert psi_drifted >= 0.25, f"Expected significant drift PSI >= 0.25, got {psi_drifted}"


def test_camera_drift_detection_and_alert():
    """Verifies that ModelDriftService diagnoses dirty lenses and dispatches alert."""
    test_cam_id = "cam-test-drift-1"

    # Establish baseline with high confidence
    base_confs = [0.80, 0.85, 0.90, 0.92, 0.95] * 10
    base_metrics = compute_distribution(base_confs, total_detections=50, window_name="baseline")
    drift_service.set_camera_baseline(test_cam_id, custom_metrics=base_metrics)

    # Mock extract_camera_observations to return a degraded dirty-lens dataset
    with patch.object(
        drift_service,
        "extract_camera_observations",
        return_value=([0.15, 0.22, 0.28, 0.30] * 5, 5),
    ):
        with patch.object(drift_service, "_dispatch_drift_alert") as mock_dispatch:
            report = drift_service.check_camera_drift(test_cam_id, camera_name="Junction 5 East")

            assert report.drift_detected is True
            assert report.drift_severity == "critical"
            assert report.psi_score >= 0.25
            assert any("dirty" in r.lower() or "shift" in r.lower() for r in report.reasons)
            assert mock_dispatch.called


# -----------------------------------------------------------------------------
# 5. MLflow Experiment Tracking Tests
# -----------------------------------------------------------------------------
def test_experiment_tracker_lifecycle(tmp_path):
    """Verifies that ExperimentTracker logs parameters, metrics, dataset links, and artifacts."""
    test_db = tmp_path / "test_mlruns.db"
    tracker = ExperimentTracker(
        experiment_name="acs-test-experiment",
        tracking_uri=f"sqlite:///{test_db}",
        enabled=True,
    )

    with tracker.start_run(run_name="unit-test-run") as run:
        assert run is not None
        tracker.log_params({"model_arch": "yolo11n", "imgsz": 640})
        tracker.log_metrics({"mAP50": 0.577, "cpu_fps": 60.5})

        # Test dummy artifact
        artifact_file = tmp_path / "dummy_report.md"
        artifact_file.write_text("# Test Report")
        tracker.log_artifact(str(artifact_file))

        # Test dataset metadata
        data_yaml = tmp_path / "data.yaml"
        data_yaml.write_text("names: ['accident', 'car']\nnc: 2\n")
        tracker.log_dataset_metadata("test_dataset", data_yaml_path=str(data_yaml))


def test_compute_file_hash(tmp_path):
    """Verifies SHA256 file fingerprinting for dataset configs."""
    f = tmp_path / "test_file.txt"
    f.write_text("automated cctv surveillance")
    h = compute_file_hash(f)
    assert len(h) == 16
    assert compute_file_hash(tmp_path / "non_existent.txt") == "missing"
