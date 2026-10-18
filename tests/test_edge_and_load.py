"""
tests.test_edge_and_load
Unit and integration tests for:
1. Batched GPU/Hardware inference engine (acs.batch_inference)
2. Edge detection agent & persistent offline queue (edge.edge_client)
3. Edge API router endpoints (/api/edge/incidents, /api/edge/heartbeat, /api/edge/nodes)
4. High-concurrency load verification for REST API and WebSocket fan-out
"""
import concurrent.futures
import json
import os
import shutil
import tempfile
import time
import numpy as np
import pytest
from fastapi.testclient import TestClient
from sqlmodel import Session, select

from acs.batch_inference import BatchedInferenceEngine
from backend.app.core.security import create_access_token
from backend.app.db.database import get_session
from backend.app.db.models import AuditLog, Incident, User
from backend.app.main import app
from edge.edge_client import EdgeQueueManager


@pytest.fixture
def temp_edge_dir():
    """Sets up a temporary working directory for edge queue testing."""
    td = tempfile.mkdtemp()
    yield td
    shutil.rmtree(td, ignore_errors=True)


def test_batched_inference_engine_sync():
    """Verifies that BatchedInferenceEngine processes batches of frames correctly."""
    engine = BatchedInferenceEngine(
        model_path="best.pt",
        max_batch_size=4,
    )
    # Generate 3 dummy test frames
    frames = [np.zeros((640, 640, 3), dtype=np.uint8) for _ in range(3)]
    results = engine.predict_batch_sync(frames)
    assert len(results) == 3
    for r in results:
        assert hasattr(r, "boxes")


def test_edge_offline_queue(temp_edge_dir):
    """Verifies persistent offline queueing, retrieval, retry tracking, and deletion."""
    db_path = os.path.join(temp_edge_dir, "test_queue.db")
    queue_mgr = EdgeQueueManager(db_path=db_path)

    incident_id = f"INC-EDGE-TEST-{time.time_ns()}"
    payload = {"incident_id": incident_id, "class_name": "car_car_accident", "severity": "high"}

    # 1. Enqueue
    queue_mgr.enqueue(
        incident_id=incident_id,
        camera_id="cam-junction-1",
        payload=payload,
        keyframe_path="/tmp/fake_kf.jpg",
        clip_path="/tmp/fake_clip.mp4",
    )

    # 2. Retrieve pending
    pending = queue_mgr.get_pending(limit=5)
    assert len(pending) == 1
    assert pending[0]["incident_id"] == incident_id
    assert json.loads(pending[0]["payload_json"])["class_name"] == "car_car_accident"

    # 3. Increment retry
    queue_mgr.increment_retry(incident_id)
    pending_updated = queue_mgr.get_pending(limit=5)
    assert pending_updated[0]["retry_count"] == 1

    # 4. Remove
    queue_mgr.remove(incident_id)
    assert len(queue_mgr.get_pending(limit=5)) == 0


def test_edge_incidents_api_endpoint(monkeypatch, temp_edge_dir):
    """Verifies that central /api/edge/incidents receives edge events and stores media."""
    client = TestClient(app)
    session = next(get_session())

    incident_id = f"INC-EDGE-UPLINK-{time.time_ns()}"
    meta = {
        "incident_id": incident_id,
        "camera_id": "edge-cam-01",
        "edge_node_id": "orin-node-alpha",
        "class_name": "car_bike_accident",
        "max_confidence": 0.88,
        "severity": "high",
        "severity_score": 85,
        "start_timestamp": "2026-10-19 04:00:00 UTC",
        "end_timestamp": "2026-10-19 04:00:05 UTC",
    }

    dummy_kf_bytes = b"\xFF\xD8\xFF\xE0\x00\x10JFIF" + b"\x00" * 100
    dummy_clip_bytes = b"\x00\x00\x00 ftypisom" + b"\x00" * 100

    resp = client.post(
        "/api/edge/incidents",
        data={"metadata": json.dumps(meta)},
        files={
            "keyframe": ("keyframe.jpg", dummy_kf_bytes, "image/jpeg"),
            "clip": ("clip.mp4", dummy_clip_bytes, "video/mp4"),
        },
        headers={"X-Edge-API-Key": "dev_edge_secret_key"},
    )
    assert resp.status_code == 201
    assert resp.json()["success"] is True

    # Check database persistence
    inc_row = session.exec(select(Incident).where(Incident.incident_id == incident_id)).first()
    assert inc_row is not None
    assert inc_row.class_name == "car_bike_accident"
    assert inc_row.severity == "high"

    # Check audit log
    audit_row = session.exec(
        select(AuditLog).where(
            AuditLog.target_id == incident_id,
            AuditLog.action == "edge.incident.ingested",
        )
    ).first()
    assert audit_row is not None
    assert "orin-node-alpha" in audit_row.username


def test_edge_heartbeat_and_nodes_api():
    """Verifies edge node telemetry ingestion and status listing."""
    client = TestClient(app)

    # 1. Post Heartbeat
    node_id = f"jetson-node-{time.time_ns()}"
    heartbeat_data = {
        "node_id": node_id,
        "cpu_percent": 14.5,
        "ram_percent": 32.0,
        "active_cameras": 4,
        "pending_queue_count": 0,
    }
    hb_resp = client.post(
        "/api/edge/heartbeat",
        json=heartbeat_data,
        headers={"X-Edge-API-Key": "dev_edge_secret_key"},
    )
    assert hb_resp.status_code == 200
    assert hb_resp.json()["status"] == "ok"

    # 2. Get Nodes
    nodes_resp = client.get(
        "/api/edge/nodes",
        headers={"X-Edge-API-Key": "dev_edge_secret_key"},
    )
    assert nodes_resp.status_code == 200
    nodes = nodes_resp.json()
    assert any(n.get("node_id") == node_id for n in nodes)


def test_high_concurrency_api_load():
    """Simulates multi-threaded burst load on the central REST API."""
    def worker_req(req_idx: int) -> int:
        client = TestClient(app)
        resp = client.get("/api/stats")
        return resp.status_code

    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        futures = [executor.submit(worker_req, i) for i in range(25)]
        results = [f.result() for f in concurrent.futures.as_completed(futures)]

    # All concurrent requests must return 200 OK
    assert len(results) == 25
    assert all(code == 200 for code in results)
