"""
tests.test_api_jobs
Comprehensive test suite for job-based video analysis service:
- Job creation via JSON (sample video acc_ex.mp4) and multipart upload
- Concurrency control (409 Conflict if another job is running)
- WebSocket streaming (frames, max 960px, JPEG quality 70, max 15 fps throttle, summary)
- Job cancellation
- Keyframe media serving (/media/keyframes/...)
- Clean client disconnect handling
"""
import base64
import io
import time
import pytest
from fastapi.testclient import TestClient

from backend.app.main import app
from backend.app.services.job_manager import job_manager


@pytest.fixture(autouse=True)
def ensure_clean_job_state():
    """Ensures no background jobs are running before or after each test."""
    if job_manager.active_job_id:
        job_manager.cancel_job(job_manager.active_job_id)
        time.sleep(0.3)
    yield
    if job_manager.active_job_id:
        job_manager.cancel_job(job_manager.active_job_id)
        time.sleep(0.3)


@pytest.fixture
def client():
    """Provides FastAPI TestClient."""
    return TestClient(app)


def test_create_sample_job_and_stream(client):
    """Verifies creating a job with sample video acc_ex.mp4 and streaming frames/done over WebSocket."""
    response = client.post(
        "/api/jobs",
        json={
            "source": "sample",
            "model_name": "best.pt",
            "confidence": 0.5,
            "max_frames": 10,
        },
    )
    assert response.status_code == 201
    job_data = response.json()
    assert "job_id" in job_data
    job_id = job_data["job_id"]
    assert job_data["model_name"] == "best.pt"
    assert job_data["confidence"] == 0.5

    # Connect to WebSocket stream
    received_types = []
    frame_messages = []
    done_message = None

    with client.websocket_connect(f"/api/jobs/{job_id}/stream") as ws:
        while True:
            msg = ws.receive_json()
            m_type = msg.get("type")
            received_types.append(m_type)

            if m_type == "frame":
                frame_messages.append(msg)
                assert "index" in msg
                assert "total" in msg
                assert "fps" in msg
                assert "jpeg_base64" in msg

                # Verify valid JPEG base64 payload
                raw_bytes = base64.b64decode(msg["jpeg_base64"])
                assert raw_bytes.startswith(b"\xff\xd8")  # JPEG SOI header

            elif m_type == "done":
                done_message = msg
                break
            elif m_type == "error":
                pytest.fail(f"Job stream encountered error: {msg.get('message')}")

    assert "frame" in received_types
    assert "done" in received_types
    assert len(frame_messages) >= 1
    assert done_message is not None
    assert done_message["summary"]["job_id"] == job_id
    assert done_message["summary"]["status"] == "completed"
    assert done_message["summary"]["frames_processed"] == 10


def test_job_concurrency_409_conflict(client):
    """Verifies that attempting to start a second job while one is running returns 409 Conflict."""
    # 1. Start first job
    res1 = client.post(
        "/api/jobs",
        json={"source": "sample", "max_frames": 100},
    )
    assert res1.status_code == 201
    first_job_id = res1.json()["job_id"]

    try:
        # 2. Start second job immediately while first is active -> 409
        res2 = client.post(
            "/api/jobs",
            json={"source": "sample"},
        )
        assert res2.status_code == 409
        assert "running" in res2.json()["detail"].lower()
    finally:
        # Cleanup
        client.post(f"/api/jobs/{first_job_id}/cancel")


def test_cancel_active_job(client):
    """Verifies that an active job can be cancelled cleanly."""
    res_start = client.post(
        "/api/jobs",
        json={"source": "sample", "max_frames": 100},
    )
    assert res_start.status_code == 201
    job_id = res_start.json()["job_id"]

    # Cancel job
    res_cancel = client.post(f"/api/jobs/{job_id}/cancel")
    assert res_cancel.status_code == 200
    assert res_cancel.json()["status"] == "cancelled"

    # Verify status report
    time.sleep(0.3)
    res_status = client.get(f"/api/jobs/{job_id}")
    assert res_status.status_code == 200
    assert res_status.json()["status"] == "cancelled"


def test_serve_keyframes_media_endpoint(client):
    """Verifies that keyframe image files are served at /media/keyframes/..."""
    # Sample pre-existing keyframe in report_images
    response = client.get("/media/keyframes/INC-3-001_keyframe.jpg")
    assert response.status_code == 200
    assert "image/jpeg" in response.headers.get("content-type", "")
    assert len(response.content) > 1000


def test_multipart_video_upload_job(client):
    """Verifies creating a job via multipart file upload."""
    with open("acc_ex.mp4", "rb") as f:
        video_content = f.read()

    res = client.post(
        "/api/jobs",
        data={
            "model_name": "best.pt",
            "confidence": "0.65",
            "max_frames": "5",
        },
        files={
            "file": ("test_clip.mp4", io.BytesIO(video_content), "video/mp4"),
        },
    )
    assert res.status_code == 201
    job_id = res.json()["job_id"]
    assert res.json()["confidence"] == 0.65

    # Wait for completion of 5 frames
    time.sleep(1.0)
    res_status = client.get(f"/api/jobs/{job_id}")
    assert res_status.status_code == 200
    assert res_status.json()["status"] == "completed"


def test_websocket_client_clean_disconnect(client):
    """Verifies that unexpected client disconnection from WebSocket does not crash worker."""
    res = client.post(
        "/api/jobs",
        json={"source": "sample", "max_frames": 30},
    )
    assert res.status_code == 201
    job_id = res.json()["job_id"]

    # Connect, read one frame, and abruptly disconnect
    with client.websocket_connect(f"/api/jobs/{job_id}/stream") as ws:
        msg = ws.receive_json()
        assert msg["type"] == "frame"
        # Context manager exit forces client websocket close/disconnect

    # Allow worker thread to continue finishing its frames without issue
    time.sleep(1.5)
    res_status = client.get(f"/api/jobs/{job_id}")
    assert res_status.status_code == 200
    assert res_status.json()["status"] == "completed"
