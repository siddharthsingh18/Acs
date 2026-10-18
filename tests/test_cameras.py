"""
tests.test_cameras
Comprehensive test suite for the Live Camera subsystem:
1. CameraSource abstraction (FileCameraSource, WebcamCameraSource, StreamCameraSource, factory).
2. Credential encryption, decryption, and secure masking (never exposing credentials in logs or API responses).
3. CameraManager isolated worker execution, tracker & EventManager isolation, frame-skip policy, and auto-reconnect backoff.
4. Camera CRUD API endpoints and WebSocket live streaming.
5. Fake RTSP live stream integration using mediamtx + ffmpeg.
"""
import asyncio
import os
import subprocess
import time
from typing import Generator
import pytest
from fastapi.testclient import TestClient
from sqlmodel import Session, SQLModel, create_engine

from backend.app.db.database import get_session
from backend.app.db.models import Camera
from backend.app.main import app
from backend.app.services.camera_manager import CameraManager, CameraWorker, camera_manager
from backend.app.services.camera_source import (
    FileCameraSource,
    StreamCameraSource,
    WebcamCameraSource,
    create_camera_source,
    decrypt_credentials,
    encrypt_credentials,
    mask_credentials,
)

from backend.app.db.database import init_db

SAMPLE_VIDEO = os.path.join(os.path.dirname(os.path.dirname(__file__)), "acc_ex.mp4")


@pytest.fixture(name="client")
def client_fixture() -> Generator[TestClient, None, None]:
    """TestClient fixture with initialized database."""
    init_db()
    with TestClient(app) as test_client:
        yield test_client


# =============================================================================
# 1. CameraSource & Factory Tests
# =============================================================================
def test_camera_source_factory_routing():
    """Verifies factory correctly routes source string to proper CameraSource class."""
    src_webcam_digit = create_camera_source("0")
    assert isinstance(src_webcam_digit, WebcamCameraSource)
    assert src_webcam_digit.device_index == 0
    assert src_webcam_digit.source_type == "webcam"

    src_webcam_prefix = create_camera_source("webcam:2")
    assert isinstance(src_webcam_prefix, WebcamCameraSource)
    assert src_webcam_prefix.device_index == 2

    src_rtsp = create_camera_source("rtsp://admin:pass@192.168.1.10:554/live")
    assert isinstance(src_rtsp, StreamCameraSource)
    assert src_rtsp.source_type == "rtsp"

    src_http = create_camera_source("http://192.168.1.50/mjpeg")
    assert isinstance(src_http, StreamCameraSource)
    assert src_http.source_type == "http"

    src_file = create_camera_source(SAMPLE_VIDEO)
    assert isinstance(src_file, FileCameraSource)
    assert src_file.source_type == "file"


def test_file_camera_source_reading_and_looping():
    """Verifies FileCameraSource can read frames and loop when reaching end of file."""
    source = FileCameraSource(SAMPLE_VIDEO, loop=True)
    assert source.open() is True
    assert source.is_opened() is True

    # Read multiple frames
    for _ in range(5):
        ok, frame = source.read()
        assert ok is True
        assert frame is not None
        assert len(frame.shape) == 3

    source.release()
    assert source.is_opened() is False


# =============================================================================
# 2. Security, Encryption & Masking Tests
# =============================================================================
def test_mask_credentials_security():
    """Verifies RTSP and HTTP credentials are strictly redacted from URLs."""
    raw_rtsp = "rtsp://admin:secret123@10.0.0.50:554/h264/stream1"
    masked_rtsp = mask_credentials(raw_rtsp)
    assert "secret123" not in masked_rtsp
    assert "admin" not in masked_rtsp
    assert masked_rtsp == "rtsp://***:***@10.0.0.50:554/h264/stream1"

    raw_http = "http://operator:MyPassword99@traffic.city.gov/live.mjpg?res=high"
    masked_http = mask_credentials(raw_http)
    assert "MyPassword99" not in masked_http
    assert masked_http.startswith("http://***:***@traffic.city.gov/live.mjpg")

    # Clean URL without credentials should remain untouched
    clean_rtsp = "rtsp://192.168.1.100:554/stream"
    assert mask_credentials(clean_rtsp) == clean_rtsp

    # File path remains untouched
    assert mask_credentials("/path/to/video.mp4") == "/path/to/video.mp4"


def test_credential_encryption_roundtrip():
    """Verifies Fernet encryption and decryption preserves credentials faithfully."""
    sensitive_url = "rtsp://security_user:HighlyConfidentialToken%21@172.16.0.4:8554/stream"
    encrypted = encrypt_credentials(sensitive_url)
    assert encrypted != sensitive_url
    assert "HighlyConfidentialToken" not in encrypted

    decrypted = decrypt_credentials(encrypted)
    assert decrypted == sensitive_url


# =============================================================================
# 3. CameraManager & Worker State Isolation Tests
# =============================================================================
def test_worker_dedicated_state_isolation():
    """Verifies each worker has its own isolated detector and AccidentEventManager."""
    loop = asyncio.new_event_loop()
    w1 = CameraWorker("cam-1", "Camera North", SAMPLE_VIDEO, confidence_override=0.60, loop=loop)
    w2 = CameraWorker("cam-2", "Camera South", SAMPLE_VIDEO, confidence_override=0.75, loop=loop)

    assert w1.detector is not w2.detector
    assert w1.event_manager is not w2.event_manager
    assert w1.detector.accident_confidence_threshold == 0.60
    assert w2.detector.accident_confidence_threshold == 0.75

    # Event manager internal state must not cross-contaminate
    w1.event_manager.consecutive_count = 3
    assert w2.event_manager.consecutive_count == 0

    loop.close()


def test_frame_skip_policy_under_load():
    """Verifies that slow subscriber queues cause frames to be dropped, never queued."""
    loop = asyncio.new_event_loop()
    worker = CameraWorker("cam-test", "Test Camera", SAMPLE_VIDEO, loop=loop)

    # Queue with maxsize 1
    slow_queue = asyncio.Queue(maxsize=1)
    worker.subscribe(slow_queue)

    # First message fills the queue
    worker._broadcast_message({"type": "frame", "frame_idx": 1})
    loop.run_until_complete(asyncio.sleep(0.01))

    # Second message must trigger frame-skip / drop rather than unbounded growth
    worker._broadcast_message({"type": "frame", "frame_idx": 2})
    loop.run_until_complete(asyncio.sleep(0.01))

    assert worker.dropped_frames >= 1
    assert slow_queue.qsize() <= 1

    worker.unsubscribe(slow_queue)
    loop.close()


# =============================================================================
# 4. REST API CRUD & Health Endpoints
# =============================================================================
def test_camera_crud_lifecycle(client: TestClient):
    """Verifies create, list, get, update, and delete endpoints for /api/cameras."""
    # 1. Create Camera with sensitive credentials in URL
    raw_url = "rtsp://testuser:supersecretpass@127.0.0.1:8554/traffic"
    create_payload = {
        "name": "Downtown Intersection 5",
        "source_url": raw_url,
        "location_lat": 37.7749,
        "location_lng": -122.4194,
        "enabled": False,  # disabled so worker doesn't connect during CRUD test
        "confidence_override": 0.65,
    }
    resp = client.post("/api/cameras", json=create_payload)
    assert resp.status_code == 201
    data = resp.json()
    cam_id = data["id"]
    assert data["name"] == "Downtown Intersection 5"
    assert data["source_type"] == "rtsp"
    # SENSITIVE CREDENTIALS MUST NEVER BE EXPOSED IN RESPONSE
    assert "supersecretpass" not in data["source_url"]
    assert "testuser" not in data["source_url"]
    assert data["source_url"] == "rtsp://***:***@127.0.0.1:8554/traffic"
    assert data["confidence_override"] == 0.65

    # 2. List Cameras
    list_resp = client.get("/api/cameras")
    assert list_resp.status_code == 200
    cameras = list_resp.json()
    assert any(c["id"] == cam_id for c in cameras)
    for c in cameras:
        assert "supersecretpass" not in c["source_url"]

    # 3. Get Single Camera
    get_resp = client.get(f"/api/cameras/{cam_id}")
    assert get_resp.status_code == 200
    assert get_resp.json()["id"] == cam_id
    assert "supersecretpass" not in get_resp.json()["source_url"]

    # 4. Update Camera
    update_resp = client.put(
        f"/api/cameras/{cam_id}",
        json={"name": "Downtown Intersection 5 (Renamed)", "confidence_override": 0.70},
    )
    assert update_resp.status_code == 200
    assert update_resp.json()["name"] == "Downtown Intersection 5 (Renamed)"
    assert update_resp.json()["confidence_override"] == 0.70

    # 5. Check Health endpoint
    health_resp = client.get(f"/api/cameras/{cam_id}/health")
    assert health_resp.status_code == 200
    health_data = health_resp.json()
    assert "connected" in health_data
    assert "fps" in health_data
    assert "dropped_frames" in health_data

    # 6. Delete Camera
    del_resp = client.delete(f"/api/cameras/{cam_id}")
    assert del_resp.status_code == 204

    # Verify deleted
    assert client.get(f"/api/cameras/{cam_id}").status_code == 404


# =============================================================================
# 5. Live WebSocket Streaming Integration Test
# =============================================================================
def test_camera_stream_websocket(client: TestClient):
    """Verifies WebSocket /api/cameras/{id}/stream receives live frames and status."""
    # Create an active camera using sample video file
    create_payload = {
        "name": "Live Video Feed Test",
        "source_url": SAMPLE_VIDEO,
        "enabled": True,
        "confidence_override": 0.50,
    }
    resp = client.post("/api/cameras", json=create_payload)
    assert resp.status_code == 201
    cam_id = resp.json()["id"]

    try:
        # Connect to stream websocket
        with client.websocket_connect(f"/api/cameras/{cam_id}/stream") as ws:
            # Wait for first message (either frame or status)
            data = ws.receive_json()
            assert "type" in data
            assert data["type"] in ("frame", "status", "heartbeat")
            if data["type"] == "frame":
                assert "frame" in data
                assert data["frame"].startswith("data:image/jpeg;base64,")
                assert "fps" in data
    finally:
        client.delete(f"/api/cameras/{cam_id}")


# =============================================================================
# 6. Fake RTSP Stream Test using MediaMTX and FFmpeg
# =============================================================================
def test_fake_rtsp_stream_with_mediamtx_and_ffmpeg():
    """Starts MediaMTX and loops a video via FFmpeg to test genuine RTSP stream capture."""
    mediamtx_proc = None
    ffmpeg_proc = None
    rtsp_url = "rtsp://127.0.0.1:8554/mystream"

    try:
        # 1. Start MediaMTX with default homebrew configuration if available
        config_file = "/opt/homebrew/etc/mediamtx/mediamtx.yml"
        cmd = ["mediamtx", config_file] if os.path.exists(config_file) else ["mediamtx"]
        mediamtx_proc = subprocess.Popen(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        time.sleep(1.5)

        # 2. Publish looped video stream to MediaMTX using ffmpeg
        ffmpeg_proc = subprocess.Popen(
            [
                "ffmpeg",
                "-re",
                "-stream_loop",
                "-1",
                "-i",
                SAMPLE_VIDEO,
                "-c",
                "copy",
                "-f",
                "rtsp",
                rtsp_url,
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        time.sleep(2.5)

        # 3. Create StreamCameraSource and verify connection and reading
        source = StreamCameraSource(rtsp_url, name="RTSP Test Cam")
        opened = source.open()
        assert opened is True
        assert source.is_opened() is True

        # Read frames from the live RTSP stream
        read_success = False
        for _ in range(10):
            ret, frame = source.read()
            if ret and frame is not None and frame.size > 0:
                read_success = True
                break
            time.sleep(0.1)

        assert read_success is True, "Failed to read valid video frame from fake RTSP stream."
        source.release()
        assert source.is_opened() is False

    finally:
        if ffmpeg_proc:
            ffmpeg_proc.terminate()
            ffmpeg_proc.wait(timeout=2.0)
        if mediamtx_proc:
            mediamtx_proc.terminate()
            mediamtx_proc.wait(timeout=2.0)
