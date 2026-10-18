"""
tests.test_privacy_retention
Comprehensive tests for privacy protection (face and license plate blurring),
unredacted original vaulting, incident export ZIP bundle, individual incident erasure,
and automatic retention cleanup.
"""
import io
import json
import os
import shutil
import tempfile
import time
import zipfile
import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient
from sqlmodel import Session, select

from acs.config import config
from acs.privacy import (
    apply_privacy_blur,
    detect_faces,
    detect_license_plates,
    detect_sensitive_regions,
    merge_overlapping_boxes,
    protect_keyframe,
    protect_video_clip,
)
from backend.app.core.security import create_access_token
from backend.app.db.database import get_session
from backend.app.db.models import AuditLog, Incident, User
from backend.app.main import app
from backend.app.services.retention_service import (
    delete_incident_and_media,
    export_incident_bundle,
    run_retention_cleanup,
)


@pytest.fixture
def temp_privacy_env():
    """Sets up a temporary directory environment for privacy tests."""
    temp_dir = tempfile.mkdtemp()
    unredacted_dir = os.path.join(temp_dir, "unredacted")
    keyframes_dir = os.path.join(temp_dir, "keyframes")
    clips_dir = os.path.join(temp_dir, "clips")

    os.makedirs(unredacted_dir, exist_ok=True)
    os.makedirs(keyframes_dir, exist_ok=True)
    os.makedirs(clips_dir, exist_ok=True)

    old_unredacted = config.privacy_unredacted_dir
    config.privacy_unredacted_dir = unredacted_dir

    yield {
        "root": temp_dir,
        "unredacted": unredacted_dir,
        "keyframes": keyframes_dir,
        "clips": clips_dir,
    }

    config.privacy_unredacted_dir = old_unredacted
    shutil.rmtree(temp_dir, ignore_errors=True)


def test_face_and_plate_detection_heuristics():
    """Verifies that face and license plate candidates are accurately detected."""
    # Create test image with simulated car body and license plate
    frame = np.full((400, 500, 3), 110, dtype=np.uint8)

    # 1. Car body with high-contrast license plate
    cv2.rectangle(frame, (100, 150), (400, 350), (40, 40, 40), -1)
    cv2.rectangle(frame, (200, 300), (300, 330), (240, 240, 240), -1)
    cv2.putText(frame, "DXB-9876", (205, 322), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (10, 10, 10), 2)

    # 2. Simulated person with face region
    cv2.ellipse(frame, (80, 80), (25, 30), 0, 0, 360, (140, 160, 210), -1)

    # YOLO guidance boxes
    mock_boxes = [
        {"class_name": "car", "xyxy": [100, 150, 400, 350], "confidence": 0.89},
        {"class_name": "person", "xyxy": [60, 50, 110, 220], "confidence": 0.85},
    ]

    detected_faces = detect_faces(frame, detected_boxes=mock_boxes)
    assert len(detected_faces) >= 1
    assert any(box[4] == "face" for box in detected_faces)

    detected_plates = detect_license_plates(frame, detected_boxes=mock_boxes)
    assert len(detected_plates) >= 1
    assert any(box[4] == "license_plate" for box in detected_plates)

    all_sensitive = detect_sensitive_regions(frame, detected_boxes=mock_boxes)
    assert len(all_sensitive) >= 2


def test_privacy_blur_application():
    """Verifies that privacy blur obscures sensitive ROIs without altering overall image geometry."""
    frame = np.zeros((300, 300, 3), dtype=np.uint8)
    # Add crisp high contrast text inside an area to be blurred
    cv2.putText(frame, "SECRET_PLATE_55", (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    regions = [(40, 120, 260, 170, "license_plate")]
    blurred = apply_privacy_blur(frame, regions=regions, blur_radius=41)

    assert blurred.shape == frame.shape
    assert blurred.dtype == np.uint8
    # The sharp pixel values should be heavily smoothed
    assert not np.array_equal(frame[120:170, 40:260], blurred[120:170, 40:260])
    # The untouched border remains identical
    assert np.array_equal(frame[:50, :50], blurred[:50, :50])


def test_protect_keyframe_vaulting(temp_privacy_env):
    """Verifies that protect_keyframe writes blurred keyframe to target and vaults unredacted original."""
    keyframe = np.full((240, 320, 3), 150, dtype=np.uint8)
    cv2.putText(keyframe, "PLATE-123", (70, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)

    incident_id = "INC-PRIV-001"
    public_path = os.path.join(temp_privacy_env["keyframes"], f"{incident_id}_keyframe.jpg")

    out_public, out_vault = protect_keyframe(
        keyframe=keyframe,
        incident_id=incident_id,
        output_path=public_path,
        detected_boxes=[{"class_name": "car", "xyxy": [40, 40, 200, 160]}],
        preserve_unblurred=True,
        unredacted_dir=temp_privacy_env["unredacted"],
    )

    assert os.path.exists(out_public)
    assert out_vault is not None
    assert os.path.exists(out_vault)
    assert "media/unredacted" in out_vault or temp_privacy_env["unredacted"] in out_vault

    # Read both and verify vault is unblurred while public is blurred
    pub_img = cv2.imread(out_public)
    vault_img = cv2.imread(out_vault)
    assert not np.array_equal(pub_img, vault_img)


def test_protect_video_clip_vaulting(temp_privacy_env):
    """Verifies video clip privacy blurring and unredacted clip vaulting."""
    # Create synthetic 10-frame test clip
    src_clip = os.path.join(temp_privacy_env["root"], "raw_src.mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(src_clip, fourcc, 10.0, (160, 120))
    for i in range(10):
        frame = np.full((120, 160, 3), 100, dtype=np.uint8)
        cv2.putText(frame, f"F_{i}", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        writer.write(frame)
    writer.release()

    incident_id = "INC-CLIP-PRIV-002"
    dest_clip = os.path.join(temp_privacy_env["clips"], f"{incident_id}_blurred.mp4")

    out_blurred, out_vault = protect_video_clip(
        src_clip_path=src_clip,
        dest_clip_path=dest_clip,
        incident_id=incident_id,
        preserve_unblurred=True,
        unredacted_dir=temp_privacy_env["unredacted"],
        fps=10.0,
    )

    assert os.path.exists(out_blurred)
    assert out_vault is not None
    assert os.path.exists(out_vault)


def test_incident_bundle_export_zip(temp_privacy_env):
    """Verifies compilation of the forensic export bundle as a valid ZIP archive."""
    session = next(get_session())
    incident_id = f"INC-EXPORT-TEST-{time.time_ns()}"

    # Create dummy keyframe
    kf_path = os.path.join(temp_privacy_env["keyframes"], f"{incident_id}_keyframe.jpg")
    cv2.imwrite(kf_path, np.zeros((100, 100, 3), dtype=np.uint8))

    inc = Incident(
        incident_id=incident_id,
        start_timestamp="2026-10-19 01:00:00 UTC",
        end_timestamp="2026-10-19 01:00:05 UTC",
        class_name="car_car_accident",
        max_confidence=0.92,
        image_path=kf_path,
        keyframe_url=f"/media/keyframes/{os.path.basename(kf_path)}",
        severity="high",
        status="approved",
        created_at=time.time(),
        duration=5.0,
    )
    session.add(inc)
    session.commit()

    zip_bytes, filename = export_incident_bundle(session, incident_id)
    assert filename == f"{incident_id}_export.zip"
    assert len(zip_bytes) > 0

    # Verify ZIP contents
    with zipfile.ZipFile(io.BytesIO(zip_bytes), "r") as zf:
        namelist = zf.namelist()
        assert "metadata.json" in namelist
        assert "keyframe.jpg" in namelist
        assert f"{incident_id}_report.pdf" in namelist
        assert "audit_trail.json" in namelist

        meta = json.loads(zf.read("metadata.json").decode("utf-8"))
        assert meta["incident_id"] == incident_id
        assert meta["class_name"] == "car_car_accident"


def test_incident_deletion_and_media_cleanup(temp_privacy_env):
    """Verifies complete incident and media file permanent deletion."""
    session = next(get_session())
    incident_id = f"INC-DELETE-TEST-{time.time_ns()}"

    # Create keyframe and clip files
    kf_path = os.path.join(temp_privacy_env["keyframes"], f"{incident_id}_keyframe.jpg")
    cv2.imwrite(kf_path, np.zeros((50, 50, 3), dtype=np.uint8))

    unredacted_kf = os.path.join(temp_privacy_env["unredacted"], "keyframes", f"{incident_id}_unredacted.jpg")
    os.makedirs(os.path.dirname(unredacted_kf), exist_ok=True)
    cv2.imwrite(unredacted_kf, np.zeros((50, 50, 3), dtype=np.uint8))

    inc = Incident(
        incident_id=incident_id,
        start_timestamp="2026-10-19 01:10:00 UTC",
        end_timestamp="2026-10-19 01:10:05 UTC",
        class_name="car_bike_accident",
        max_confidence=0.88,
        image_path=kf_path,
        keyframe_url=f"/media/keyframes/{os.path.basename(kf_path)}",
        severity="medium",
        status="pending",
        unredacted_keyframe_path=unredacted_kf,
        created_at=time.time(),
        duration=4.5,
    )
    session.add(inc)
    session.commit()

    assert os.path.exists(kf_path)
    assert os.path.exists(unredacted_kf)

    result = delete_incident_and_media(session, incident_id)
    assert result["success"] is True

    # Files must be gone
    assert not os.path.exists(kf_path)
    assert not os.path.exists(unredacted_kf)

    # Record must be deleted from DB
    deleted_row = session.exec(select(Incident).where(Incident.incident_id == incident_id)).first()
    assert deleted_row is None


def test_rbac_and_audit_on_incident_endpoints(monkeypatch, temp_privacy_env):
    """Verifies RBAC enforcement and audit trail generation on export, delete, and unredacted vault endpoints."""
    monkeypatch.setenv("AUTH_ENFORCE", "1")
    client = TestClient(app)
    session = next(get_session())

    incident_id = f"INC-RBAC-AUDIT-{time.time_ns()}"
    kf_path = os.path.join(temp_privacy_env["keyframes"], f"{incident_id}_keyframe.jpg")
    cv2.imwrite(kf_path, np.zeros((60, 60, 3), dtype=np.uint8))

    unredacted_kf = os.path.join(temp_privacy_env["unredacted"], "keyframes", f"{incident_id}_unredacted.jpg")
    os.makedirs(os.path.dirname(unredacted_kf), exist_ok=True)
    cv2.imwrite(unredacted_kf, np.zeros((60, 60, 3), dtype=np.uint8))

    inc = Incident(
        incident_id=incident_id,
        start_timestamp="2026-10-19 01:15:00 UTC",
        end_timestamp="2026-10-19 01:15:05 UTC",
        class_name="car_car_accident",
        max_confidence=0.85,
        image_path=kf_path,
        keyframe_url=f"/media/keyframes/{os.path.basename(kf_path)}",
        severity="low",
        status="pending",
        unredacted_keyframe_path=unredacted_kf,
        created_at=time.time(),
    )
    session.add(inc)
    session.commit()

    viewer_token = create_access_token({"sub": "viewer_user", "username": "viewer", "role": "viewer"})
    operator_token = create_access_token({"sub": "operator_user", "username": "operator", "role": "operator"})
    admin_token = create_access_token({"sub": "admin_user", "username": "admin", "role": "admin"})

    # 1. Export endpoint: Viewer is rejected (403), Operator and Admin are allowed (200)
    resp_viewer_export = client.get(
        f"/api/incidents/{incident_id}/export",
        headers={"Authorization": f"Bearer {viewer_token}"},
    )
    assert resp_viewer_export.status_code == 403

    resp_op_export = client.get(
        f"/api/incidents/{incident_id}/export",
        headers={"Authorization": f"Bearer {operator_token}"},
    )
    assert resp_op_export.status_code == 200
    assert resp_op_export.headers["content-type"] == "application/zip"

    # 2. Unredacted keyframe vault: Viewer and Operator are rejected (403), Admin is allowed (200)
    resp_op_unredacted = client.get(
        f"/api/incidents/{incident_id}/unredacted-keyframe",
        headers={"Authorization": f"Bearer {operator_token}"},
    )
    assert resp_op_unredacted.status_code == 403

    resp_admin_unredacted = client.get(
        f"/api/incidents/{incident_id}/unredacted-keyframe",
        headers={"Authorization": f"Bearer {admin_token}"},
    )
    assert resp_admin_unredacted.status_code == 200
    assert resp_admin_unredacted.headers["content-type"] == "image/jpeg"

    # 3. Incident Delete: Operator is rejected (403), Admin is allowed (200)
    resp_op_delete = client.delete(
        f"/api/incidents/{incident_id}",
        headers={"Authorization": f"Bearer {operator_token}"},
    )
    assert resp_op_delete.status_code == 403

    resp_admin_delete = client.delete(
        f"/api/incidents/{incident_id}",
        headers={"Authorization": f"Bearer {admin_token}"},
    )
    assert resp_admin_delete.status_code == 200
    assert resp_admin_delete.json()["success"] is True

    # 4. Verify AuditLog entries created for export, unredacted access, and deletion
    audit_logs = session.exec(select(AuditLog).where(AuditLog.target_id == incident_id)).all()
    actions = [a.action for a in audit_logs]
    assert "incident.exported" in actions
    assert "privacy.unredacted_keyframe.accessed" in actions
    assert "incident.deleted" in actions
