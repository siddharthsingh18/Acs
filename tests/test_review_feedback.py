"""
tests/test_review_feedback.py
Unit and integration tests for turning operator decisions into training data:
- Approve/Reject decision storage with keyframe and clip paths in ReviewFeedback.
- Fast Review Queue query filtering for uncertain detections (conf 0.3-0.6 or YOLO/VLM disagreement).
- Feedback export script generating YOLO hard negatives (empty .txt) and positive candidates.
- Manifest generation documenting who reviewed what, when, and source artifacts.
- Frozen test set protection assertion ensuring zero leakage.
"""
import json
import os
import shutil
import tempfile
import time
from pathlib import Path
from typing import Generator

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient
from sqlalchemy.pool import StaticPool
from sqlmodel import Session, SQLModel, create_engine, select

from backend.app.db.database import get_session
from backend.app.db.models import Incident, ReviewFeedback
from backend.app.main import app
from backend.app.services.incident_service import (
    get_review_queue_items,
    get_review_stats,
    record_review_decision,
    undo_review_decision,
    update_incident_status,
)
from ml.scripts.export_feedback import export_feedback_to_staging, parse_boxes_to_yolo_lines
from ml.scripts.retrain_workflow import assert_frozen_test_set_isolation


@pytest.fixture
def test_db_session() -> Generator[Session, None, None]:
    """Provides an isolated in-memory SQLite database session with StaticPool."""
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    SQLModel.metadata.create_all(engine)
    with Session(engine) as session:
        yield session


@pytest.fixture
def client(test_db_session: Session) -> Generator[TestClient, None, None]:
    """TestClient overriding get_session dependency with isolated test session."""
    def _override():
        yield test_db_session

    app.dependency_overrides[get_session] = _override
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()


# -----------------------------------------------------------------------------
# 1. Decision Storage Tests
# -----------------------------------------------------------------------------
def test_approve_reject_stores_label_with_keyframe_and_clip(test_db_session: Session):
    """Every Approve or Reject in dashboard stores label, keyframe, and clip in ReviewFeedback."""
    # Create sample incident with keyframe and video clips
    inc = Incident(
        incident_id="INC-STORE-001",
        start_timestamp="2026-10-18T12:00:00Z",
        end_timestamp="2026-10-18T12:00:05Z",
        class_name="car_car_accident",
        max_confidence=0.88,
        image_path="/data/keyframes/inc_001.jpg",
        keyframe_url="/media/keyframes/inc_001.jpg",
        clip_path="/data/clips/inc_001.mp4",
        clip_annotated_path="/data/clips/inc_001_ann.mp4",
        severity="high",
        status="pending",
        created_at=time.time(),
        duration=5.0,
    )
    test_db_session.add(inc)
    test_db_session.commit()

    # 1. Test Approve
    updated = update_incident_status(
        session=test_db_session,
        incident_id="INC-STORE-001",
        new_status="approved",
        reviewer_name="Officer Sarah",
        notes="Confirmed high impact collision",
    )
    assert updated is not None
    assert updated.status == "approved"
    assert updated.reviewed_label == "accident"

    fb_approved = test_db_session.exec(
        select(ReviewFeedback).where(ReviewFeedback.incident_id == "INC-STORE-001")
    ).first()
    assert fb_approved is not None
    assert fb_approved.decision == "accident"
    assert fb_approved.image_path == "/data/keyframes/inc_001.jpg"
    assert fb_approved.clip_path == "/data/clips/inc_001.mp4"
    assert fb_approved.clip_annotated_path == "/data/clips/inc_001_ann.mp4"
    assert fb_approved.reviewer_name == "Officer Sarah"
    assert fb_approved.notes == "Confirmed high impact collision"
    assert not fb_approved.is_exported

    # 2. Test Reject on another incident -> stores hard negative label
    inc2 = Incident(
        incident_id="INC-STORE-002",
        start_timestamp="2026-10-18T12:10:00Z",
        end_timestamp="2026-10-18T12:10:04Z",
        class_name="car_car_accident",
        max_confidence=0.45,
        image_path="/data/keyframes/inc_002.jpg",
        keyframe_url="/media/keyframes/inc_002.jpg",
        clip_path="/data/clips/inc_002.mp4",
        status="pending",
        created_at=time.time(),
        duration=4.0,
    )
    test_db_session.add(inc2)
    test_db_session.commit()

    updated2 = update_incident_status(
        session=test_db_session,
        incident_id="INC-STORE-002",
        new_status="rejected",
        reviewer_name="Officer John",
        notes="False alarm: normal lane change",
    )
    assert updated2 is not None
    assert updated2.status == "rejected"
    assert updated2.reviewed_label == "not_accident"

    fb_rejected = test_db_session.exec(
        select(ReviewFeedback).where(ReviewFeedback.incident_id == "INC-STORE-002")
    ).first()
    assert fb_rejected is not None
    assert fb_rejected.decision == "not_accident"
    assert fb_rejected.image_path == "/data/keyframes/inc_002.jpg"
    assert fb_rejected.clip_path == "/data/clips/inc_002.mp4"


# -----------------------------------------------------------------------------
# 2. Review Queue Query Filtering Tests
# -----------------------------------------------------------------------------
def test_review_queue_filtering_uncertain_detections(test_db_session: Session):
    """Review queue filters uncertain detections (conf 0.3-0.6 or YOLO/VLM disagreement)."""
    now = time.time()
    # 1. Clear accident (conf 0.95, high confidence, verified) -> Should NOT be in review queue
    inc_clear = Incident(
        incident_id="INC-CLEAR",
        start_timestamp="2026-10-18T10:00:00Z",
        end_timestamp="2026-10-18T10:00:05Z",
        class_name="car_car_accident",
        max_confidence=0.95,
        verification_status="auto_confirmed",
        vlm_confidence=0.92,
        image_path="test.jpg",
        keyframe_url="/media/keyframes/test.jpg",
        status="pending",
        created_at=now,
    )
    # 2. Borderline confidence (conf 0.45, within 0.3-0.6) -> MUST be in review queue
    inc_borderline = Incident(
        incident_id="INC-BORDERLINE",
        start_timestamp="2026-10-18T10:05:00Z",
        end_timestamp="2026-10-18T10:05:05Z",
        class_name="car_bike_accident",
        max_confidence=0.45,
        verification_status="unverified",
        image_path="test.jpg",
        keyframe_url="/media/keyframes/test.jpg",
        status="pending",
        created_at=now + 1,
    )
    # 3. Disagreement: High YOLO (0.75) but VLM flagged needs_review (0.15) -> MUST be in review queue
    inc_disagreement = Incident(
        incident_id="INC-DISAGREE",
        start_timestamp="2026-10-18T10:10:00Z",
        end_timestamp="2026-10-18T10:10:05Z",
        class_name="car_car_accident",
        max_confidence=0.75,
        verification_status="needs_review",
        vlm_confidence=0.15,
        vlm_reason="Vehicles passed without contact",
        image_path="test.jpg",
        keyframe_url="/media/keyframes/test.jpg",
        status="pending",
        created_at=now + 2,
    )
    # 4. Low sub-threshold noise (< 0.30) without disagreement -> Should NOT be in review queue
    inc_noise = Incident(
        incident_id="INC-NOISE",
        start_timestamp="2026-10-18T10:15:00Z",
        end_timestamp="2026-10-18T10:15:05Z",
        class_name="car_car_accident",
        max_confidence=0.20,
        verification_status="suppressed",
        image_path="test.jpg",
        keyframe_url="/media/keyframes/test.jpg",
        status="pending",
        created_at=now + 3,
    )

    test_db_session.add_all([inc_clear, inc_borderline, inc_disagreement, inc_noise])
    test_db_session.commit()

    queue_items = get_review_queue_items(test_db_session)
    queued_ids = [it["incident_id"] for it in queue_items]

    assert "INC-BORDERLINE" in queued_ids
    assert "INC-DISAGREE" in queued_ids
    assert "INC-CLEAR" not in queued_ids
    assert "INC-NOISE" not in queued_ids

    # Verify uncertainty reason explanation is attached
    disagree_item = next(it for it in queue_items if it["incident_id"] == "INC-DISAGREE")
    assert "YOLO/VLM verification flagged" in disagree_item["uncertainty_reason"]


def test_review_queue_fast_decision_and_undo(test_db_session: Session):
    """Operators can record Accident, Not accident, Wrong class, or Undo."""
    inc = Incident(
        incident_id="INC-ACTION-001",
        start_timestamp="2026-10-18T11:00:00Z",
        end_timestamp="2026-10-18T11:00:05Z",
        class_name="car_car_accident",
        max_confidence=0.55,
        image_path="test.jpg",
        keyframe_url="/media/keyframes/test.jpg",
        status="pending",
        created_at=time.time(),
    )
    test_db_session.add(inc)
    test_db_session.commit()

    # Record Wrong Class correction
    fb = record_review_decision(
        session=test_db_session,
        incident_id="INC-ACTION-001",
        decision="wrong_class",
        corrected_class="car_bike_accident",
        reviewer_name="Triage Officer",
        notes="Bicycle rider involved, not car-to-car",
    )
    assert fb is not None
    assert fb.decision == "wrong_class"
    assert fb.corrected_class == "car_bike_accident"

    inc_refreshed = test_db_session.exec(
        select(Incident).where(Incident.incident_id == "INC-ACTION-001")
    ).first()
    assert inc_refreshed.status == "approved"
    assert inc_refreshed.reviewed_label == "wrong_class"
    assert inc_refreshed.corrected_class == "car_bike_accident"

    # Test Undo reverts to pending and removes unexported feedback
    undone = undo_review_decision(test_db_session, "INC-ACTION-001")
    assert undone.status == "pending"
    assert undone.reviewed_label is None

    fb_after = test_db_session.exec(
        select(ReviewFeedback).where(ReviewFeedback.incident_id == "INC-ACTION-001")
    ).first()
    assert fb_after is None


# -----------------------------------------------------------------------------
# 3. Feedback Export Script & Hard Negatives Tests
# -----------------------------------------------------------------------------
def test_export_feedback_creates_hard_negatives_and_candidates():
    """export_feedback.py exports hard negatives as empty .txt files and candidates with YOLO boxes."""
    tmp_dir = Path(tempfile.mkdtemp(prefix="acs_test_export_"))
    try:
        db_path = tmp_dir / "test.db"
        engine = create_engine(f"sqlite:///{db_path}")
        SQLModel.metadata.create_all(engine)

        # Create dummy keyframe images
        img_neg = tmp_dir / "dummy_neg.jpg"
        img_pos = tmp_dir / "dummy_pos.jpg"
        cv2.imwrite(str(img_neg), np.zeros((100, 100, 3), dtype=np.uint8))
        cv2.imwrite(str(img_pos), np.zeros((100, 100, 3), dtype=np.uint8))

        with Session(engine) as session:
            # 1. Hard negative record (decision = not_accident)
            fb_neg = ReviewFeedback(
                incident_id="INC-EXP-NEG",
                decision="not_accident",
                original_class="car_car_accident",
                confidence=0.42,
                image_path=str(img_neg),
                reviewer_name="Inspector Alice",
                reviewed_at="2026-10-18T14:00:00Z",
                notes="Shadow caused false trigger",
                is_exported=False,
                created_at=time.time(),
            )
            # 2. Approved candidate record (decision = accident with custom boxes)
            boxes_json = json.dumps([
                {"class": "car_car_accident", "x_center": 0.5, "y_center": 0.5, "width": 0.3, "height": 0.3}
            ])
            fb_pos = ReviewFeedback(
                incident_id="INC-EXP-POS",
                decision="accident",
                original_class="car_car_accident",
                confidence=0.58,
                image_path=str(img_pos),
                reviewer_name="Inspector Bob",
                reviewed_at="2026-10-18T14:10:00Z",
                notes="Verified collision candidate",
                corrected_boxes=boxes_json,
                is_exported=False,
                created_at=time.time() + 1,
            )
            session.add_all([fb_neg, fb_pos])
            session.commit()

        staging_dir = tmp_dir / "staging"
        result = export_feedback_to_staging(
            db_path=db_path,
            staging_dir=staging_dir,
            export_all=True,
            dry_run=False,
        )

        assert result["status"] == "success"
        assert result["hard_negatives"] == 1
        assert result["candidates_approved"] == 1

        # Check Hard Negative YOLO label file (Must be 0 bytes empty file)
        neg_lbl = staging_dir / "labels" / "feedback_INC-EXP-NEG.txt"
        assert neg_lbl.exists(), "Hard negative label file must exist"
        assert os.path.getsize(neg_lbl) == 0, "Hard negative label file in YOLO format must be 0 bytes (empty)"

        # Check Candidate YOLO label file
        pos_lbl = staging_dir / "labels" / "feedback_INC-EXP-POS.txt"
        assert pos_lbl.exists(), "Candidate label file must exist"
        content = pos_lbl.read_text().strip()
        assert len(content) > 0, "Candidate label file must contain box annotations"
        parts = content.split()
        assert parts[0] == "6"  # class id for car_car_accident
        assert float(parts[1]) == pytest.approx(0.5, rel=1e-3)
        assert float(parts[2]) == pytest.approx(0.5, rel=1e-3)

        # Check Manifest JSON
        manifest_path = staging_dir / "manifest.json"
        assert manifest_path.exists(), "manifest.json must be generated"
        manifest = json.loads(manifest_path.read_text())
        assert manifest["total_items"] == 2
        assert manifest["hard_negatives"] == 1
        assert manifest["candidates"] == 1

        # Check reviewer provenance in manifest
        items_by_id = {it["incident_id"]: it for it in manifest["items"]}
        assert items_by_id["INC-EXP-NEG"]["reviewer_name"] == "Inspector Alice"
        assert items_by_id["INC-EXP-NEG"]["is_hard_negative"] is True
        assert items_by_id["INC-EXP-POS"]["reviewer_name"] == "Inspector Bob"
        assert items_by_id["INC-EXP-POS"]["is_hard_negative"] is False

    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


# -----------------------------------------------------------------------------
# 4. Frozen Test Set Protection Assertion Test
# -----------------------------------------------------------------------------
def test_frozen_test_set_protection_detects_leakage():
    """assert_frozen_test_set_isolation raises AssertionError if any candidate image matches frozen test set."""
    tmp_dir = Path(tempfile.mkdtemp(prefix="acs_test_isolation_"))
    try:
        # Create identical images
        img_a = tmp_dir / "test_frozen_sample.jpg"
        img_b = tmp_dir / "candidate_leaked.jpg"
        img_c = tmp_dir / "candidate_clean.jpg"

        blank = np.zeros((100, 100, 3), dtype=np.uint8)
        blank[20:50, 20:50] = 255  # Pattern
        cv2.imwrite(str(img_a), blank)
        cv2.imwrite(str(img_b), blank)  # Identical to frozen test image!

        # Different pattern for clean candidate
        clean_pat = np.zeros((100, 100, 3), dtype=np.uint8)
        clean_pat[60:90, 60:90] = 128
        cv2.imwrite(str(img_c), clean_pat)

        # Clean candidate passes
        assert_frozen_test_set_isolation(
            train_images=[img_c],
            val_images=[],
            frozen_test_images=[img_a],
            holdout_images=[],
            overlap_threshold=6,
        )

        # Leaked candidate MUST fail and raise AssertionError
        with pytest.raises(AssertionError) as exc_info:
            assert_frozen_test_set_isolation(
                train_images=[img_b],
                val_images=[],
                frozen_test_images=[img_a],
                holdout_images=[],
                overlap_threshold=6,
            )
        assert "CRITICAL DATA LEAKAGE" in str(exc_info.value)
        assert "The frozen test set must NEVER be trained on" in str(exc_info.value)

    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


# -----------------------------------------------------------------------------
# 5. Review API Endpoints Integration Test
# -----------------------------------------------------------------------------
def test_review_queue_api_endpoints(client: TestClient, test_db_session: Session):
    """Integration test for GET /api/review-queue, POST decision, and GET /api/review-queue/stats."""
    # Seed borderline incident
    inc = Incident(
        incident_id="INC-API-999",
        start_timestamp="2026-10-18T18:00:00Z",
        end_timestamp="2026-10-18T18:00:04Z",
        class_name="car_car_accident",
        max_confidence=0.48,
        image_path="/media/keyframes/api_999.jpg",
        keyframe_url="/media/keyframes/api_999.jpg",
        status="pending",
        created_at=time.time(),
    )
    test_db_session.add(inc)
    test_db_session.commit()

    # 1. Fetch review queue
    res = client.get("/api/review-queue")
    assert res.status_code == 200
    queue_data = res.json()
    assert queue_data["total"] >= 1
    item = next(it for it in queue_data["items"] if it["incident_id"] == "INC-API-999")
    assert item["class_name"] == "car_car_accident"
    assert "Borderline YOLO confidence" in item["uncertainty_reason"]

    # 2. Submit Not Accident (Hard negative)
    post_res = client.post(
        "/api/review-queue/INC-API-999/decision",
        json={
            "decision": "not_accident",
            "reviewer_name": "Supervisor Chen",
            "notes": "Non-accident near miss",
        },
    )
    assert post_res.status_code == 200
    fb_data = post_res.json()
    assert fb_data["decision"] == "not_accident"
    assert fb_data["reviewer_name"] == "Supervisor Chen"

    # 3. Fetch stats
    stats_res = client.get("/api/review-queue/stats")
    assert stats_res.status_code == 200
    stats = stats_res.json()
    assert stats["total_reviewed"] >= 1
    assert stats["hard_negatives"] >= 1
    assert stats["staging_pending_export"] >= 1
