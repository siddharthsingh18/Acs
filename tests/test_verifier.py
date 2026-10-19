"""
tests.test_verifier
Comprehensive unit and integration test suite for the ACS Verification Stage:
1. Cheap motion/physics pre-check (overlap, velocity change, stopped post-collision).
2. Strict structured VLM prompt and schema parsing.
3. Multi-frame keyframe collection (before, peak, after).
4. Tri-state decision matrix (auto_confirmed, needs_review, suppressed, unverified).
5. Operational resilience: timeouts, retries, caching, daily call budget, and offline fallbacks.
6. Alert manager verification gate enforcement.
"""
import os
import tempfile
from typing import List
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from acs.alerts import send_incident_alert
from acs.detector import AccidentEventManager, AccidentIncident, DetectionBox, DetectionResult
from acs.llm import VerificationReportModel, verify_incident_frames_vlm
from acs.verifier import (
    IncidentVerificationCache,
    PhysicsCheckResult,
    VLMBudgetManager,
    evaluate_physics_motion,
    verify_incident,
)


# -----------------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------------
@pytest.fixture
def sample_frames() -> List[np.ndarray]:
    """Generates 3 synthetic BGR video frames."""
    f1 = np.full((100, 100, 3), 50, dtype=np.uint8)
    f2 = np.full((100, 100, 3), 150, dtype=np.uint8)
    f3 = np.full((100, 100, 3), 200, dtype=np.uint8)
    return [f1, f2, f3]


@pytest.fixture
def sample_incident(sample_frames: List[np.ndarray]) -> AccidentIncident:
    """Creates a sample AccidentIncident for testing."""
    return AccidentIncident(
        incident_id="INC-TEST-001",
        start_timestamp="2026-10-18 12:00:00 UTC",
        end_timestamp="2026-10-18 12:00:05 UTC",
        class_name="car_car_accident",
        max_confidence=0.88,
        image_path="/tmp/fake_keyframe.jpg",
        start_time=100.0,
        end_time=105.0,
        keyframe=sample_frames[1],
        verification_frames=sample_frames,
    )


# -----------------------------------------------------------------------------
# 1. Cheap Motion / Physics Check Tests
# -----------------------------------------------------------------------------
class TestPhysicsMotionCheck:
    """Evaluates the heuristic physics and motion pre-check."""

    def test_overlapping_vehicle_boxes_passes(self):
        """Bounding-box overlap between vehicles must pass physics check."""
        box_a = DetectionBox(bbox=(10, 10, 50, 50), confidence=0.9, class_id=3, class_name="car", is_accident=False)
        box_b = DetectionBox(bbox=(30, 30, 70, 70), confidence=0.9, class_id=3, class_name="car", is_accident=False)

        history = [
            (0.0, [box_a, DetectionBox(bbox=(100, 100, 140, 140), confidence=0.8, class_id=3, class_name="car", is_accident=False)]),
            (0.1, [box_a, box_b]),  # Overlapping boxes (IoU > 0.05)
            (0.2, [box_a, box_b]),
        ]

        result: PhysicsCheckResult = evaluate_physics_motion(history)
        assert result.passed is True
        assert result.overlap_detected is True
        assert result.max_iou > 0.05

    def test_sudden_velocity_deceleration_passes(self):
        """Vehicle with significant speed abruptly stopping must pass physics check."""
        # Vehicle track 1 moving fast then stopping abruptly
        history = [
            (0.0, [DetectionBox(bbox=(10, 10, 30, 30), confidence=0.9, class_id=3, class_name="car", is_accident=False, track_id=1)]),
            (0.1, [DetectionBox(bbox=(35, 10, 55, 30), confidence=0.9, class_id=3, class_name="car", is_accident=False, track_id=1)]), # dx=25
            (0.2, [DetectionBox(bbox=(37, 10, 57, 30), confidence=0.9, class_id=3, class_name="car", is_accident=False, track_id=1)]), # dx=2 (sudden stop)
            (0.3, [DetectionBox(bbox=(37, 10, 57, 30), confidence=0.9, class_id=3, class_name="car", is_accident=False, track_id=1)]),
        ]

        result = evaluate_physics_motion(history)
        assert result.passed is True
        assert result.sudden_velocity_change is True

    def test_stopped_vehicles_after_overlap_passes(self):
        """Stationary vehicle following an overlap event must pass physics check."""
        history = [
            # Frame 0: moving toward each other
            (0.0, [
                DetectionBox(bbox=(10, 20, 30, 40), confidence=0.8, class_id=3, class_name="car", is_accident=False, track_id=1),
                DetectionBox(bbox=(50, 20, 70, 40), confidence=0.8, class_id=3, class_name="car", is_accident=False, track_id=2),
            ]),
            # Frame 1: overlap
            (0.1, [
                DetectionBox(bbox=(25, 20, 45, 40), confidence=0.9, class_id=3, class_name="car", is_accident=False, track_id=1),
                DetectionBox(bbox=(35, 20, 55, 40), confidence=0.9, class_id=3, class_name="car", is_accident=False, track_id=2),
            ]),
            # Frame 2 & 3: stopped in place
            (0.2, [
                DetectionBox(bbox=(25, 20, 45, 40), confidence=0.9, class_id=3, class_name="car", is_accident=False, track_id=1),
                DetectionBox(bbox=(35, 20, 55, 40), confidence=0.9, class_id=3, class_name="car", is_accident=False, track_id=2),
            ]),
            (0.3, [
                DetectionBox(bbox=(25, 20, 45, 40), confidence=0.9, class_id=3, class_name="car", is_accident=False, track_id=1),
                DetectionBox(bbox=(35, 20, 55, 40), confidence=0.9, class_id=3, class_name="car", is_accident=False, track_id=2),
            ]),
        ]

        result = evaluate_physics_motion(history)
        assert result.passed is True
        assert result.overlap_detected is True
        assert result.stopped_after_overlap is True

    def test_static_reflection_fails_physics_check(self):
        """A single static reflection (no overlap, no sudden deceleration) must fail physics check."""
        # Single box remaining stationary on pavement with 0 overlap and 0 motion
        history = [
            (0.0, [DetectionBox(bbox=(100, 100, 130, 130), confidence=0.8, class_id=5, class_name="car_car_accident", is_accident=True)]),
            (0.1, [DetectionBox(bbox=(100, 100, 130, 130), confidence=0.8, class_id=5, class_name="car_car_accident", is_accident=True)]),
            (0.2, [DetectionBox(bbox=(100, 100, 130, 130), confidence=0.8, class_id=5, class_name="car_car_accident", is_accident=True)]),
        ]

        result = evaluate_physics_motion(history)
        assert result.passed is False
        assert result.overlap_detected is False
        assert result.sudden_velocity_change is False
        assert "Motion/physics check failed" in result.reason

    def test_smooth_non_interacting_traffic_fails(self):
        """Single car smoothly cruising without interactions or deceleration must fail physics check."""
        history = [
            (0.0, [DetectionBox(bbox=(10, 20, 30, 40), confidence=0.8, class_id=3, class_name="car", is_accident=False, track_id=1)]),
            (0.1, [DetectionBox(bbox=(15, 20, 35, 40), confidence=0.8, class_id=3, class_name="car", is_accident=False, track_id=1)]),
            (0.2, [DetectionBox(bbox=(20, 20, 40, 40), confidence=0.8, class_id=3, class_name="car", is_accident=False, track_id=1)]),
            (0.3, [DetectionBox(bbox=(25, 20, 45, 40), confidence=0.8, class_id=3, class_name="car", is_accident=False, track_id=1)]),
        ]

        result = evaluate_physics_motion(history)
        assert result.passed is False

    def test_insufficient_history_gracefully_defaults_to_true(self):
        """History with fewer than 2 frames defaults to passed to avoid accidental suppression."""
        assert evaluate_physics_motion([]).passed is True
        assert evaluate_physics_motion([(0.0, [])]).passed is True


# -----------------------------------------------------------------------------
# 2. VLM Structured Prompt & Response Schema Tests
# -----------------------------------------------------------------------------
class TestVLMStructuredVerification:
    """Tests for the VLM prompt construction and structured response model."""

    def test_verification_report_model_schema_validation(self):
        """VerificationReportModel enforces valid field types, ranges, and literals."""
        valid_data = {
            "is_accident": True,
            "confidence": 0.92,
            "reason": "Visible collision impact between two sedans at intersection",
            "severity": "high",
            "vehicles_involved": 2,
        }
        model = VerificationReportModel(**valid_data)
        assert model.is_accident is True
        assert model.confidence == 0.92
        assert model.severity == "high"
        assert model.vehicles_involved == 2

        # Invalid severity raises ValidationError
        with pytest.raises(Exception):
            VerificationReportModel(**{**valid_data, "severity": "catastrophic"})

        # Out-of-bounds confidence raises ValidationError
        with pytest.raises(Exception):
            VerificationReportModel(**{**valid_data, "confidence": 1.5})

    def test_vlm_offline_graceful_fallback_when_api_key_missing(self, sample_frames: List[np.ndarray]):
        """Without API credentials, verify_incident_frames_vlm returns unverified fallback."""
        with patch.dict(os.environ, {"GOOGLE_API_KEY": "", "GEMINI_API_KEY": ""}, clear=True):
            res = verify_incident_frames_vlm(frames=sample_frames, detections="car_car_accident")
            assert isinstance(res, dict)
            assert res["verification_status"] == "unverified"
            assert "is_accident" in res
            assert "confidence" in res
            assert "reason" in res

    def test_vlm_timeout_and_retry_graceful_fallback(self, sample_frames: List[np.ndarray]):
        """Simulated API timeout triggers exponential backoff and returns unverified fallback."""
        with patch("acs.llm.genai") as mock_genai:
            mock_client = MagicMock()
            mock_genai.Client.return_value = mock_client
            # Simulate timeout or connection failure
            mock_client.models.generate_content.side_effect = TimeoutError("Connection timed out")

            res = verify_incident_frames_vlm(
                frames=sample_frames,
                api_key="fake-test-key",
                timeout=0.2,
                max_retries=1,
            )
            assert res["verification_status"] == "unverified"
            assert "VLM verification failed" in res["reason"]


# -----------------------------------------------------------------------------
# 3. Budget and Cache Management Tests
# -----------------------------------------------------------------------------
class TestBudgetAndCacheManager:
    """Tests daily budget enforcement and per-incident result caching."""

    def test_daily_budget_enforcement(self):
        """Budget manager restricts calls once daily budget is exceeded."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            budget_file = os.path.join(tmp_dir, "vlm_budget.json")
            mgr = VLMBudgetManager(budget_file=budget_file, daily_budget=3)

            assert mgr.can_call() is True
            assert mgr.record_call() == 1
            assert mgr.record_call() == 2
            assert mgr.record_call() == 3
            # Limit reached
            assert mgr.can_call() is False

            stats = mgr.get_stats()
            assert stats["calls_today"] == 3
            assert stats["remaining"] == 0

    def test_cache_roundtrip_and_deduplication(self):
        """Cache correctly returns stored decision for matching incident ID."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            cache_file = os.path.join(tmp_dir, "verification_cache.json")
            cache = IncidentVerificationCache(cache_file=cache_file)

            decision_dict = {
                "status": "auto_confirmed",
                "vlm_called": True,
                "physics_passed": True,
                "confidence": 0.95,
                "reason": "Visual impact confirmed",
                "alert_allowed": True,
                "severity": "high",
                "vehicles_involved": 2,
            }
            cache.set("INC-100-001", decision_dict)

            hit = cache.get("INC-100-001")
            assert hit is not None
            assert hit["status"] == "auto_confirmed"
            assert hit["confidence"] == 0.95

            # Miss
            assert cache.get("INC-NONEXISTENT") is None


# -----------------------------------------------------------------------------
# 4. Tri-State Decision Matrix & Verification Pipeline Tests
# -----------------------------------------------------------------------------
class TestVerificationDecisionPipeline:
    """Tests the full verify_incident pipeline decision matrix."""

    def test_auto_confirmed_when_yolo_and_vlm_agree(self, sample_incident: AccidentIncident):
        """Auto-confirm when YOLO detects accident and VLM confirms is_accident=True."""
        mock_vlm_output = {
            "is_accident": True,
            "confidence": 0.90,
            "reason": "Two cars collided with front bumper deformity",
            "severity": "medium",
            "vehicles_involved": 2,
            "verification_status": "verified",
        }

        with tempfile.TemporaryDirectory() as tmp_dir:
            bm = VLMBudgetManager(budget_file=os.path.join(tmp_dir, "b.json"), daily_budget=10)
            vc = IncidentVerificationCache(cache_file=os.path.join(tmp_dir, "c.json"))

            with patch("acs.verifier.verify_incident_frames_vlm", return_value=mock_vlm_output):
                decision = verify_incident(
                    incident=sample_incident,
                    budget_manager=bm,
                    cache=vc,
                    force_vlm=True,
                )

                assert decision.status == "auto_confirmed"
                assert decision.alert_allowed is True
                assert decision.vlm_called is True
                assert decision.confidence >= 0.88
                assert decision.vehicles_involved == 2

    def test_needs_review_when_yolo_and_vlm_disagree(self, sample_incident: AccidentIncident):
        """Disagreement between YOLO and VLM must set status to needs_review (alert held)."""
        mock_vlm_output = {
            "is_accident": False,
            "confidence": 0.85,
            "reason": "Vehicles are parked closely without structural damage or impact",
            "severity": "low",
            "vehicles_involved": 2,
            "verification_status": "verified",
        }

        with tempfile.TemporaryDirectory() as tmp_dir:
            bm = VLMBudgetManager(budget_file=os.path.join(tmp_dir, "b.json"), daily_budget=10)
            vc = IncidentVerificationCache(cache_file=os.path.join(tmp_dir, "c.json"))

            with patch("acs.verifier.verify_incident_frames_vlm", return_value=mock_vlm_output):
                decision = verify_incident(
                    incident=sample_incident,
                    budget_manager=bm,
                    cache=vc,
                    force_vlm=True,
                )

                assert decision.status == "needs_review"
                assert decision.alert_allowed is False  # Alert withheld
                assert "Needs Human Review" in decision.reason

    def test_cheap_physics_check_failure_skips_vlm(self, sample_incident: AccidentIncident):
        """When physics check clearly fails, VLM call must be skipped to save cost."""
        # Static ghost box history
        ghost_history = [
            (0.0, [DetectionBox(bbox=(10, 10, 30, 30), confidence=0.7, class_id=5, class_name="car_car_accident", is_accident=True)]),
            (0.1, [DetectionBox(bbox=(10, 10, 30, 30), confidence=0.7, class_id=5, class_name="car_car_accident", is_accident=True)]),
        ]

        with tempfile.TemporaryDirectory() as tmp_dir:
            bm = VLMBudgetManager(budget_file=os.path.join(tmp_dir, "b.json"), daily_budget=10)
            vc = IncidentVerificationCache(cache_file=os.path.join(tmp_dir, "c.json"))

            with patch("acs.verifier.verify_incident_frames_vlm") as mock_vlm:
                decision = verify_incident(
                    incident=sample_incident,
                    detection_history=ghost_history,
                    budget_manager=bm,
                    cache=vc,
                    force_vlm=False,
                )

                # VLM must NOT have been called
                mock_vlm.assert_not_called()
                assert decision.vlm_called is False
                assert decision.physics_passed is False
                assert decision.alert_allowed is False
                assert decision.status in ("needs_review", "suppressed")
                assert "Physics check failed" in decision.reason

    def test_daily_budget_exhaustion_falls_back_to_unverified(self, sample_incident: AccidentIncident):
        """When daily VLM budget is reached, falls back to unverified YOLO-only without crashing."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            bm = VLMBudgetManager(budget_file=os.path.join(tmp_dir, "b.json"), daily_budget=0)  # 0 budget
            vc = IncidentVerificationCache(cache_file=os.path.join(tmp_dir, "c.json"))

            with patch("acs.verifier.verify_incident_frames_vlm") as mock_vlm:
                decision = verify_incident(
                    incident=sample_incident,
                    budget_manager=bm,
                    cache=vc,
                    force_vlm=False,
                )
                mock_vlm.assert_not_called()
                assert decision.status == "unverified"
                assert decision.alert_allowed is True
                assert "Daily VLM call budget reached" in decision.reason

    def test_vlm_api_failure_falls_back_to_unverified_never_crashes(self, sample_incident: AccidentIncident):
        """VLM API failure returns unverified decision without crashing."""
        mock_unverified = {
            "is_accident": True,
            "confidence": 0.5,
            "reason": "VLM verification failed (API Timeout); falling back",
            "severity": "medium",
            "vehicles_involved": 2,
            "verification_status": "unverified",
        }

        with tempfile.TemporaryDirectory() as tmp_dir:
            bm = VLMBudgetManager(budget_file=os.path.join(tmp_dir, "b.json"), daily_budget=10)
            vc = IncidentVerificationCache(cache_file=os.path.join(tmp_dir, "c.json"))

            with patch("acs.verifier.verify_incident_frames_vlm", return_value=mock_unverified):
                decision = verify_incident(
                    incident=sample_incident,
                    budget_manager=bm,
                    cache=vc,
                    force_vlm=True,
                )

                assert decision.status == "unverified"
                assert decision.alert_allowed is True
                assert "unverified" in decision.reason


# -----------------------------------------------------------------------------
# 5. Alert Manager Verification Gate Integration Tests
# -----------------------------------------------------------------------------
class TestAlertsVerificationGate:
    """Verifies that send_incident_alert respects verification status."""

    def test_suppressed_incident_never_dispatches_alert(self, sample_incident: AccidentIncident):
        """Incidents marked 'suppressed' must be withheld from alerting."""
        sample_incident.verification_status = "suppressed"
        res = send_incident_alert(
            incident=sample_incident,
            report_text="Test accident report",
            dry_run=True,
        )
        assert res.get("suppressed") is True

    def test_needs_review_incident_held_from_alerting(self, sample_incident: AccidentIncident):
        """Incidents marked 'needs_review' must be held without dispatching alerts."""
        sample_incident.verification_status = "needs_review"
        res = send_incident_alert(
            incident=sample_incident,
            report_text="Test accident report",
            dry_run=True,
        )
        assert res.get("held_for_review") is True

    def test_auto_confirmed_incident_dispatches_alert(self, sample_incident: AccidentIncident):
        """Auto-confirmed incidents proceed to alert delivery."""
        sample_incident.verification_status = "auto_confirmed"
        res = send_incident_alert(
            incident=sample_incident,
            report_text="Auto-confirmed collision report",
            channels=["console"],
            dry_run=True,
        )
        assert res.get("console") is True

    def test_unverified_fallback_dispatches_with_annotation(self, sample_incident: AccidentIncident):
        """Unverified incidents dispatch alerts with an unverified warning flag."""
        sample_incident.verification_status = "unverified"
        res = send_incident_alert(
            incident=sample_incident,
            report_text="YOLO-only detection report",
            channels=["console"],
            dry_run=True,
        )
        assert res.get("console") is True


# -----------------------------------------------------------------------------
# 6. AccidentEventManager Multi-Frame Collection Integration Tests
# -----------------------------------------------------------------------------
class TestEventManagerVerificationFrames:
    """Verifies that AccidentEventManager collects 3 to 5 frames on incident closure."""

    def test_event_manager_collects_3_to_5_frames_on_incident_close(self):
        """AccidentEventManager populates verification_frames and verification_frame_paths."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            mgr = AccidentEventManager(
                consecutive_frames_threshold=3,
                timeout_seconds=0.2,
                cooldown_seconds=1.0,
                output_dir=tmp_dir,
                reports_dir=tmp_dir,
                auto_save=True,
                save_verification_frames=True,
            )

            frame = np.full((50, 50, 3), 120, dtype=np.uint8)
            det_box = DetectionBox(bbox=(10, 10, 40, 40), confidence=0.85, class_id=5, class_name="car_car_accident", is_accident=True)
            res = DetectionResult(annotated_frame=frame, detections=[det_box], accident_detected=True, detected_classes={"car_car_accident"}, accident_classes={"car_car_accident"})

            # Feed 3 accident frames
            mgr.update(res, frame=frame, timestamp=0.0)
            mgr.update(res, frame=frame, timestamp=0.1)
            mgr.update(res, frame=frame, timestamp=0.2)

            assert mgr.is_incident_active is True

            # Trigger timeout closure
            quiet_res = DetectionResult(annotated_frame=frame, detections=[], accident_detected=False)
            completed = mgr.update(quiet_res, frame=frame, timestamp=0.6)

            assert completed is not None
            assert len(completed.verification_frames) >= 3
            assert len(completed.verification_frames) <= 5
            assert len(completed.verification_frame_paths) >= 3
            for path in completed.verification_frame_paths:
                assert os.path.exists(path)
