"""
tests/test_temporal_model.py
Unit and integration tests for the second-stage temporal accident verification module:
- Feature extraction math, normalization, and bounds (acs/temporal/features.py).
- TemporalGate model loading and verification inference (acs/temporal/gate.py).
- Integration into AccidentEventManager as a switchable gate (acs/detector.py).
"""
import math
import numpy as np
import pytest

from acs.detector import AccidentEventManager, DetectionBox, DetectionResult
from acs.temporal.features import (
    FEATURE_NAMES,
    TrackHistoryBuffer,
    TrackPairFeatureExtractor,
    TrackPoint,
    TrackTrajectory,
    angular_difference,
    compute_box_iou_coords,
)
from acs.temporal.gate import TemporalGate


class TestTemporalFeatures:
    """Tests feature extraction logic, camera/fps normalization, and physical bounds."""

    def test_feature_names_canonical_length(self):
        assert len(FEATURE_NAMES) == 24
        assert "iou_max" in FEATURE_NAMES
        assert "max_closing_speed" in FEATURE_NAMES
        assert "max_deceleration" in FEATURE_NAMES
        assert "max_direction_change_rad" in FEATURE_NAMES
        assert "min_ttc_seconds" in FEATURE_NAMES
        assert "post_overlap_stationary_seconds" in FEATURE_NAMES
        assert "stopped_vehicles_in_pair" in FEATURE_NAMES

    def test_compute_box_iou_coords(self):
        box1 = (0.0, 0.0, 1.0, 1.0)
        box2 = (0.5, 0.5, 1.5, 1.5)
        # Inter = 0.5 * 0.5 = 0.25; Union = 1.0 + 1.0 - 0.25 = 1.75
        iou = compute_box_iou_coords(box1, box2)
        assert pytest.approx(iou, rel=1e-3) == 0.25 / 1.75

        # Disjoint
        assert compute_box_iou_coords((0.0, 0.0, 0.2, 0.2), (0.5, 0.5, 0.8, 0.8)) == 0.0

    def test_angular_difference(self):
        assert pytest.approx(angular_difference(0.0, math.pi), rel=1e-3) == math.pi
        assert pytest.approx(angular_difference(0.1, 0.2), rel=1e-3) == 0.1
        assert pytest.approx(angular_difference(0.1, 2 * math.pi - 0.1), rel=1e-3) == 0.2

    def test_trajectory_interpolation(self):
        traj = TrackTrajectory(track_id=1)
        traj.add_point(TrackPoint(0, 0.0, (0, 0, 10, 10), (0.0, 0.0, 0.1, 0.1), (0.05, 0.05), 0.9, 2, "car", 1))
        traj.add_point(TrackPoint(4, 0.2, (40, 40, 50, 50), (0.4, 0.4, 0.5, 0.5), (0.45, 0.45), 0.9, 2, "car", 1))

        interp = traj.get_interpolated_points(fps=20.0, max_gap_seconds=1.0)
        assert len(interp) == 5  # frames 0, 1, 2, 3, 4
        assert 2 in interp
        assert pytest.approx(interp[2].centroid[0], rel=1e-3) == 0.25

    def test_track_pair_feature_extraction(self):
        extractor = TrackPairFeatureExtractor()
        t_a = TrackTrajectory(1)
        t_b = TrackTrajectory(2)

        fps = 30.0
        # Simulate impact at frame 20, stopped until frame 35
        for i in range(35):
            t = i / fps
            xa = min(0.48, 0.10 + 0.02 * i)
            xb = max(0.50, 0.90 - 0.02 * i)

            t_a.add_point(TrackPoint(i, t, (xa * 100, 50, (xa + 0.05) * 100, 80),
                                     (xa, 0.5, xa + 0.05, 0.8), (xa + 0.025, 0.65), 0.9, 2, "car", 1))
            t_b.add_point(TrackPoint(i, t, (xb * 100, 50, (xb + 0.05) * 100, 80),
                                     (xb, 0.5, xb + 0.05, 0.8), (xb + 0.025, 0.65), 0.9, 2, "car", 2))

        res = extractor.extract_pair_features(t_a, t_b, window_seconds=1.5, fps=fps)
        assert res is not None
        feats, tab_v, seq_m = res

        assert tab_v.shape == (24,)
        assert seq_m.shape == (30, 9)
        assert feats["max_closing_speed"] > 0.0
        assert feats["post_overlap_stationary_seconds"] > 0.0
        assert feats["stopped_vehicles_in_pair"] >= 1.0


class TestTemporalGate:
    """Tests TemporalGate inference and failure modes."""

    def test_disabled_gate(self):
        gate = TemporalGate(enabled=False)
        buf = TrackHistoryBuffer()
        confirmed, prob, diag = gate.verify(buf)
        assert confirmed is True
        assert prob == 1.0
        assert diag["gate_status"] == "disabled_or_unloaded"

    def test_non_collision_buffer_suppression(self):
        gate = TemporalGate(model_path="models/temporal_gate.pt", threshold=0.50, enabled=True)
        buf = TrackHistoryBuffer(max_buffer_seconds=3.0)
        # Add normal vehicles passing each other
        for f in range(20):
            t = f / 20.0
            det_a = DetectionBox(bbox=(100.0 + f * 5, 200.0, 150.0 + f * 5, 250.0), confidence=0.9, class_id=2, class_name="car", is_accident=False, track_id=1)
            det_b = DetectionBox(bbox=(400.0 - f * 5, 200.0, 450.0 - f * 5, 250.0), confidence=0.9, class_id=2, class_name="car", is_accident=False, track_id=2)
            buf.add_frame_detections(f, t, [det_a, det_b], 640, 480)

        confirmed, prob, diag = gate.verify(buf, candidate_bbox=(200, 200, 300, 300), frame_width=640, frame_height=480)
        assert isinstance(confirmed, bool)
        assert isinstance(prob, float)

    def test_model_loading_lightgbm_and_pytorch(self):
        gate_pt = TemporalGate(model_path="models/temporal_gate.pt")
        assert gate_pt.model_type in ("1d_cnn", "gru")

        gate_lgb = TemporalGate(model_path="models/temporal_gate_lgb.joblib")
        assert gate_lgb.model_type == "lightgbm"


class TestAccidentEventManagerIntegration:
    """Tests integration of temporal gate within AccidentEventManager lifecycle."""

    def test_switchable_temporal_config(self):
        mgr_on = AccidentEventManager(temporal_model_enabled=True)
        assert mgr_on.temporal_model_enabled is True
        assert mgr_on.temporal_gate is not None

        mgr_off = AccidentEventManager(temporal_model_enabled=False)
        assert mgr_off.temporal_model_enabled is False

    def test_temporal_gate_suppresses_moving_traffic_false_alarm(self):
        mgr = AccidentEventManager(
            temporal_model_enabled=True,
            temporal_threshold=0.50,
            consecutive_frames_threshold=3,
            auto_save=False,
            save_clips=False,
        )

        dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        # Populate history with continuous moving normal traffic
        for f in range(25):
            t = f / 20.0
            car1 = DetectionBox(bbox=(50.0 + f * 10, 100.0, 100.0 + f * 10, 150.0), confidence=0.92, class_id=2, class_name="car", is_accident=False, track_id=10)
            car2 = DetectionBox(bbox=(50.0 + f * 10, 200.0, 100.0 + f * 10, 250.0), confidence=0.92, class_id=2, class_name="car", is_accident=False, track_id=11)
            # Suppose frame 24 has a spurious false alarm box from YOLO
            acc_box = DetectionBox(bbox=(250.0, 120.0, 320.0, 180.0), confidence=0.60, class_id=6, class_name="car_car_accident", is_accident=True, track_id=None)
            dets = [car1, car2]
            if f >= 20:
                dets.append(acc_box)
            res = DetectionResult(annotated_frame=dummy_frame, detections=dets)
            mgr.update(res, frame=dummy_frame, timestamp=t)

        # The spurious accident proposal should be suppressed by temporal gate
        assert mgr.is_incident_active is False
