"""
tests.test_severity
Unit and integration tests for the collision severity scoring engine:
- Synthetic track motion simulations (high-speed head-on, motorcycle vs pedestrian, low-speed bump).
- Closing speed calculation and camera calibration (ppm) normalization.
- Post-impact state analysis (stationary vehicles, people on ground, blocked travel lanes).
- Multimodal VLM severity integration.
- Configurable threshold mapping (low, medium, high).
- Dynamic emergency unit dispatch routing (EMS, police, fire, tow).
- Pluggable model architecture (replacing heuristic scorer with custom ML model).
- SQLite persistence and API schema serialization.
"""
import unittest
from typing import List, Tuple

import numpy as np

from acs.detector import AccidentIncident, DetectionBox
from acs.severity import (
    BaseSeverityScorer,
    CameraCalibration,
    MLSeverityScorer,
    RuleBasedSeverityScorer,
    SeverityFeatures,
    SeverityResult,
    evaluate_incident_severity,
    extract_severity_features,
    get_severity_scorer,
    register_severity_scorer,
)
from backend.app.db.database import init_db
from backend.app.routers.incidents import _to_incident_response
from backend.app.services.incident_service import save_incident_to_db


# Helper to construct synthetic detection frame records
def create_synthetic_frame(
    frame_idx: int,
    boxes: List[Tuple[Tuple[float, float, float, float], str, int]],
) -> List[DetectionBox]:
    """Creates a list of DetectionBox objects for a given synthetic frame.

    Args:
        frame_idx: Frame sequence index.
        boxes: List of tuples (bbox, class_name, track_id).
    """
    res = []
    for bbox, cls_name, tid in boxes:
        res.append(
            DetectionBox(
                bbox=bbox,
                confidence=0.88,
                class_id=1,
                class_name=cls_name,
                is_accident="accident" in cls_name or cls_name == "accident",
                track_id=tid,
            )
        )
    return res


class TestSeverityFeatureExtraction(unittest.TestCase):
    """Tests feature extraction across synthetic track telemetry scenarios."""

    def test_synthetic_high_speed_head_on_tracks(self):
        """Simulates 2 vehicles closing in rapidly before impact and stopping afterward."""
        # Frame 0 to 14: Car 1 moves right (100 -> 380), Car 2 moves left (700 -> 420)
        # Impact at Frame 15 (around x=400)
        # Frames 16 to 30: both vehicles remain stationary around impact zone
        detection_history = []
        fps = 30.0

        for f in range(30):
            if f <= 14:
                # Pre-impact moving tracks: ~20 px/frame each -> ~40 px/frame closing rate = ~1200 px/sec
                c1_x1 = 100.0 + f * 20.0
                c1_box = (c1_x1, 200.0, c1_x1 + 60.0, 240.0)

                c2_x2 = 700.0 - f * 20.0
                c2_box = (c2_x2 - 60.0, 200.0, c2_x2, 240.0)
                detection_history.append(
                    create_synthetic_frame(
                        f,
                        [
                            (c1_box, "car", 101),
                            (c2_box, "car", 102),
                        ],
                    )
                )
            else:
                # Post-impact: both stopped near center (x=400)
                c1_box = (380.0, 200.0, 440.0, 245.0)
                c2_box = (400.0, 195.0, 460.0, 240.0)
                detection_history.append(
                    create_synthetic_frame(
                        f,
                        [
                            (c1_box, "car_car_accident", 101),
                            (c2_box, "car_car_accident", 102),
                        ],
                    )
                )

        incident = AccidentIncident(
            incident_id="INC-SYN-001",
            start_timestamp="2026-10-18T12:00:00",
            end_timestamp="2026-10-18T12:00:01",
            class_name="car_car_accident",
            max_confidence=0.92,
            image_path="test_keyframe.jpg",
            start_time=100.0,
            end_time=101.0,
            vlm_severity="high",
            vlm_confidence=0.95,
        )

        features = extract_severity_features(incident, detection_history=detection_history)

        # Assertions on collision dynamics
        self.assertEqual(features.vehicles_count, 2)
        self.assertGreater(features.closing_speed_px_per_sec, 600.0)
        self.assertGreaterEqual(features.stationary_vehicles_post_impact, 1)
        self.assertTrue(features.blocked_lanes)
        self.assertFalse(features.people_on_ground)
        self.assertEqual(features.vlm_severity, "high")

    def test_synthetic_motorcycle_vs_pedestrian_with_person_down(self):
        """Simulates motorcycle striking a pedestrian, leaving person prone on the roadway."""
        detection_history = []
        for f in range(20):
            if f <= 10:
                # Upright pedestrian (height 50 > width 20)
                ped_box = (300.0, 200.0, 320.0, 250.0)
                moto_box = (100.0 + f * 18.0, 210.0, 140.0 + f * 18.0, 245.0)
                detection_history.append(
                    create_synthetic_frame(
                        f,
                        [
                            (ped_box, "person", 201),
                            (moto_box, "motorcycle", 202),
                        ],
                    )
                )
            else:
                # Post-impact: fallen pedestrian prone (width 60 > height 20)
                ped_box = (300.0, 230.0, 360.0, 250.0)
                moto_box = (310.0, 220.0, 350.0, 255.0)
                detection_history.append(
                    create_synthetic_frame(
                        f,
                        [
                            (ped_box, "person", 201),
                            (moto_box, "bike_person_accident", 202),
                        ],
                    )
                )

        incident = AccidentIncident(
            incident_id="INC-SYN-002",
            start_timestamp="2026-10-18T12:05:00",
            end_timestamp="2026-10-18T12:05:01",
            class_name="bike_person_accident",
            max_confidence=0.89,
            image_path="test_keyframe.jpg",
            start_time=200.0,
            end_time=201.0,
            vlm_severity="medium",
            vlm_confidence=0.85,
        )

        features = extract_severity_features(incident, detection_history=detection_history)

        self.assertEqual(features.pedestrians_count, 1)
        self.assertEqual(features.motorcycles_count, 1)
        self.assertTrue(features.people_on_ground)

    def test_camera_calibration_ppm_normalization(self):
        """Tests converting pixel velocities into calibrated physical meters/second."""
        # 1 vehicle moving at 300 px/sec
        detection_history = []
        for f in range(10):
            box = (100.0 + f * 10.0, 100.0, 150.0 + f * 10.0, 140.0)  # 10 px/frame * 30 fps = 300 px/s
            detection_history.append(create_synthetic_frame(f, [(box, "car", 1)]))

        incident = AccidentIncident(
            incident_id="INC-CAL-001",
            start_timestamp="2026-10-18T12:10:00",
            end_timestamp="2026-10-18T12:10:01",
            class_name="car_object_accident",
            max_confidence=0.80,
            image_path="test.jpg",
            start_time=0.0,
            end_time=1.0,
        )

        # Calibration: 20 pixels per meter
        calibration = CameraCalibration(pixels_per_meter=20.0, fps=30.0)
        features = extract_severity_features(incident, detection_history=detection_history, calibration=calibration)

        self.assertAlmostEqual(features.closing_speed_px_per_sec, 300.0, delta=15.0)
        self.assertIsNotNone(features.closing_speed_mps)
        self.assertAlmostEqual(features.closing_speed_mps, 15.0, delta=1.0)  # 300 / 20 = 15 m/s (~54 km/h)


class TestRuleBasedSeverityScorer(unittest.TestCase):
    """Tests multi-factor score calculation, threshold mapping, and unit routing."""

    def setUp(self):
        self.scorer = RuleBasedSeverityScorer()

    def test_high_severity_collision_routing(self):
        """Critical collision with high speed, multiple disabled vehicles, and VLM high."""
        features = SeverityFeatures(
            total_entities=3,
            vehicles_count=2,
            heavy_vehicles_count=1,
            closing_speed_px_per_sec=750.0,
            closing_speed_mps=22.0,
            stationary_vehicles_post_impact=2,
            blocked_lanes=True,
            vlm_severity="high",
            vlm_confidence=0.92,
        )

        result = self.scorer.score(features, low_max=35, medium_max=70)

        # Must qualify for high severity (>70)
        self.assertGreater(result.score, 70)
        self.assertEqual(result.level, "high")

        # Must route all major emergency units including Fire and EMS
        self.assertIn("EMS", result.recommended_units)
        self.assertIn("police", result.recommended_units)
        self.assertIn("fire", result.recommended_units)
        self.assertIn("tow", result.recommended_units)

    def test_pedestrian_and_motorcycle_escalates_ems_routing(self):
        """Even with lower closing speeds, pedestrian/motorcycle involvement mandates EMS dispatch."""
        features = SeverityFeatures(
            total_entities=2,
            vehicles_count=0,
            pedestrians_count=1,
            motorcycles_count=1,
            closing_speed_px_per_sec=120.0,
            people_on_ground=True,
            vlm_severity="low",
            vlm_confidence=0.70,
        )

        result = self.scorer.score(features, low_max=35, medium_max=70)

        # EMS must be dispatched regardless of baseline tier
        self.assertIn("EMS", result.recommended_units)
        self.assertIn("police", result.recommended_units)

    def test_low_speed_parking_bumper_tap(self):
        """Minor collision with low speed, no injuries, and low VLM severity."""
        features = SeverityFeatures(
            total_entities=1,
            vehicles_count=1,
            pedestrians_count=0,
            motorcycles_count=0,
            closing_speed_px_per_sec=80.0,
            closing_speed_mps=2.5,
            stationary_vehicles_post_impact=0,
            people_on_ground=False,
            blocked_lanes=False,
            vlm_severity="low",
            vlm_confidence=0.85,
        )

        result = self.scorer.score(features, low_max=35, medium_max=70)

        # Must be in low severity tier (<= 35)
        self.assertLessEqual(result.score, 35)
        self.assertEqual(result.level, "low")

        # Only Tow and Police required; no Fire, no EMS
        self.assertIn("police", result.recommended_units)
        self.assertIn("tow", result.recommended_units)
        self.assertNotIn("fire", result.recommended_units)
        self.assertNotIn("EMS", result.recommended_units)

    def test_transparent_breakdown_structure(self):
        """Ensures all 4 transparent breakdown sub-scores are populated with metrics and explanations."""
        features = SeverityFeatures(
            total_entities=2,
            vehicles_count=2,
            closing_speed_px_per_sec=420.0,
            stationary_vehicles_post_impact=1,
            vlm_severity="medium",
            vlm_confidence=0.80,
        )

        result = self.scorer.score(features)
        components = {c.key: c for c in result.components}

        # Validate 4 core components exist
        self.assertIn("vulnerable_users", components)
        self.assertIn("closing_speed", components)
        self.assertIn("post_impact", components)
        self.assertIn("vlm_opinion", components)

        for comp in result.components:
            self.assertGreaterEqual(comp.score, 0.0)
            self.assertLessEqual(comp.score, comp.max_score)
            self.assertTrue(len(comp.explanation) > 0)

        # Sum of sub-scores matches final clamped score within rounding
        sub_sum = sum(c.score for c in result.components)
        self.assertEqual(result.score, int(round(sub_sum)))

    def test_configurable_thresholds_override(self):
        """Tests that changing low_max and medium_max alters the mapped severity tier."""
        features = SeverityFeatures(
            total_entities=2,
            vehicles_count=2,
            closing_speed_px_per_sec=400.0,
            stationary_vehicles_post_impact=1,
            vlm_severity="medium",
        )
        # Score is ~41
        res_default = self.scorer.score(features, low_max=35, medium_max=70)
        self.assertEqual(res_default.level, "medium")

        # If low_max is raised to 50, score <= 50 becomes "low"
        res_custom = self.scorer.score(features, low_max=50, medium_max=85)
        self.assertEqual(res_custom.level, "low")


class TestPluggableModelScorer(unittest.TestCase):
    """Tests the pluggable architecture: substituting the heuristic with a trained ML model."""

    def test_ml_scorer_with_mock_model(self):
        """Verifies an ML model predicting numerical scores integrates cleanly."""
        class MockTrainedEstimator:
            def predict(self, X):
                # Returns 88 for severe, regardless of features
                return np.array([88.0])

        ml_scorer = MLSeverityScorer(model=MockTrainedEstimator(), version="mock_xgb_v1")
        features = SeverityFeatures(
            total_entities=2,
            vehicles_count=2,
            closing_speed_px_per_sec=500.0,
            vlm_severity="high",
        )

        result = ml_scorer.score(features, low_max=35, medium_max=70)

        self.assertEqual(result.score, 88)
        self.assertEqual(result.level, "high")
        self.assertEqual(result.scorer_version, "mock_xgb_v1")
        self.assertIn("mock_xgb_v1", result.factors[0])

        # Transparent sub-scores are still preserved from telemetry for UI explainability
        self.assertEqual(len(result.components), 4)

    def test_ml_scorer_with_classifier_probabilities(self):
        """Verifies an ML classifier with predict_proba outputs weighted calibrated scores."""
        class MockClassifier:
            def predict_proba(self, X):
                # Probabilities for [low, medium, high]: [0.0, 0.2, 0.8] -> 0*20 + 0.2*55 + 0.8*85 = 79.0
                return np.array([[0.0, 0.2, 0.8]])

        ml_scorer = MLSeverityScorer(model=MockClassifier(), version="mock_clf_v1")
        features = SeverityFeatures(
            total_entities=3,
            vehicles_count=2,
            closing_speed_px_per_sec=650.0,
        )

        result = ml_scorer.score(features)
        self.assertEqual(result.score, 79)
        self.assertEqual(result.level, "high")

    def test_global_scorer_registration_and_retrieval(self):
        """Tests registering a custom model into the registry and retrieving it."""
        class CustomScorer(BaseSeverityScorer):
            def score(self, features, low_max=35, medium_max=70):
                return SeverityResult(
                    score=99,
                    level="high",
                    recommended_units=["EMS", "police", "fire", "tow"],
                    components=[],
                    factors=["Custom trained neural network prediction."],
                    scorer_version="custom_nn_v1",
                )

        custom = CustomScorer()
        register_severity_scorer("custom_nn", custom)

        retrieved = get_severity_scorer("custom_nn")
        self.assertIs(retrieved, custom)

        incident = AccidentIncident(
            incident_id="INC-PLUG-001",
            start_timestamp="2026-10-18T12:00:00",
            end_timestamp="2026-10-18T12:00:01",
            class_name="car_car_accident",
            max_confidence=0.9,
            image_path="test.jpg",
            start_time=0.0,
            end_time=1.0,
        )
        res = evaluate_incident_severity(incident, scorer_type="custom_nn")
        self.assertEqual(res.score, 99)
        self.assertEqual(res.scorer_version, "custom_nn_v1")


class TestEndToEndSeverityPersistence(unittest.TestCase):
    """Tests end-to-end SQLite persistence and API schema formatting."""

    def setUp(self):
        init_db()

    def test_save_and_retrieve_incident_with_severity_breakdown(self):
        """Verifies saving severity score, breakdown JSON, and recommended units into SQLite."""
        inc_id = "INC-SEV-TEST-99"
        breakdown_dict = {
            "score": 78,
            "level": "high",
            "recommended_units": ["EMS", "police", "fire", "tow"],
            "components": [
                {
                    "key": "vulnerable_users",
                    "label": "Vulnerable Users",
                    "score": 25.0,
                    "max_score": 35.0,
                    "weight_pct": 35.0,
                    "explanation": "Motorcycle involved (+15), 2 vehicles total (+10)",
                }
            ],
            "factors": ["High pre-impact closing velocity."],
            "scorer_version": "rule_based_v1",
        }
        units_list = ["EMS", "police", "fire", "tow"]

        saved_inc = save_incident_to_db(
            incident_id=inc_id,
            start_timestamp="2026-10-18T14:00:00",
            end_timestamp="2026-10-18T14:00:02",
            class_name="car_bike_accident",
            max_confidence=0.91,
            image_path="test_kf.jpg",
            duration=2.0,
            severity="high",
            severity_score=78,
            severity_breakdown=breakdown_dict,
            recommended_units=units_list,
        )

        self.assertEqual(saved_inc.severity, "high")
        self.assertEqual(saved_inc.severity_score, 78)
        self.assertIsNotNone(saved_inc.severity_breakdown)
        self.assertIn("vulnerable_users", saved_inc.severity_breakdown)

        # Verify API response formatting
        api_response = _to_incident_response(saved_inc)
        self.assertEqual(api_response.severity, "high")
        self.assertEqual(api_response.severity_score, 78)
        self.assertIsInstance(api_response.severity_breakdown, dict)
        self.assertEqual(api_response.severity_breakdown["score"], 78)
        self.assertEqual(api_response.recommended_units, ["EMS", "police", "fire", "tow"])


if __name__ == "__main__":
    unittest.main()
