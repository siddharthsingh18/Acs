"""
Unit tests for AccidentEventManager and ByteTrack tracking in acs/detector.py.
Covers:
- 60-frame crash sequence yielding exactly one incident.
- Sub-threshold (< 5 frames) detection rejection.
- Non-accident class filtering ('car', 'person').
- Confidence threshold filtering.
- Cooldown period enforcement (10s delay).
- Keyframe highest-confidence retention.
- Stream end active incident flushing.
- CSV and keyframe persistence validation.
"""
import csv
import os
import shutil
import tempfile
import unittest
import numpy as np

from acs.detector import (
    AccidentDetector,
    AccidentEventManager,
    DetectionBox,
    DetectionResult,
)


class TestAccidentEventManager(unittest.TestCase):
    """Test suite for the AccidentEventManager state machine."""

    def setUp(self):
        """Create temporary storage directories for each test."""
        self.test_dir = tempfile.mkdtemp()
        self.images_dir = os.path.join(self.test_dir, "report_images")
        self.reports_dir = os.path.join(self.test_dir, "reports")
        self.csv_file = "incidents.csv"

    def tearDown(self):
        """Clean up temporary test files."""
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

    def _create_detection_result(
        self,
        is_accident: bool = False,
        class_name: str = "car",
        confidence: float = 0.85,
    ) -> DetectionResult:
        """Helper to create a synthetic DetectionResult."""
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        box = DetectionBox(
            bbox=(10.0, 10.0, 50.0, 50.0),
            confidence=confidence,
            class_id=0,
            class_name=class_name,
            is_accident=is_accident,
            track_id=1,
        )
        return DetectionResult(
            annotated_frame=frame,
            detections=[box] if confidence > 0 else [],
            accident_detected=is_accident,
            detected_classes={class_name} if confidence > 0 else set(),
            accident_classes={class_name} if is_accident else set(),
        )

    def test_60_frame_crash_yields_exactly_one_incident(self):
        """A 60-frame crash sequence at 30 FPS followed by quiet frames must yield exactly 1 incident."""
        manager = AccidentEventManager(
            consecutive_frames_threshold=5,
            timeout_seconds=3.0,
            cooldown_seconds=10.0,
            confidence_threshold=0.5,
            output_dir=self.images_dir,
            reports_dir=self.reports_dir,
            csv_filename=self.csv_file,
            auto_save=True,
        )

        fps = 30.0
        peak_frame_idx = 35
        peak_confidence = 0.97
        completed_incidents = []

        # 1. Feed 60 consecutive frames with car_car_accident detections
        for i in range(60):
            t = i / fps
            conf = 0.80 + (0.17 * (1.0 - abs(i - peak_frame_idx) / 60.0))
            if i == peak_frame_idx:
                conf = peak_confidence

            # Create distinct synthetic frame for peak detection
            frame = np.full((100, 100, 3), fill_value=i, dtype=np.uint8)
            det = self._create_detection_result(
                is_accident=True,
                class_name="car_car_accident",
                confidence=conf,
            )
            inc = manager.update(det, frame=frame, timestamp=t)
            if inc:
                completed_incidents.append(inc)

        # At frame 59, the incident should still be active because accident detections are ongoing
        self.assertTrue(manager.is_incident_active)
        self.assertEqual(len(completed_incidents), 0)

        # 2. Feed 100 non-accident frames (3.33 seconds > 3.0s timeout)
        for i in range(60, 160):
            t = i / fps
            frame = np.zeros((100, 100, 3), dtype=np.uint8)
            empty_det = DetectionResult(
                annotated_frame=frame,
                detections=[],
                accident_detected=False,
            )
            inc = manager.update(empty_det, frame=frame, timestamp=t)
            if inc:
                completed_incidents.append(inc)

        # 3. Assertions
        # Must yield EXACTLY ONE incident
        self.assertEqual(len(completed_incidents), 1)
        incident = completed_incidents[0]

        # Max confidence must match peak detection
        self.assertAlmostEqual(incident.max_confidence, peak_confidence, places=2)
        self.assertEqual(incident.class_name, "car_car_accident")

        # Peak keyframe must match peak frame data (fill_value == peak_frame_idx)
        self.assertIsNotNone(incident.keyframe)
        self.assertEqual(incident.keyframe[0, 0, 0], peak_frame_idx)

        # Disk persistence: Exactly one keyframe image must exist
        self.assertTrue(os.path.exists(incident.image_path))
        saved_images = os.listdir(self.images_dir)
        self.assertEqual(len(saved_images), 1)

        # Master CSV log: Exactly one row plus header must exist
        csv_path = os.path.join(self.reports_dir, self.csv_file)
        self.assertTrue(os.path.exists(csv_path))

        with open(csv_path, "r", encoding="utf-8") as f:
            reader = list(csv.reader(f))
            self.assertEqual(len(reader), 2)  # Header + 1 data row
            header, row = reader[0], reader[1]
            self.assertEqual(
                header,
                ["incident_id", "start_timestamp", "end_timestamp", "class", "max_confidence", "image_path"],
            )
            self.assertEqual(row[0], incident.incident_id)
            self.assertEqual(row[3], "car_car_accident")
            self.assertAlmostEqual(float(row[4]), peak_confidence, places=2)
            self.assertEqual(row[5], incident.image_path)

    def test_sub_threshold_consecutive_frames_rejected(self):
        """A sequence of 4 accident frames (< 5) must not start an incident."""
        manager = AccidentEventManager(
            consecutive_frames_threshold=5,
            timeout_seconds=3.0,
            cooldown_seconds=10.0,
            output_dir=self.images_dir,
            reports_dir=self.reports_dir,
        )

        completed = []
        # 4 accident frames
        for i in range(4):
            t = i * 0.1
            det = self._create_detection_result(is_accident=True, class_name="car_car_accident")
            inc = manager.update(det, timestamp=t)
            if inc:
                completed.append(inc)

        self.assertFalse(manager.is_incident_active)

        # Broken streak: 10 frames of no accident
        for i in range(4, 15):
            t = i * 0.1
            det = self._create_detection_result(is_accident=False, class_name="car")
            inc = manager.update(det, timestamp=t)
            if inc:
                completed.append(inc)

        self.assertFalse(manager.is_incident_active)
        self.assertEqual(len(completed), 0)
        self.assertEqual(len(manager.incidents), 0)

    def test_non_accident_classes_do_not_trigger_incident(self):
        """Traffic detections of non-accident classes ('car', 'person') must never trigger an incident."""
        manager = AccidentEventManager(
            consecutive_frames_threshold=5,
            output_dir=self.images_dir,
            reports_dir=self.reports_dir,
        )

        completed = []
        # 30 frames of normal car and pedestrian detections
        for i in range(30):
            t = i * 0.1
            cls = "car" if i % 2 == 0 else "person"
            det = self._create_detection_result(is_accident=False, class_name=cls, confidence=0.99)
            inc = manager.update(det, timestamp=t)
            if inc:
                completed.append(inc)

        self.assertFalse(manager.is_incident_active)
        self.assertEqual(len(completed), 0)

    def test_confidence_threshold_filtering(self):
        """Accident detections below confidence threshold must be ignored."""
        manager = AccidentEventManager(
            consecutive_frames_threshold=5,
            confidence_threshold=0.75,
            output_dir=self.images_dir,
            reports_dir=self.reports_dir,
        )

        # 10 frames of low confidence accident detection (0.60 < 0.75)
        for i in range(10):
            t = i * 0.1
            det = self._create_detection_result(
                is_accident=True,
                class_name="car_car_accident",
                confidence=0.60,
            )
            manager.update(det, timestamp=t)

        self.assertFalse(manager.is_incident_active)

    def test_cooldown_enforcement(self):
        """Accident detections occurring within cooldown window must not start a new incident."""
        manager = AccidentEventManager(
            consecutive_frames_threshold=5,
            timeout_seconds=3.0,
            cooldown_seconds=10.0,
            output_dir=self.images_dir,
            reports_dir=self.reports_dir,
            csv_filename=self.csv_file,
            auto_save=True,
        )

        # Incident 1: 10 frames of accident from t=0.0 to t=0.9
        for i in range(10):
            t = i * 0.1
            det = self._create_detection_result(is_accident=True, class_name="car_car_accident")
            manager.update(det, timestamp=t)

        # 4 seconds of quiet from t=1.0 to t=5.0 (exceeds timeout 3.0s)
        inc1 = None
        for i in range(10, 50):
            t = i * 0.1
            det = self._create_detection_result(is_accident=False)
            res = manager.update(det, timestamp=t)
            if res:
                inc1 = res

        self.assertIsNotNone(inc1)
        self.assertEqual(len(manager.incidents), 1)
        end_time_1 = inc1.end_time

        # Attempt incident 2 DURING COOLDOWN (e.g. t = end_time_1 + 4.0s < 10.0s cooldown)
        cooldown_start = end_time_1 + 1.0
        for i in range(10):
            t = cooldown_start + (i * 0.1)
            det = self._create_detection_result(is_accident=True, class_name="car_car_accident")
            res = manager.update(det, timestamp=t)
            self.assertIsNone(res)

        # Must NOT have started during cooldown
        self.assertFalse(manager.is_incident_active)
        self.assertEqual(len(manager.incidents), 1)

        # Now advance time BEYOND COOLDOWN (t = end_time_1 + 11.0s > 10.0s cooldown)
        after_cooldown = end_time_1 + 11.0
        for i in range(10):
            t = after_cooldown + (i * 0.1)
            det = self._create_detection_result(is_accident=True, class_name="bike_bike_accident")
            manager.update(det, timestamp=t)

        self.assertTrue(manager.is_incident_active)

        # End incident 2 with quiet period
        inc2 = None
        for i in range(10, 50):
            t = after_cooldown + (i * 0.1)
            det = self._create_detection_result(is_accident=False)
            res = manager.update(det, timestamp=t)
            if res:
                inc2 = res

        self.assertIsNotNone(inc2)
        self.assertEqual(len(manager.incidents), 2)
        self.assertEqual(inc2.class_name, "bike_bike_accident")

        # Master CSV must contain 2 data rows
        csv_path = os.path.join(self.reports_dir, self.csv_file)
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = list(csv.reader(f))
            self.assertEqual(len(reader), 3)  # Header + 2 incident rows

    def test_keyframe_highest_confidence_selected(self):
        """During an incident, the keyframe must be the frame with maximum confidence."""
        manager = AccidentEventManager(
            consecutive_frames_threshold=5,
            timeout_seconds=3.0,
            cooldown_seconds=10.0,
            confidence_threshold=0.5,
            output_dir=self.images_dir,
            reports_dir=self.reports_dir,
        )

        peak_idx = 12
        peak_conf = 0.99
        for i in range(20):
            t = i * 0.1
            conf = peak_conf if i == peak_idx else 0.70
            frame = np.full((100, 100, 3), fill_value=i, dtype=np.uint8)
            det = self._create_detection_result(
                is_accident=True,
                class_name="car_car_accident",
                confidence=conf,
            )
            manager.update(det, frame=frame, timestamp=t)

        inc = manager.end_active_incident(timestamp=2.0)
        self.assertIsNotNone(inc)
        self.assertEqual(inc.max_confidence, peak_conf)
        self.assertEqual(inc.keyframe[0, 0, 0], peak_idx)

    def test_end_active_incident_on_stream_end(self):
        """End of video stream flushes an active incident even without trailing quiet frames."""
        manager = AccidentEventManager(
            consecutive_frames_threshold=5,
            output_dir=self.images_dir,
            reports_dir=self.reports_dir,
        )

        for i in range(10):
            t = i * 0.1
            det = self._create_detection_result(is_accident=True, class_name="car_car_accident")
            manager.update(det, timestamp=t)

        self.assertTrue(manager.is_incident_active)

        # Video ends: flush
        final_inc = manager.end_active_incident(timestamp=1.0)
        self.assertIsNotNone(final_inc)
        self.assertFalse(manager.is_incident_active)
        self.assertEqual(len(manager.incidents), 1)

    def test_detector_bytetrack_tracking(self):
        """Verify AccidentDetector track_frame and track alias execute ByteTrack without errors."""
        detector = AccidentDetector(model_path="best.pt")
        dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)

        # ByteTrack inference
        result = detector.track_frame(dummy_frame)
        self.assertIsInstance(result, DetectionResult)
        self.assertIsNotNone(result.annotated_frame)

        # Alias verification
        alias_result = detector.track(dummy_frame)
        self.assertIsInstance(alias_result, DetectionResult)


if __name__ == "__main__":
    unittest.main()
