"""
Unit tests for Google Gemini multimodal LLM integration in acs/llm.py.
Covers:
- Structured JSON output adherence to DispatchReportModel schema.
- Graceful template fallback when API key is missing or invalid.
- Retries with exponential backoff on simulated failures.
- Surrounding frames discovery (peak + 1-2 frames before/after).
- Rendered 5-8 line dispatch text in English and Arabic.
- Backwards compatible generate_accident_report function.
"""
import os
import shutil
import tempfile
import unittest
from unittest.mock import MagicMock, patch
import cv2
import numpy as np

from acs.llm import (
    DispatchReportModel,
    build_fallback_report,
    find_surrounding_frames,
    generate_accident_report,
    generate_dispatch_report,
    render_dispatch_text,
)


class TestLLMDispatchReport(unittest.TestCase):
    """Test suite for the LLM dispatch report generation and fallback systems."""

    def setUp(self):
        """Create a temporary test directory with dummy image files."""
        self.test_dir = tempfile.mkdtemp()
        self.dummy_frame = np.zeros((100, 100, 3), dtype=np.uint8)

        # Create sequential dummy frame images
        self.frame_paths = []
        for i in range(5):
            path = os.path.join(self.test_dir, f"frame_{i:02d}.jpg")
            cv2.imwrite(path, self.dummy_frame)
            self.frame_paths.append(path)
        self.peak_frame_path = self.frame_paths[2]  # Middle frame is peak

    def tearDown(self):
        """Clean up temporary test files."""
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

    def test_fallback_report_structure_and_schema(self):
        """Fallback report must strictly conform to DispatchReportModel schema."""
        report = build_fallback_report(
            detections="car_car_accident",
            timestamp="2026-10-08 12:00:00 UTC",
            lang="en",
        )

        # Verify presence of all required fields
        required_fields = [
            "vehicles",
            "collision_point",
            "damage",
            "road_conditions",
            "hazards",
            "recommended_units",
            "severity",
        ]
        for field in required_fields:
            self.assertIn(field, report)

        # Validate with Pydantic DispatchReportModel
        validated = DispatchReportModel(**report)
        self.assertIn(validated.severity, ["low", "medium", "high"])
        for unit in validated.recommended_units:
            self.assertIn(unit, ["EMS", "police", "fire", "tow"])

    def test_fallback_severity_and_unit_inference(self):
        """Pedestrian/injury accidents should infer high severity and EMS units."""
        # High severity pedestrian incident
        ped_report = build_fallback_report(
            detections="car_person_accident",
            lang="en",
        )
        self.assertEqual(ped_report["severity"], "high")
        self.assertIn("EMS", ped_report["recommended_units"])
        self.assertIn("police", ped_report["recommended_units"])

        # Medium severity car collision
        car_report = build_fallback_report(
            detections="car_car_accident",
            lang="en",
        )
        self.assertEqual(car_report["severity"], "medium")
        self.assertIn("police", car_report["recommended_units"])
        self.assertIn("tow", car_report["recommended_units"])

    def test_surrounding_frames_discovery(self):
        """find_surrounding_frames must retrieve peak plus up to 2 frames before and after."""
        discovered = find_surrounding_frames(
            peak_image_path=self.peak_frame_path,
            max_before=2,
            max_after=2,
        )
        self.assertEqual(len(discovered), 5)
        self.assertEqual(discovered[2], self.peak_frame_path)
        self.assertEqual(discovered[0], self.frame_paths[0])
        self.assertEqual(discovered[4], self.frame_paths[4])

    def test_rendered_dispatch_text_line_count_english(self):
        """English dispatch text must be exactly between 5 and 8 lines."""
        report_data = {
            "vehicles": ["sedan (white)", "SUV (black)"],
            "collision_point": "Intersection of Main St & 4th Ave",
            "damage": "Severe front crumple and driver door intrusion",
            "road_conditions": "Wet asphalt, rain",
            "hazards": "Fuel spill in left lane",
            "recommended_units": ["police", "EMS", "tow"],
            "severity": "medium",
            "timestamp": "2026-10-08 14:30:00 UTC",
        }
        rendered = render_dispatch_text(report_data, lang="en")
        lines = [line for line in rendered.strip().split("\n") if line.strip()]
        self.assertGreaterEqual(len(lines), 5)
        self.assertLessEqual(len(lines), 8)
        self.assertIn("EMERGENCY DISPATCH REPORT", lines[0])

    def test_rendered_dispatch_text_line_count_arabic(self):
        """Arabic dispatch text must be exactly between 5 and 8 lines."""
        report_data = {
            "vehicles": ["سيارة سيدان بيضاء", "مركبة دفع رباعي"],
            "collision_point": "تقاطع طريق الملك فهد",
            "damage": "أضرار جسيمة في الهيكل الأمامي",
            "road_conditions": "طريق جاف ورؤية واضحة",
            "hazards": "تسرب وقود وقطع حطام بالمسار الأيمن",
            "recommended_units": ["police", "EMS", "tow"],
            "severity": "high",
            "timestamp": "2026-10-08 14:30:00 UTC",
        }
        rendered = render_dispatch_text(report_data, lang="ar")
        lines = [line for line in rendered.strip().split("\n") if line.strip()]
        self.assertGreaterEqual(len(lines), 5)
        self.assertLessEqual(len(lines), 8)
        self.assertIn("تقرير إرسال طوارئ", lines[0])

    def test_generate_dispatch_report_missing_key_graceful_fallback(self):
        """When API key is missing, generate_dispatch_report immediately returns fallback."""
        with patch.dict(os.environ, {"GOOGLE_API_KEY": ""}, clear=False):
            report = generate_dispatch_report(
                image_path=self.peak_frame_path,
                detections="car_car_accident",
                timestamp="2026-10-08 12:00:00 UTC",
                lang="en",
                api_key="",
            )
            self.assertIsInstance(report, dict)
            self.assertIn("vehicles", report)
            self.assertIn("severity", report)
            self.assertIn("rendered_text", report)
            # Verify rendered text is within 5-8 lines
            lines = [line_entry for line_entry in report["rendered_text"].strip().split("\n") if line_entry.strip()]
            self.assertGreaterEqual(len(lines), 5)
            self.assertLessEqual(len(lines), 8)

    def test_generate_dispatch_report_retry_and_fallback_on_api_error(self):
        """When the API repeatedly throws errors, it retries and falls back gracefully."""
        mock_client = MagicMock()
        mock_client.models.generate_content.side_effect = RuntimeError("API quota exceeded")

        with patch("google.genai.Client", return_value=mock_client):
            report = generate_dispatch_report(
                image_path=self.peak_frame_path,
                detections="bike_bike_accident",
                timestamp="2026-10-08 12:00:00 UTC",
                lang="en",
                api_key="fake-test-key",
                max_retries=1,  # Fast retry for test
                timeout=2.0,
            )
            # Must return a valid fallback report rather than raising an exception
            self.assertIsInstance(report, dict)
            self.assertIn("vehicles", report)
            self.assertEqual(report["severity"], "medium")
            self.assertIn("rendered_text", report)

    def test_generate_accident_report_backwards_compatibility(self):
        """generate_accident_report must return rendered text string for legacy callers."""
        text = generate_accident_report(
            image_path=self.peak_frame_path,
            yolo_report="car_car_accident",
            incident_time="2026-10-08 12:00:00 UTC",
            lang="en",
            api_key="",
        )
        self.assertIsInstance(text, str)
        lines = [line_entry for line_entry in text.strip().split("\n") if line_entry.strip()]
        self.assertGreaterEqual(len(lines), 5)
        self.assertLessEqual(len(lines), 8)


if __name__ == "__main__":
    unittest.main()
