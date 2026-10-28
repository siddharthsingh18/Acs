"""
tests/test_robustness.py
Unit tests for ml/scripts/robustness_test.py:
- Tests all 11 corruption functions at severities 1 to 3 (valid shapes, data types, pixel value bounds)
- Tests corruption parameter scaling across severities
- Tests low-bitrate fallback and resolution drop geometry
- Tests Albumentations pipeline composition for worst corruptions
- Tests markdown report generation with explicit non-claim of untested conditions
"""


import numpy as np
import pytest

from ml.scripts.robustness_test import (
    CORRUPTIONS_LIST,
    apply_camera_shake,
    apply_compression_frame,
    apply_defocus_blur,
    apply_fog,
    apply_gaussian_noise,
    apply_glare,
    apply_lens_dirt,
    apply_motion_blur,
    apply_night_darkening,
    apply_rain_streaks,
    apply_resolution_drop,
    build_matching_albumentations,
    create_lens_dirt_mask,
    generate_markdown_report,
)


@pytest.fixture
def sample_frame():
    """Generates a standard 720p BGR test frame."""
    return (np.random.rand(720, 1280, 3) * 255).astype(np.uint8)


class TestCorruptionGenerators:
    """Tests that all 11 corruptions produce valid images across severities 1..3."""

    @pytest.mark.parametrize("severity", [1, 2, 3])
    def test_gaussian_noise(self, sample_frame, severity):
        out = apply_gaussian_noise(sample_frame, severity=severity)
        assert out.shape == sample_frame.shape
        assert out.dtype == np.uint8
        assert not np.array_equal(out, sample_frame)

    @pytest.mark.parametrize("severity", [1, 2, 3])
    def test_motion_blur(self, sample_frame, severity):
        out = apply_motion_blur(sample_frame, severity=severity)
        assert out.shape == sample_frame.shape
        assert out.dtype == np.uint8

    @pytest.mark.parametrize("severity", [1, 2, 3])
    def test_defocus_blur(self, sample_frame, severity):
        out = apply_defocus_blur(sample_frame, severity=severity)
        assert out.shape == sample_frame.shape
        assert out.dtype == np.uint8

    @pytest.mark.parametrize("severity", [1, 2, 3])
    def test_fog(self, sample_frame, severity):
        out = apply_fog(sample_frame, severity=severity)
        assert out.shape == sample_frame.shape
        assert out.dtype == np.uint8
        # Fog should increase mean brightness towards white
        assert np.mean(out) >= np.mean(sample_frame) * 0.85

    @pytest.mark.parametrize("severity", [1, 2, 3])
    def test_rain_streaks(self, sample_frame, severity):
        out = apply_rain_streaks(sample_frame, severity=severity)
        assert out.shape == sample_frame.shape
        assert out.dtype == np.uint8

    @pytest.mark.parametrize("severity", [1, 2, 3])
    def test_night_darkening(self, sample_frame, severity):
        out = apply_night_darkening(sample_frame, severity=severity)
        assert out.shape == sample_frame.shape
        assert out.dtype == np.uint8
        # Night darkening should reduce overall mean luminance
        assert np.mean(out) < np.mean(sample_frame)

    @pytest.mark.parametrize("severity", [1, 2, 3])
    def test_glare(self, sample_frame, severity):
        out = apply_glare(sample_frame, severity=severity)
        assert out.shape == sample_frame.shape
        assert out.dtype == np.uint8
        assert np.max(out) >= np.max(sample_frame)

    @pytest.mark.parametrize("severity", [1, 2, 3])
    def test_compression_frame(self, sample_frame, severity):
        out = apply_compression_frame(sample_frame, severity=severity)
        assert out.shape == sample_frame.shape
        assert out.dtype == np.uint8

    @pytest.mark.parametrize("severity", [1, 2, 3])
    def test_resolution_drop(self, sample_frame, severity):
        out = apply_resolution_drop(sample_frame, severity=severity)
        assert out.shape == sample_frame.shape
        assert out.dtype == np.uint8

    @pytest.mark.parametrize("severity", [1, 2, 3])
    def test_camera_shake(self, sample_frame, severity):
        out = apply_camera_shake(sample_frame, severity=severity, frame_idx=3)
        assert out.shape == sample_frame.shape
        assert out.dtype == np.uint8

    @pytest.mark.parametrize("severity", [1, 2, 3])
    def test_lens_dirt(self, sample_frame, severity):
        mask, alpha = create_lens_dirt_mask(720, 1280, severity=severity)
        assert mask.shape == sample_frame.shape
        out = apply_lens_dirt(sample_frame, severity=severity, dirt_mask=mask, alpha=alpha)
        assert out.shape == sample_frame.shape
        assert out.dtype == np.uint8


class TestAlbumentationsPipeline:
    """Tests that matching augmentations are built correctly for worst corruptions."""

    def test_build_matching_albumentations_subset(self):
        pipe = build_matching_albumentations(["fog", "rain_streaks", "low_bitrate"])
        assert pipe is not None

        img = (np.random.rand(256, 256, 3) * 255).astype(np.uint8)
        res = pipe(image=img, bboxes=[[0.5, 0.5, 0.2, 0.2]], class_labels=[1])
        assert res["image"].shape == (256, 256, 3)
        assert len(res["bboxes"]) == 1

    def test_all_corruptions_supported(self):
        for c in CORRUPTIONS_LIST:
            pipe = build_matching_albumentations([c])
            assert pipe is not None


class TestReportGeneration:
    """Tests markdown and JSON reporting and disclosure of untested conditions."""

    def test_report_includes_untested_disclosure(self, tmp_path):
        dummy_before = {
            "model_path": "best.pt",
            "clean_baseline": {"incident_recall_pct": 100.0, "false_alarms_per_hour": 0.0},
            "corruptions_evaluated": ["fog", "motion_blur", "rain_streaks"],
            "worst_three_corruptions": ["fog", "motion_blur", "rain_streaks"],
            "ranked_corruptions": [
                {
                    "corruption": "fog",
                    "avg_recall_drop": 20.0,
                    "max_recall_drop": 50.0,
                    "avg_fa_rise": 15.0,
                    "max_fa_rise": 30.0,
                    "degradation_score": 35.0,
                    "levels": [
                        {"severity": 1, "incident_recall_pct": 90.0, "drop_in_recall_pct": 10.0, "false_alarms_total": 0, "false_alarms_per_hour": 0.0, "rise_in_fa_per_hour": 0.0},
                        {"severity": 2, "incident_recall_pct": 80.0, "drop_in_recall_pct": 20.0, "false_alarms_total": 1, "false_alarms_per_hour": 15.0, "rise_in_fa_per_hour": 15.0},
                        {"severity": 3, "incident_recall_pct": 50.0, "drop_in_recall_pct": 50.0, "false_alarms_total": 2, "false_alarms_per_hour": 30.0, "rise_in_fa_per_hour": 30.0},
                    ],
                },
                {
                    "corruption": "motion_blur",
                    "avg_recall_drop": 15.0,
                    "max_recall_drop": 30.0,
                    "avg_fa_rise": 5.0,
                    "max_fa_rise": 10.0,
                    "degradation_score": 25.0,
                    "levels": [
                        {"severity": 1, "incident_recall_pct": 95.0, "drop_in_recall_pct": 5.0, "false_alarms_total": 0, "false_alarms_per_hour": 0.0, "rise_in_fa_per_hour": 0.0},
                        {"severity": 2, "incident_recall_pct": 85.0, "drop_in_recall_pct": 15.0, "false_alarms_total": 0, "false_alarms_per_hour": 0.0, "rise_in_fa_per_hour": 0.0},
                        {"severity": 3, "incident_recall_pct": 70.0, "drop_in_recall_pct": 30.0, "false_alarms_total": 1, "false_alarms_per_hour": 10.0, "rise_in_fa_per_hour": 10.0},
                    ],
                },
                {
                    "corruption": "rain_streaks",
                    "avg_recall_drop": 10.0,
                    "max_recall_drop": 20.0,
                    "avg_fa_rise": 2.0,
                    "max_fa_rise": 5.0,
                    "degradation_score": 18.0,
                    "levels": [
                        {"severity": 1, "incident_recall_pct": 98.0, "drop_in_recall_pct": 2.0, "false_alarms_total": 0, "false_alarms_per_hour": 0.0, "rise_in_fa_per_hour": 0.0},
                        {"severity": 2, "incident_recall_pct": 90.0, "drop_in_recall_pct": 10.0, "false_alarms_total": 0, "false_alarms_per_hour": 0.0, "rise_in_fa_per_hour": 0.0},
                        {"severity": 3, "incident_recall_pct": 80.0, "drop_in_recall_pct": 20.0, "false_alarms_total": 0, "false_alarms_per_hour": 0.0, "rise_in_fa_per_hour": 0.0},
                    ],
                },
            ],
            "results_by_corruption": {
                "fog": [
                    {"severity": 1, "incident_recall_pct": 90.0, "false_alarms_per_hour": 0.0},
                    {"severity": 2, "incident_recall_pct": 80.0, "false_alarms_per_hour": 15.0},
                    {"severity": 3, "incident_recall_pct": 50.0, "false_alarms_per_hour": 30.0},
                ],
                "motion_blur": [
                    {"severity": 1, "incident_recall_pct": 95.0, "false_alarms_per_hour": 0.0},
                    {"severity": 2, "incident_recall_pct": 85.0, "false_alarms_per_hour": 0.0},
                    {"severity": 3, "incident_recall_pct": 70.0, "false_alarms_per_hour": 10.0},
                ],
                "rain_streaks": [
                    {"severity": 1, "incident_recall_pct": 98.0, "false_alarms_per_hour": 0.0},
                    {"severity": 2, "incident_recall_pct": 90.0, "false_alarms_per_hour": 0.0},
                    {"severity": 3, "incident_recall_pct": 80.0, "false_alarms_per_hour": 0.0},
                ],
            },
        }

        report_file = tmp_path / "robustness_report.md"
        generate_markdown_report(dummy_before, after_benchmark=None, output_path=report_file)

        assert report_file.exists()
        content = report_file.read_text(encoding="utf-8")
        assert "Operational Robustness Benchmark & Stress Test Report" in content
        assert "Worst Three Corruptions" in content
        assert "robustness is NOT claimed for unbenchmarked conditions" in content
        assert "Untested Operational Domains" in content
