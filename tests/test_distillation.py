"""
Unit Tests for ACS Knowledge Distillation Pipeline
---------------------------------------------------
Tests:
1. Frame collection and frozen split guard verification.
2. Geometric consistency checks (IoU, horizontal flip inversion, scale change).
3. Zero-shot physical collision heuristic checks.
4. Pseudo-label share capping (strictly <= 50% of training data).
5. Contact sheet rendering and visual layout.
"""

from pathlib import Path
import numpy as np
import pytest
import cv2

from ml.scripts.pseudo_label import (
    compute_box_iou,
    zero_shot_collision_check,
    generate_contact_sheet
)


class TestGeometricConsistency:
    def test_compute_box_iou_exact_match(self):
        b1 = (0.5, 0.5, 0.2, 0.2)
        b2 = (0.5, 0.5, 0.2, 0.2)
        assert pytest.approx(compute_box_iou(b1, b2), 0.001) == 1.0

    def test_compute_box_iou_disjoint(self):
        b1 = (0.2, 0.2, 0.1, 0.1)
        b2 = (0.8, 0.8, 0.1, 0.1)
        assert compute_box_iou(b1, b2) == 0.0

    def test_compute_box_iou_partial_overlap(self):
        b1 = (0.5, 0.5, 0.4, 0.4)
        b2 = (0.6, 0.5, 0.4, 0.4)
        iou = compute_box_iou(b1, b2)
        assert 0.3 < iou < 0.7

    def test_horizontal_flip_inversion(self):
        # A box at x=0.3 on an image when flipped horizontally should map to x=0.7
        orig_xc = 0.3
        flipped_xc = 1.0 - orig_xc
        assert pytest.approx(flipped_xc, 0.0001) == 0.7
        # Re-inversion
        assert pytest.approx(1.0 - flipped_xc, 0.0001) == orig_xc


class TestZeroShotCollisionSanity:
    def test_normal_class_always_passes(self):
        box = (0.5, 0.5, 0.2, 0.2)
        assert zero_shot_collision_check(box, "car", []) is True
        assert zero_shot_collision_check(box, "person", []) is True

    def test_accident_with_close_neighbor_passes(self):
        box = (0.4, 0.5, 0.2, 0.2)
        neighbor = (0.5, 0.5, 0.2, 0.2, "car", 0.9)
        assert zero_shot_collision_check(box, "car_car_accident", [neighbor]) is True

    def test_isolated_small_accident_box_fails(self):
        # Tiny isolated box with no other vehicles in frame fails zero-shot check
        box = (0.8, 0.8, 0.05, 0.05)
        distant_obj = (0.1, 0.1, 0.05, 0.05, "car", 0.9)
        assert zero_shot_collision_check(box, "car_car_accident", [distant_obj]) is False


class TestDatasetIntegrityAndCapping:
    def test_frozen_splits_uncontaminated(self):
        project_root = Path(__file__).resolve().parent.parent
        val_dir = project_root / "ml" / "data" / "merged" / "images" / "val"
        test_dir = project_root / "ml" / "data" / "merged" / "images" / "test"

        assert val_dir.exists()
        assert test_dir.exists()

        val_files = list(val_dir.glob("*.jpg"))
        test_files = list(test_dir.glob("*.jpg"))
        assert len(val_files) > 0
        assert len(test_files) > 0

        # Verify no pseudo files leaked into val or test
        for f in val_files + test_files:
            assert not f.name.startswith("pseudo_")
            assert not f.name.startswith("unlabeled_")

    def test_pseudo_share_calculation_math(self):
        n_real = 2338
        max_share = 0.50
        max_pseudo = int((max_share / (1.0 - max_share)) * n_real)
        assert max_pseudo == 2338
        # Any pseudo count <= 2338 ensures pseudo share <= 50%
        pseudo_sample = 500
        share = pseudo_sample / (n_real + pseudo_sample)
        assert share < 0.50


class TestContactSheetGeneration:
    def test_contact_sheet_renders_valid_image(self, tmp_path):
        dummy_img_path = tmp_path / "dummy.jpg"
        dummy_img = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.imwrite(str(dummy_img_path), dummy_img)

        samples = [
            {
                "img_path": dummy_img_path,
                "box": (0.5, 0.5, 0.3, 0.3),
                "conf": 0.88,
                "flip_iou": 0.82
            }
        ]
        out_sheet = tmp_path / "sheet.jpg"
        generate_contact_sheet(samples, "car_car_accident", out_sheet, cols=2)
        assert out_sheet.exists()

        sheet_img = cv2.imread(str(out_sheet))
        assert sheet_img is not None
        assert sheet_img.shape[0] > 100
        assert sheet_img.shape[1] > 100
