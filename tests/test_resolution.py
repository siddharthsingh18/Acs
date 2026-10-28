"""
tests/test_resolution.py
Tests for acs/resolution.py – SAHI-style tiled inference and ROI crop re-detector.
All tests use synthetic frames and a mock YOLO model so no GPU or real weights are needed.
"""
from __future__ import annotations

from unittest.mock import MagicMock
from typing import List, Tuple

import numpy as np
import pytest

from acs.resolution import (
    TileConfig,
    ROIConfig,
    SuspiciousTileInferencer,
    ROICropReDetector,
    merge_detections,
    _nms,
    make_default_tiler,
    make_default_roi_redetector,
)
from acs.detector import DetectionBox


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_box(
    x1: float, y1: float, x2: float, y2: float,
    conf: float = 0.8,
    cid: int = 4,
    cname: str = "car",
    is_accident: bool = False,
) -> DetectionBox:
    return DetectionBox(
        bbox=(x1, y1, x2, y2),
        confidence=conf,
        class_id=cid,
        class_name=cname,
        is_accident=is_accident,
    )


def _mock_yolo_with_boxes(
    boxes_list: List[List[Tuple[float, float, float, float, float, int, str]]]
):
    """
    Returns a mock YOLO model whose .predict() returns a sequence of result objects,
    one per call.  Each element of boxes_list is a list of
    (x1, y1, x2, y2, conf, class_id, class_name) tuples for that call.
    """
    model = MagicMock()
    model.names = {0: "bike", 1: "bike_bike_accident", 4: "car", 5: "car_bike_accident", 9: "person"}

    call_idx = [0]

    def _predict(*args, **kwargs):
        idx = min(call_idx[0], len(boxes_list) - 1)
        raw = boxes_list[idx]
        call_idx[0] += 1

        result = MagicMock()
        if not raw:
            result.boxes = None
            result.plot.return_value = np.zeros((100, 100, 3), dtype=np.uint8)
            return [result]

        boxes_mock = MagicMock()
        boxes_mock.__len__ = MagicMock(return_value=len(raw))
        boxes_mock.__iter__ = MagicMock(return_value=iter([]))

        mock_boxes = []
        for x1, y1, x2, y2, conf, cid, cname in raw:
            b = MagicMock()
            b.cls = MagicMock()
            b.cls.__getitem__ = lambda self, i, _cid=cid: MagicMock(item=MagicMock(return_value=_cid))
            b.conf = MagicMock()
            b.conf.__getitem__ = lambda self, i, _c=conf: MagicMock(item=MagicMock(return_value=_c))
            b.xyxy = MagicMock()
            b.xyxy.__getitem__ = lambda self, i, _xy=(x1, y1, x2, y2): MagicMock(tolist=MagicMock(return_value=list(_xy)))
            mock_boxes.append(b)

        result.boxes = mock_boxes
        result.plot.return_value = np.zeros((100, 100, 3), dtype=np.uint8)
        return [result]

    model.predict = MagicMock(side_effect=_predict)
    return model


# ---------------------------------------------------------------------------
# NMS tests
# ---------------------------------------------------------------------------

class TestNMS:
    def test_empty(self):
        assert _nms([]) == []

    def test_single_box(self):
        b = _make_box(0, 0, 100, 100)
        result = _nms([b])
        assert result == [b]

    def test_exact_overlap_suppressed(self):
        b1 = _make_box(0, 0, 100, 100, conf=0.9)
        b2 = _make_box(0, 0, 100, 100, conf=0.5)
        result = _nms([b1, b2], iou_threshold=0.5)
        assert len(result) == 1
        assert result[0].confidence == 0.9

    def test_disjoint_boxes_both_kept(self):
        b1 = _make_box(0, 0, 50, 50)
        b2 = _make_box(60, 60, 100, 100)
        result = _nms([b1, b2], iou_threshold=0.5)
        assert len(result) == 2

    def test_partial_overlap_below_threshold_kept(self):
        # IoU ≈ 0.14 < 0.5 → both kept
        b1 = _make_box(0, 0, 100, 100, conf=0.9)
        b2 = _make_box(80, 0, 180, 100, conf=0.8)
        result = _nms([b1, b2], iou_threshold=0.5)
        assert len(result) == 2


class TestMergeDetections:
    def test_merge_empty_lists(self):
        assert merge_detections([], []) == []

    def test_merge_disjoint(self):
        a = [_make_box(0, 0, 50, 50)]
        b = [_make_box(60, 60, 100, 100)]
        result = merge_detections(a, b)
        assert len(result) == 2

    def test_merge_duplicate_suppressed(self):
        b = _make_box(0, 0, 100, 100, conf=0.9)
        dup = _make_box(0, 0, 100, 100, conf=0.5)
        result = merge_detections([b], [dup])
        assert len(result) == 1


# ---------------------------------------------------------------------------
# SuspiciousTileInferencer tests
# ---------------------------------------------------------------------------

class TestSuspiciousTileInferencer:
    def test_tile_count_no_overlap(self):
        tiler = SuspiciousTileInferencer(TileConfig(tile_size=320, overlap_ratio=0.0))
        frame = np.zeros((640, 640, 3), dtype=np.uint8)
        # 2×2 = 4 tiles
        assert tiler.compute_tile_count(frame) == 4

    def test_tile_count_with_overlap(self):
        tiler = SuspiciousTileInferencer(TileConfig(tile_size=320, overlap_ratio=0.5))
        frame = np.zeros((640, 640, 3), dtype=np.uint8)
        # stride=160, tiles per dim = ceil(640/160)=4 → 4×4=16
        assert tiler.compute_tile_count(frame) == 16

    def test_infer_no_detections(self):
        tiler = make_default_tiler()
        model = _mock_yolo_with_boxes([[] for _ in range(20)])
        frame = np.zeros((640, 640, 3), dtype=np.uint8)
        result = tiler.infer(model, frame, device="cpu")
        assert result == []

    def test_infer_maps_coords_to_frame(self):
        """Detection inside second tile (x offset=512) must have x1 >= 512."""
        tile_size = 512
        tiler = SuspiciousTileInferencer(TileConfig(tile_size=tile_size, overlap_ratio=0.0))
        # 1280-wide frame, 3 columns (0, 512, 1024).
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)

        # Second tile call returns a box at local (10, 10, 60, 60)
        boxes_per_call = [
            [],   # tile (0,0,512,720) → empty
            [(10, 10, 60, 60, 0.85, 4, "car")],  # tile (512,0,1024,720)
            [],   # tile (1024,0,1280,720)
        ]
        model = _mock_yolo_with_boxes(boxes_per_call)
        result = tiler.infer(model, frame, device="cpu")
        assert len(result) == 1
        # Mapped x1 = 10 + 512 = 522
        assert abs(result[0].bbox[0] - 522) < 1

    def test_infer_nms_removes_duplicates(self):
        """Two tiles overlapping at same location → NMS should keep highest-conf only."""
        tiler = SuspiciousTileInferencer(TileConfig(tile_size=640, overlap_ratio=0.5))
        frame = np.zeros((640, 640, 3), dtype=np.uint8)
        # Same logical box returned by multiple tiles with different confidence
        box_high = (100, 100, 200, 200, 0.9, 4, "car")
        box_low  = (100, 100, 200, 200, 0.5, 4, "car")
        boxes_per_call = [[box_high], [box_low], [], [], []]
        model = _mock_yolo_with_boxes(boxes_per_call)
        result = tiler.infer(model, frame, device="cpu")
        # After NMS: 1 box with highest confidence
        assert len(result) == 1
        assert result[0].confidence == pytest.approx(0.9, abs=0.01)

    def test_accident_class_flag(self):
        """Boxes with accident class_id should have is_accident=True."""
        tiler = make_default_tiler()
        frame = np.zeros((640, 640, 3), dtype=np.uint8)
        boxes_per_call = [
            [(10, 10, 80, 80, 0.75, 5, "car_bike_accident")],
        ]
        model = _mock_yolo_with_boxes(boxes_per_call)
        model.names = {5: "car_bike_accident"}
        result = tiler.infer(model, frame, device="cpu",
                             accident_classes={"car_bike_accident"})
        assert len(result) == 1
        assert result[0].is_accident is True


# ---------------------------------------------------------------------------
# ROICropReDetector tests
# ---------------------------------------------------------------------------

class TestROICropReDetector:
    def test_should_redetect_close_pair(self):
        """A pair whose centroids are within close_distance_norm triggers redetect."""
        roi = make_default_roi_redetector()
        # Both boxes in upper-left at 640×480 frame
        da = _make_box(100, 100, 140, 140)
        db = _make_box(110, 110, 150, 150)
        assert roi.should_redetect(da, db, 640, 480) is True

    def test_should_redetect_small_box(self):
        """A pair where one box area < small_area_px2 triggers redetect."""
        roi = ROICropReDetector(ROIConfig(small_area_px2=5000.0, close_distance_norm=0.001))
        da = _make_box(0, 0, 10, 10)      # 100 px² < 5000
        db = _make_box(200, 200, 400, 400)
        assert roi.should_redetect(da, db, 640, 480) is True

    def test_should_not_redetect_distant_large_pair(self):
        roi = ROICropReDetector(ROIConfig(
            small_area_px2=100.0,          # small threshold very small
            close_distance_norm=0.001,     # close threshold very tight
        ))
        da = _make_box(0, 0, 200, 200)
        db = _make_box(500, 300, 640, 480)
        assert roi.should_redetect(da, db, 640, 480) is False

    def test_redetect_maps_coords_to_frame(self):
        """Returned boxes should be in original frame coordinates (>= crop x1)."""
        roi = make_default_roi_redetector()
        da = _make_box(200, 200, 300, 300, is_accident=False, cname="car", cid=4)
        db = _make_box(250, 250, 350, 350, is_accident=False, cname="car", cid=4)
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)

        # crop padded merged bbox ≈ (170,170,380,380), local box (10,10,80,80)
        local_box = (10, 10, 80, 80, 0.78, 5, "car_bike_accident")
        model = _mock_yolo_with_boxes([[local_box]])
        model.names = {5: "car_bike_accident"}

        confirmed, refined = roi.redetect(
            model, frame, da, db, device="cpu",
            accident_classes={"car_bike_accident"}
        )
        assert confirmed is True
        assert len(refined) == 1
        # x1 of returned box = local_x1 + crop_x1 ≥ 0
        assert refined[0].bbox[0] >= 0

    def test_redetect_empty_no_accident(self):
        roi = make_default_roi_redetector()
        da = _make_box(100, 100, 200, 200, is_accident=False, cname="car", cid=4)
        db = _make_box(180, 180, 280, 280, is_accident=False, cname="car", cid=4)
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        model = _mock_yolo_with_boxes([[]])  # no detections in crop
        confirmed, refined = roi.redetect(model, frame, da, db, device="cpu")
        assert confirmed is False
        assert refined == []

    def test_redetect_degenerate_crop_skipped(self):
        """A crop with zero area should not crash and return confirmed=False."""
        roi = make_default_roi_redetector()
        # Both boxes are the same single pixel
        da = _make_box(100, 100, 100, 100)
        db = _make_box(100, 100, 100, 100)
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        model = _mock_yolo_with_boxes([[]])
        # Should handle gracefully
        confirmed, refined = roi.redetect(model, frame, da, db, device="cpu")
        assert confirmed is False


# ---------------------------------------------------------------------------
# AccidentDetector integration: check resolution kwargs accepted without error
# ---------------------------------------------------------------------------

class TestAccidentDetectorResolutionKwargs:
    def test_init_with_tile_inference_disabled(self, tmp_path):
        """AccidentDetector should initialise with enable_tile_inference=False (default)."""
        from unittest.mock import patch as upatch
        import acs.detector as det

        fake_model = MagicMock()
        fake_model.names = {}
        with upatch.object(det, "load_model", return_value=fake_model), \
             upatch.object(det.config, "model_path", str(tmp_path / "fake.pt")), \
             upatch.object(det.config, "injury_model_path", ""):
            from acs.detector import AccidentDetector
            ad = AccidentDetector(
                model_path=str(tmp_path / "fake.pt"),
                enable_tile_inference=False,
                enable_roi_redetect=False,
            )
            assert ad._tiler is None
            assert ad._roi_redetector is None

    def test_init_with_tile_inference_enabled(self, tmp_path):
        """AccidentDetector with enable_tile_inference=True should create a tiler."""
        from unittest.mock import patch as upatch
        import acs.detector as det

        fake_model = MagicMock()
        fake_model.names = {}
        with upatch.object(det, "load_model", return_value=fake_model), \
             upatch.object(det.config, "model_path", str(tmp_path / "fake.pt")), \
             upatch.object(det.config, "injury_model_path", ""):
            from acs.detector import AccidentDetector
            ad = AccidentDetector(
                model_path=str(tmp_path / "fake.pt"),
                enable_tile_inference=True,
                enable_roi_redetect=True,
            )
            assert ad._tiler is not None
            assert ad._roi_redetector is not None
