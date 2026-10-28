"""
acs.resolution
──────────────────────────────────────────────────────────────────────────────
Two complementary techniques to improve small/far-field object recall without
running every frame at high resolution:

1.  SAHI-style *suspicious-region tiled inference* (SuspiciousTileInferencer)
    – Applied only to frames (or regions) already flagged as suspicious by the
      ByteTrack event manager (e.g. two tracks converging rapidly).
    – Slices the frame into overlapping tiles, runs the detector on each tile
      at imgsz=640, then merges detections back to frame coordinates via
      weighted-box fusion (NMS-equivalent).

2.  ROI crop re-detector (ROICropReDetector)
    – Called when a track pair's centroid distance falls below a configurable
      threshold (e.g. the pair is "close").
    – Crops a padded bounding box around the pair, re-runs the detector at a
      higher resolution (e.g. 960), and merges the refined detections.
    – Returns confirmed=True/False along with the merged detection list.

Both classes are stateless w.r.t. video frames and are safe to call from any
thread.  They accept the primary YOLO model and return DetectionBox lists in
original-frame pixel coordinates so the caller can merge with existing results.

Usage example (inside predict_frame):
    if suspicious:
        extra = tiler.infer(model, frame, conf=0.35)
        detections = merge_nms(detections + extra, iou=0.5)

    if track_pair_close:
        confirmed, extra = roi_redetect.redetect(model, frame, bbox_pair, conf=0.40)
        detections = merge_nms(detections + extra, iou=0.5)
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np

from acs.config import ACCIDENT_CLASSES  # lightweight, no circular dep
from acs.taxonomy import decode_model_class

from acs.detector import DetectionBox

# ---------------------------------------------------------------------------
# NMS helper – operates entirely in pixel coords
# ---------------------------------------------------------------------------

def _nms(
    boxes: List[DetectionBox],
    iou_threshold: float = 0.50,
) -> List[DetectionBox]:
    """Greedy NMS over DetectionBox list, sorted descending by confidence."""
    if not boxes:
        return []
    boxes = sorted(boxes, key=lambda b: b.confidence, reverse=True)
    kept: List[DetectionBox] = []
    suppressed = [False] * len(boxes)
    for i, bi in enumerate(boxes):
        if suppressed[i]:
            continue
        kept.append(bi)
        for j in range(i + 1, len(boxes)):
            if suppressed[j]:
                continue
            bj = boxes[j]
            # compute IoU
            x1 = max(bi.bbox[0], bj.bbox[0])
            y1 = max(bi.bbox[1], bj.bbox[1])
            x2 = min(bi.bbox[2], bj.bbox[2])
            y2 = min(bi.bbox[3], bj.bbox[3])
            iw = max(0.0, x2 - x1)
            ih = max(0.0, y2 - y1)
            inter = iw * ih
            area_i = max(1e-9, (bi.bbox[2] - bi.bbox[0]) * (bi.bbox[3] - bi.bbox[1]))
            area_j = max(1e-9, (bj.bbox[2] - bj.bbox[0]) * (bj.bbox[3] - bj.bbox[1]))
            union = area_i + area_j - inter
            iou = inter / union if union > 0 else 0.0
            if iou >= iou_threshold:
                suppressed[j] = True
    return kept


def merge_detections(
    a: List[DetectionBox],
    b: List[DetectionBox],
    iou_threshold: float = 0.50,
) -> List[DetectionBox]:
    """Merges two detection lists via greedy NMS."""
    return _nms(a + b, iou_threshold)


# ---------------------------------------------------------------------------
# 1. Suspicious-region tiled inference (SAHI-style)
# ---------------------------------------------------------------------------

@dataclass
class TileConfig:
    """Configuration for SAHI-style tiled inference."""
    tile_size: int = 640        # pixels per tile (square)
    overlap_ratio: float = 0.20 # fractional overlap between adjacent tiles
    min_tile_conf: float = 0.30 # lower conf inside tiles (compensates for partial boxes)
    nms_iou: float = 0.50       # IoU threshold for post-tile NMS
    infer_imgsz: int = 640      # imgsz passed to YOLO predict for each tile


class SuspiciousTileInferencer:
    """
    Applies tiled inference ONLY to frames already flagged as suspicious.

    Design choices that keep FPS on budget:
      - Called at most once per event window (not every frame).
      - The caller is responsible for flagging only the suspicious frames
        (e.g. when the temporal gate probability > 0.3 but < threshold, or
        when a track pair's centroid distance drops below a configurable limit).
      - Returns additional DetectionBox candidates in original-frame coordinates.
      - Does NOT re-run the full tracker; caller merges into existing detections.

    Example (in AccidentDetector.predict_frame):
        if suspicious_flag:
            extra = self.tiler.infer(self.primary_model, processed_frame, ...)
            detections = merge_detections(detections, extra)
    """

    def __init__(self, cfg: Optional[TileConfig] = None) -> None:
        self.cfg = cfg or TileConfig()

    def infer(
        self,
        model,  # ultralytics YOLO
        frame: np.ndarray,
        device: str = "cpu",
        accident_classes: Optional[set] = None,
    ) -> List[DetectionBox]:
        """
        Tiles ``frame``, runs YOLO on each tile, and returns merged detections
        in original-frame pixel coordinates.

        Args:
            model:           Loaded ultralytics YOLO model.
            frame:           BGR numpy array (H, W, 3).
            device:          Inference device ('mps', 'cuda', 'cpu').
            accident_classes: Set of accident class names (from acs.config).

        Returns:
            List[DetectionBox] in original-frame pixel coordinates.
        """
        acc_cls = accident_classes or set(ACCIDENT_CLASSES)
        cfg = self.cfg
        H, W = frame.shape[:2]

        stride = max(1, int(cfg.tile_size * (1.0 - cfg.overlap_ratio)))
        tiles: List[Tuple[int, int, int, int]] = []  # (x1, y1, x2, y2)

        y = 0
        while y < H:
            x = 0
            while x < W:
                x2 = min(x + cfg.tile_size, W)
                y2 = min(y + cfg.tile_size, H)
                tiles.append((x, y, x2, y2))
                if x2 == W:
                    break
                x += stride
            if y + cfg.tile_size >= H:
                break
            y += stride

        all_boxes: List[DetectionBox] = []
        for tx1, ty1, tx2, ty2 in tiles:
            tile = frame[ty1:ty2, tx1:tx2]
            results = model.predict(
                source=tile,
                imgsz=cfg.infer_imgsz,
                conf=cfg.min_tile_conf,
                verbose=False,
                device=device,
            )
            boxes = results[0].boxes
            if boxes is None or len(boxes) == 0:
                continue
            for box in boxes:
                cid = int(box.cls[0].item())
                conf = float(box.conf[0].item())
                cid, cname, incident_type = decode_model_class(cid, model.names)
                is_acc = cname.lower().strip() in acc_cls
                xyxy = box.xyxy[0].tolist()
                # Map tile-local coords → frame coords
                all_boxes.append(DetectionBox(
                    bbox=(
                        xyxy[0] + tx1,
                        xyxy[1] + ty1,
                        xyxy[2] + tx1,
                        xyxy[3] + ty1,
                    ),
                    confidence=conf,
                    class_id=cid,
                    class_name=cname,
                    is_accident=is_acc,
                    track_id=None,
                    incident_type=incident_type,
                ))

        return _nms(all_boxes, cfg.nms_iou)

    def compute_tile_count(self, frame: np.ndarray) -> int:
        """Returns the number of tiles for a given frame (diagnostic helper)."""
        H, W = frame.shape[:2]
        stride = max(1, int(self.cfg.tile_size * (1.0 - self.cfg.overlap_ratio)))
        nx = math.ceil(W / stride)
        ny = math.ceil(H / stride)
        return nx * ny


# ---------------------------------------------------------------------------
# 2. ROI crop re-detector
# ---------------------------------------------------------------------------

@dataclass
class ROIConfig:
    """Configuration for ROI crop re-detection."""
    pad_ratio: float = 0.30   # extra padding around the merged bbox of a track pair
    redetect_imgsz: int = 960 # resolution for the crop re-run
    conf: float = 0.40        # confidence threshold inside the crop
    nms_iou: float = 0.50
    # Pixel area threshold below which we consider vehicles "small/far"
    # and worth re-running.  At 640 wide, 32px²=1024 px²; scale accordingly.
    small_area_px2: float = 1600.0
    # Centroid distance (normalized) below which pairs are considered "close"
    close_distance_norm: float = 0.12


class ROICropReDetector:
    """
    Re-runs the detector at higher resolution on a cropped region around a
    suspicious track pair.

    Called from the temporal gate integration layer – not from every frame –
    only when:
      (a) The normalised centroid distance between two tracked objects drops
          below ``close_distance_norm``, OR
      (b) The tracked objects are classified as small/far (area < threshold).

    Returns the refined DetectionBox list mapped back to full-frame coordinates.
    """

    def __init__(self, cfg: Optional[ROIConfig] = None) -> None:
        self.cfg = cfg or ROIConfig()

    def _merged_bbox(self, boxes: Sequence[Tuple[float, float, float, float]]) -> Tuple[float, float, float, float]:
        """Merged bounding box that covers all input boxes."""
        x1 = min(b[0] for b in boxes)
        y1 = min(b[1] for b in boxes)
        x2 = max(b[2] for b in boxes)
        y2 = max(b[3] for b in boxes)
        return (x1, y1, x2, y2)

    def should_redetect(
        self,
        det_a: DetectionBox,
        det_b: DetectionBox,
        frame_w: int,
        frame_h: int,
    ) -> bool:
        """
        Heuristic gate: True if the pair is close or either box is small.
        This is called cheaply BEFORE the expensive re-detection.
        """
        cfg = self.cfg
        # Centroid distance (normalised)
        cxa = (det_a.bbox[0] + det_a.bbox[2]) / 2.0 / max(frame_w, 1)
        cya = (det_a.bbox[1] + det_a.bbox[3]) / 2.0 / max(frame_h, 1)
        cxb = (det_b.bbox[0] + det_b.bbox[2]) / 2.0 / max(frame_w, 1)
        cyb = (det_b.bbox[1] + det_b.bbox[3]) / 2.0 / max(frame_h, 1)
        dist = math.hypot(cxa - cxb, cya - cyb)
        if dist < cfg.close_distance_norm:
            return True
        # Area of smaller box
        area_a = (det_a.bbox[2] - det_a.bbox[0]) * (det_a.bbox[3] - det_a.bbox[1])
        area_b = (det_b.bbox[2] - det_b.bbox[0]) * (det_b.bbox[3] - det_b.bbox[1])
        if min(area_a, area_b) < cfg.small_area_px2:
            return True
        return False

    def redetect(
        self,
        model,           # ultralytics YOLO
        frame: np.ndarray,
        det_a: DetectionBox,
        det_b: DetectionBox,
        device: str = "cpu",
        accident_classes: Optional[set] = None,
    ) -> Tuple[bool, List[DetectionBox]]:
        """
        Crops the padded union of (det_a, det_b) from ``frame``, re-runs the
        detector at higher resolution, and returns:
          (accident_confirmed, refined_detections_in_frame_coords)

        If either model fires an accident class in the crop → accident_confirmed=True.
        """
        acc_cls = accident_classes or set(ACCIDENT_CLASSES)
        cfg = self.cfg
        H, W = frame.shape[:2]

        merged = self._merged_bbox([det_a.bbox, det_b.bbox])
        bw = merged[2] - merged[0]
        bh = merged[3] - merged[1]
        pad_x = bw * cfg.pad_ratio
        pad_y = bh * cfg.pad_ratio

        x1 = max(0, int(merged[0] - pad_x))
        y1 = max(0, int(merged[1] - pad_y))
        x2 = min(W, int(merged[2] + pad_x))
        y2 = min(H, int(merged[3] + pad_y))

        if x2 <= x1 or y2 <= y1:
            return False, []

        crop = frame[y1:y2, x1:x2]
        results = model.predict(
            source=crop,
            imgsz=cfg.redetect_imgsz,
            conf=cfg.conf,
            verbose=False,
            device=device,
        )

        refined: List[DetectionBox] = []
        accident_confirmed = False

        boxes = results[0].boxes
        if boxes is not None and len(boxes) > 0:
            for box in boxes:
                cid = int(box.cls[0].item())
                conf = float(box.conf[0].item())
                cid, cname, incident_type = decode_model_class(cid, model.names)
                is_acc = cname.lower().strip() in acc_cls
                xyxy = box.xyxy[0].tolist()
                # Map crop-local → frame coords
                refined.append(DetectionBox(
                    bbox=(
                        xyxy[0] + x1,
                        xyxy[1] + y1,
                        xyxy[2] + x1,
                        xyxy[3] + y1,
                    ),
                    confidence=conf,
                    class_id=cid,
                    class_name=cname,
                    is_accident=is_acc,
                    track_id=None,
                    incident_type=incident_type,
                ))
                if is_acc:
                    accident_confirmed = True

        return accident_confirmed, _nms(refined, cfg.nms_iou)


# ---------------------------------------------------------------------------
# Convenience factory
# ---------------------------------------------------------------------------

def make_default_tiler(tile_size: int = 640, overlap: float = 0.20) -> SuspiciousTileInferencer:
    """Returns a SuspiciousTileInferencer with sensible defaults."""
    return SuspiciousTileInferencer(TileConfig(tile_size=tile_size, overlap_ratio=overlap))


def make_default_roi_redetector(redetect_imgsz: int = 960) -> ROICropReDetector:
    """Returns an ROICropReDetector with sensible defaults."""
    return ROICropReDetector(ROIConfig(redetect_imgsz=redetect_imgsz))
