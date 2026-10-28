"""
acs.detector
Provides model loading, ByteTrack tracking inference, structured detections,
and the AccidentEventManager state machine for incident lifecycle tracking.
No Streamlit dependencies are included to ensure testability.
"""
from collections import deque
from dataclasses import dataclass, field
import datetime
import logging
import os
import time
from typing import TYPE_CHECKING, Any, Deque, Dict, List, Optional, Set, Tuple
import cv2
import numpy as np
from ultralytics import YOLO

from acs.config import config, ACCIDENT_CLASSES
from acs.taxonomy import decode_model_class

if TYPE_CHECKING:
    from acs.resolution import ROIConfig, TileConfig

@dataclass
class DetectionBox:
    """Represents a single detected or tracked bounding box and its metadata."""

    bbox: Tuple[float, float, float, float]  # (x1, y1, x2, y2)
    confidence: float
    class_id: int
    class_name: str
    is_accident: bool
    track_id: Optional[int] = None
    incident_type: Optional[str] = None


# Optional resolution-enhancement helpers (SAHI tiles + ROI crop)
_RESOLUTION_AVAILABLE = True

logger = logging.getLogger("acs.detector")


@dataclass
class DetectionResult:
    """Contains structured detection outputs and the annotated frame for a single video frame."""

    annotated_frame: np.ndarray
    detections: List[DetectionBox] = field(default_factory=list)
    accident_detected: bool = False
    detected_classes: Set[str] = field(default_factory=set)
    accident_classes: Set[str] = field(default_factory=set)


@dataclass
class AccidentIncident:
    """Represents a discrete accident incident spanning multiple video frames."""

    incident_id: str
    start_timestamp: str
    end_timestamp: str
    class_name: str
    max_confidence: float
    image_path: str
    start_time: float
    end_time: float
    incident_type: Optional[str] = None
    keyframe: Optional[np.ndarray] = None
    last_accident_time: Optional[float] = None
    verification_status: str = "unverified"
    vlm_confidence: Optional[float] = None
    vlm_reason: Optional[str] = None
    vlm_severity: Optional[str] = None
    vehicles_involved: Optional[int] = None
    physics_passed: Optional[bool] = None
    verification_frames: List[np.ndarray] = field(default_factory=list)
    verification_frame_paths: List[str] = field(default_factory=list)
    detection_history: List[Any] = field(default_factory=list)
    clip_path: Optional[str] = None
    clip_annotated_path: Optional[str] = None
    impact_offset_seconds: Optional[float] = None
    severity_score: Optional[int] = None
    severity: Optional[str] = None
    severity_breakdown: Optional[Dict[str, Any]] = None
    recommended_units: Optional[List[str]] = None
    temporal_confirmed: Optional[bool] = None
    temporal_prob: Optional[float] = None
    temporal_diagnostics: Optional[Dict[str, Any]] = None


def load_model(weights_path: str) -> YOLO:
    """Loads a YOLO model from the given weights path.

    Args:
        weights_path: Path to the .pt weights file.

    Returns:
        YOLO: Initialized Ultralytics YOLO model.

    Raises:
        FileNotFoundError: If weights_path does not exist on disk.
        RuntimeError: If model initialization fails.
    """
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Model weights file '{weights_path}' was not found.")
    try:
        return YOLO(weights_path)
    except Exception as exc:
        raise RuntimeError(f"Failed to load YOLO model from '{weights_path}': {exc}") from exc


def compute_box_iou(box1: Tuple[float, float, float, float], box2: Tuple[float, float, float, float]) -> float:
    """Computes Intersection over Union (IoU) between two (x1, y1, x2, y2) bounding boxes."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter_w = max(0.0, x2 - x1)
    inter_h = max(0.0, y2 - y1)
    inter_area = inter_w * inter_h

    area1 = max(0.0, box1[2] - box1[0]) * max(0.0, box1[3] - box1[1])
    area2 = max(0.0, box2[2] - box2[0]) * max(0.0, box2[3] - box2[1])
    union_area = area1 + area2 - inter_area
    if union_area <= 0.0:
        return 0.0
    return inter_area / union_area


def compute_intersection_ratio(box_acc: Tuple[float, float, float, float], box_norm: Tuple[float, float, float, float]) -> float:
    """Computes the ratio of intersection area to the area of the accident box."""
    x1 = max(box_acc[0], box_norm[0])
    y1 = max(box_acc[1], box_norm[1])
    x2 = min(box_acc[2], box_norm[2])
    y2 = min(box_acc[3], box_norm[3])

    inter_w = max(0.0, x2 - x1)
    inter_h = max(0.0, y2 - y1)
    inter_area = inter_w * inter_h

    area_acc = max(0.0, box_acc[2] - box_acc[0]) * max(0.0, box_acc[3] - box_acc[1])
    if area_acc <= 0.0:
        return 0.0
    return inter_area / area_acc


class AccidentDetector:
    """Encapsulates YOLO inference and ByteTrack object tracking."""

    def __init__(
        self,
        model_path: Optional[str] = None,
        injury_model_path: Optional[str] = None,
        confidence_threshold: Optional[float] = None,
        accident_confidence_threshold: Optional[float] = None,
        normal_confidence_threshold: Optional[float] = None,
        injury_confidence_threshold: Optional[float] = None,
        device: Optional[str] = None,
        accident_classes: Optional[Set[str]] = None,
        require_overlapping_entities: Optional[bool] = None,
        overlap_iou_threshold: Optional[float] = None,
        nms_iou: Optional[float] = None,
        tracker_track_buffer: Optional[int] = None,
        tracker_match_thresh: Optional[float] = None,
        # ── Resolution enhancement (SAHI tiles + ROI crop re-detection) ────
        enable_tile_inference: bool = False,
        tile_config: Optional["TileConfig"] = None,  # type: ignore[type-arg]
        enable_roi_redetect: bool = False,
        roi_config: Optional["ROIConfig"] = None,    # type: ignore[type-arg]
    ):
        """Initializes primary detection and optional secondary injury models.

        Args:
            model_path: Path to the primary detection model.
            injury_model_path: Optional path to the injury detection model.
            confidence_threshold: Legacy confidence cutoff (falls back to accident threshold).
            accident_confidence_threshold: Confidence threshold for accident classes.
            normal_confidence_threshold: Confidence threshold for normal traffic entities.
            injury_confidence_threshold: Confidence cutoff for injury detections.
            device: Target hardware device ('cuda:0', 'mps', or 'cpu').
            accident_classes: Set of class labels indicating an accident.
            require_overlapping_entities: Whether accident boxes must overlap with road entities.
            overlap_iou_threshold: Minimum overlap ratio to validate accident box.
            nms_iou: Non-maximum suppression IoU threshold.
            tracker_track_buffer: ByteTrack frame buffer for lost tracks.
            tracker_match_thresh: ByteTrack association threshold.
            enable_tile_inference: Enable SAHI-style tiled inference on suspicious frames.
                Applied only when the primary pass already fires a candidate accident
                detection (not on every frame) to avoid FPS regression.
            tile_config: TileConfig controlling tile size, overlap, and inference imgsz.
            enable_roi_redetect: Enable ROI crop re-detection at higher resolution when
                a track pair's centroid distance falls below close_distance_norm or
                either box is classified as small/far (area < small_area_px2).
            roi_config: ROIConfig controlling crop padding, redetect_imgsz, and gates.
        """
        self.model_path = model_path or config.model_path
        self.injury_model_path = injury_model_path or config.injury_model_path

        self.accident_confidence_threshold = (
            accident_confidence_threshold
            if accident_confidence_threshold is not None
            else getattr(config, "accident_confidence_threshold", confidence_threshold or config.confidence_threshold)
        )
        self.normal_confidence_threshold = (
            normal_confidence_threshold
            if normal_confidence_threshold is not None
            else getattr(config, "normal_confidence_threshold", 0.35)
        )
        self.confidence_threshold = self.accident_confidence_threshold

        self.injury_confidence_threshold = (
            injury_confidence_threshold or config.injury_confidence_threshold
        )
        self.device = device or config.device
        self.accident_classes = accident_classes or set(ACCIDENT_CLASSES)

        self.require_overlapping_entities = (
            require_overlapping_entities
            if require_overlapping_entities is not None
            else getattr(config, "require_overlapping_entities", False)
        )
        self.overlap_iou_threshold = (
            overlap_iou_threshold
            if overlap_iou_threshold is not None
            else getattr(config, "overlap_iou_threshold", 0.05)
        )
        self.nms_iou = (
            nms_iou
            if nms_iou is not None
            else getattr(config, "nms_iou", 0.50)
        )
        self.tracker_track_buffer = (
            tracker_track_buffer
            if tracker_track_buffer is not None
            else getattr(config, "tracker_track_buffer", 30)
        )
        self.tracker_match_thresh = (
            tracker_match_thresh
            if tracker_match_thresh is not None
            else getattr(config, "tracker_match_thresh", 0.80)
        )

        # Load primary YOLO model
        self.primary_model = load_model(self.model_path)

        # Load secondary injury model if present, otherwise None
        self.injury_model: Optional[YOLO] = None
        if self.injury_model_path and os.path.exists(self.injury_model_path):
            self.injury_model = load_model(self.injury_model_path)

        # Resolution enhancement modules (disabled by default)
        self._tiler: Optional[Any] = None
        self._roi_redetector: Optional[Any] = None

        if enable_tile_inference and _RESOLUTION_AVAILABLE:
            try:
                from acs.resolution import SuspiciousTileInferencer, TileConfig
                self._tiler = SuspiciousTileInferencer(tile_config)
                logger.info(
                    "SuspiciousTileInferencer enabled (tile=%dpx, overlap=%.0f%%)",
                    (tile_config or TileConfig()).tile_size,
                    (tile_config or TileConfig()).overlap_ratio * 100,
                )
            except ImportError as _err:
                logger.warning("Could not load SuspiciousTileInferencer: %s", _err)

        if enable_roi_redetect and _RESOLUTION_AVAILABLE:
            try:
                from acs.resolution import ROICropReDetector, ROIConfig
                self._roi_redetector = ROICropReDetector(roi_config)
                logger.info(
                    "ROICropReDetector enabled (redetect_imgsz=%d)",
                    (roi_config or ROIConfig()).redetect_imgsz,
                )
            except ImportError as _err:
                logger.warning("Could not load ROICropReDetector: %s", _err)

    def predict_frame(
        self,
        frame: np.ndarray,
        resize_width: int = 1080,
        track: bool = False,
    ) -> DetectionResult:
        """Runs object detection or ByteTrack tracking on a single frame.

        Args:
            frame: Input image frame as a BGR numpy array.
            resize_width: Target width to scale the frame while preserving aspect ratio.
            track: Whether to execute ByteTrack tracking with track IDs (persist=True).

        Returns:
            DetectionResult: Structured detection outputs and annotated frame.
        """
        height, width = frame.shape[:2]
        if width > 0 and height > 0 and resize_width > 0:
            new_height = int(height * (resize_width / width))
            processed_frame = cv2.resize(frame, (resize_width, new_height))
        else:
            processed_frame = frame

        # Lowest threshold to extract both accident and normal entities
        min_conf = min(self.accident_confidence_threshold, self.normal_confidence_threshold)

        # Run primary YOLO tracking or standard prediction
        if track:
            tracker_cfg = "bytetrack.yaml"
            if self.tracker_track_buffer != 30 or abs(self.tracker_match_thresh - 0.8) > 1e-4:
                temp_tracker = os.path.join(config.temp_dir, f"bytetrack_b{self.tracker_track_buffer}_m{int(self.tracker_match_thresh*100)}.yaml")
                if not os.path.exists(temp_tracker):
                    os.makedirs(os.path.dirname(temp_tracker), exist_ok=True)
                    with open(temp_tracker, "w", encoding="utf-8") as f:
                        f.write(f"tracker_type: bytetrack\ntrack_high_thresh: 0.25\ntrack_low_thresh: 0.1\nnew_track_thresh: 0.25\ntrack_buffer: {self.tracker_track_buffer}\nmatch_thresh: {self.tracker_match_thresh}\nfuse_score: True\n")
                tracker_cfg = temp_tracker

            results = self.primary_model.track(
                source=processed_frame,
                persist=True,
                tracker=tracker_cfg,
                conf=min_conf,
                iou=self.nms_iou,
                device=self.device,
                imgsz=640,
                verbose=False,
            )
        else:
            results = self.primary_model.predict(
                source=processed_frame,
                conf=min_conf,
                iou=self.nms_iou,
                device=self.device,
                imgsz=640,
                verbose=False,
            )

        annotated_frame = results[0].plot()
        detected_boxes: List[DetectionBox] = []
        candidate_accident_boxes: List[DetectionBox] = []
        normal_entity_boxes: List[DetectionBox] = []

        boxes = results[0].boxes
        if boxes is not None and len(boxes) > 0:
            for box in boxes:
                class_id = int(box.cls[0].item())
                confidence = float(box.conf[0].item())
                class_id, class_name, incident_type = decode_model_class(class_id, self.primary_model.names)
                is_accident = class_name.lower().strip() in self.accident_classes

                track_id: Optional[int] = None
                if box.id is not None:
                    track_id = int(box.id[0].item())

                xyxy = box.xyxy[0].tolist()
                detection = DetectionBox(
                    bbox=(xyxy[0], xyxy[1], xyxy[2], xyxy[3]),
                    confidence=confidence,
                    class_id=class_id,
                    class_name=class_name,
                    is_accident=is_accident,
                    track_id=track_id,
                    incident_type=incident_type,
                )

                if is_accident:
                    if confidence >= self.accident_confidence_threshold:
                        candidate_accident_boxes.append(detection)
                else:
                    if confidence >= self.normal_confidence_threshold:
                        normal_entity_boxes.append(detection)
                        detected_boxes.append(detection)

        # Validate accident boxes with overlapping entity rule if enabled
        validated_accident_boxes: List[DetectionBox] = []
        if self.require_overlapping_entities:
            for acc_b in candidate_accident_boxes:
                has_overlap = False
                for norm_b in normal_entity_boxes:
                    iou = compute_box_iou(acc_b.bbox, norm_b.bbox)
                    ratio = compute_intersection_ratio(acc_b.bbox, norm_b.bbox)
                    if max(iou, ratio) >= self.overlap_iou_threshold:
                        has_overlap = True
                        break
                if has_overlap:
                    validated_accident_boxes.append(acc_b)
        else:
            validated_accident_boxes = candidate_accident_boxes

        detected_boxes.extend(validated_accident_boxes)
        accident_detected = len(validated_accident_boxes) > 0
        detected_classes = {b.class_name for b in detected_boxes}
        accident_classes_set = {b.class_name for b in validated_accident_boxes}

        # ── Resolution-enhancement pass 1: suspicious-region tiled inference ──
        # Fires only when the primary pass has *any* candidate accident (suspicious).
        # This avoids running tiling on clean frames and keeps FPS budget.
        if self._tiler is not None and len(candidate_accident_boxes) > 0:
            try:
                tile_extras = self._tiler.infer(
                    self.primary_model,
                    processed_frame,
                    device=self.device,
                    accident_classes=self.accident_classes,
                )
                if tile_extras:
                    # Merge tile detections with existing, applying per-class thresholds
                    filtered_tile = [
                        b for b in tile_extras
                        if b.confidence >= (
                            self.accident_confidence_threshold if b.is_accident
                            else self.normal_confidence_threshold
                        )
                    ]
                    from acs.resolution import merge_detections
                    merged = merge_detections(detected_boxes, filtered_tile, self.nms_iou)
                    new_acc = [b for b in merged if b.is_accident and b not in detected_boxes]
                    if new_acc:
                        validated_accident_boxes.extend(new_acc)
                        detected_boxes = merged
                        accident_detected = True
                        detected_classes = {b.class_name for b in detected_boxes}
                        accident_classes_set = {b.class_name for b in detected_boxes if b.is_accident}
                        logger.debug("Tiler added %d new accident detections.", len(new_acc))
            except Exception as _tile_err:  # pragma: no cover
                logger.warning("Tiled inference failed (skipping): %s", _tile_err)

        # ── Resolution-enhancement pass 2: ROI crop re-detection ──
        # Fires only when ≥2 tracked entities exist and any pair is close/small.
        if self._roi_redetector is not None and len(normal_entity_boxes) >= 2:
            fh, fw = processed_frame.shape[:2]
            redetect_added: List[DetectionBox] = []
            try:
                for ai in range(len(normal_entity_boxes)):
                    for bi in range(ai + 1, len(normal_entity_boxes)):
                        da = normal_entity_boxes[ai]
                        db = normal_entity_boxes[bi]
                        if self._roi_redetector.should_redetect(da, db, fw, fh):
                            _confirmed, _refined = self._roi_redetector.redetect(
                                self.primary_model,
                                processed_frame,
                                da,
                                db,
                                device=self.device,
                                accident_classes=self.accident_classes,
                            )
                            if _confirmed:
                                acc_refined = [
                                    b for b in _refined
                                    if b.is_accident
                                    and b.confidence >= self.accident_confidence_threshold
                                ]
                                redetect_added.extend(acc_refined)
            except Exception as _roi_err:  # pragma: no cover
                logger.warning("ROI re-detect failed (skipping): %s", _roi_err)

            if redetect_added:
                from acs.resolution import merge_detections
                detected_boxes = merge_detections(detected_boxes, redetect_added, self.nms_iou)
                validated_accident_boxes.extend(redetect_added)
                accident_detected = True
                detected_classes = {b.class_name for b in detected_boxes}
                accident_classes_set = {b.class_name for b in detected_boxes if b.is_accident}
                logger.debug("ROI redetect confirmed %d accident detections.", len(redetect_added))

        accident_classes = accident_classes_set

        # Run secondary injury model if available and an accident was detected
        if accident_detected and self.injury_model is not None:
            injury_results = self.injury_model.predict(
                source=annotated_frame,
                conf=self.injury_confidence_threshold,
                device=self.device,
                imgsz=640,
                verbose=False,
            )
            annotated_frame = injury_results[0].plot()
            injury_boxes = injury_results[0].boxes
            if injury_boxes is not None and len(injury_boxes) > 0:
                for box in injury_boxes:
                    class_id = int(box.cls[0].item())
                    confidence = float(box.conf[0].item())
                    class_name = self.injury_model.names.get(class_id, f"injury_{class_id}")
                    xyxy = box.xyxy[0].tolist()
                    detected_boxes.append(
                        DetectionBox(
                            bbox=(xyxy[0], xyxy[1], xyxy[2], xyxy[3]),
                            confidence=confidence,
                            class_id=class_id,
                            class_name=class_name,
                            is_accident=True,
                        )
                    )
                    detected_classes.add(class_name)
                    accident_classes.add(class_name)

        return DetectionResult(
            annotated_frame=annotated_frame,
            detections=detected_boxes,
            accident_detected=accident_detected,
            detected_classes=detected_classes,
            accident_classes=accident_classes,
        )

    def track_frame(
        self,
        frame: np.ndarray,
        resize_width: int = 1080,
    ) -> DetectionResult:
        """Runs ByteTrack object tracking on a single frame.

        Args:
            frame: Input image frame as a BGR numpy array.
            resize_width: Target width to scale the frame while preserving aspect ratio.

        Returns:
            DetectionResult: Structured detection outputs with tracking IDs.
        """
        return self.predict_frame(frame=frame, resize_width=resize_width, track=True)

    def track(
        self,
        frame: np.ndarray,
        resize_width: int = 1080,
    ) -> DetectionResult:
        """Alias for track_frame."""
        return self.track_frame(frame=frame, resize_width=resize_width)


class AccidentEventManager:
    """Aggregates per-frame accident detections into discrete, high-confidence incidents.

    Lifecycle:
        1. Accumulates accident detections across consecutive frames.
        2. Starts an incident when N consecutive frames (default 5) contain an accident
           exceeding the confidence threshold.
        3. Tracks and updates the keyframe based on maximum detection confidence.
        4. Ends the incident after M seconds (default 3.0s) without accident detections.
        5. Enforces a cooldown (default 10.0s) before any new incident can start.
        6. On incident end, saves exactly one keyframe and appends one CSV row.
    """

    def __init__(
        self,
        consecutive_frames_threshold: Optional[int] = None,
        timeout_seconds: Optional[float] = None,
        cooldown_seconds: Optional[float] = None,
        confidence_threshold: Optional[float] = None,
        accident_confidence_threshold: Optional[float] = None,
        accident_classes: Optional[Set[str]] = None,
        output_dir: Optional[str] = None,
        reports_dir: Optional[str] = None,
        csv_filename: str = "incidents.csv",
        auto_save: bool = True,
        auto_verify: bool = False,
        save_verification_frames: bool = False,
        clips_dir: Optional[str] = None,
        save_clips: bool = True,
        clip_pre_seconds: Optional[float] = None,
        clip_post_seconds: Optional[float] = None,
        target_fps: float = 30.0,
        temporal_model_enabled: Optional[bool] = None,
        temporal_model_path: Optional[str] = None,
        temporal_threshold: Optional[float] = None,
    ):
        """Initializes the event manager with timing and storage parameters.

        Args:
            consecutive_frames_threshold: Required consecutive accident frames to start (defaults to config).
            timeout_seconds: Inactivity period before incident termination (defaults to config).
            cooldown_seconds: Delay after incident end before new one can start (defaults to config).
            confidence_threshold: Confidence cutoff for an accident detection to count.
            accident_confidence_threshold: Explicit threshold for accident classes (falls back to confidence_threshold).
            accident_classes: Set of class labels treated as accidents.
            output_dir: Directory to store incident keyframes.
            reports_dir: Directory to append the incidents CSV log.
            csv_filename: File name for the master incidents CSV.
            auto_save: Whether to automatically write keyframe and CSV row on incident end.
            auto_verify: Whether to automatically run VLM and physics verification on close.
            save_verification_frames: Whether to save multi-frame verification images to output_dir.
            clips_dir: Directory to store incident H.264 video clips.
            save_clips: Whether to record and save 8s-pre to 10s-post H.264 clips.
            clip_pre_seconds: Duration of footage collected before impact (default: 8.0s).
            clip_post_seconds: Duration of footage collected after impact (default: 10.0s).
            target_fps: Estimated stream framerate for clip encoding.
            temporal_model_enabled: Whether to gate accident confirmation using second-stage temporal model.
            temporal_model_path: Checkpoint file path for temporal gate model (.pt or .joblib).
            temporal_threshold: Probability cutoff for temporal accident confirmation (default 0.50).
        """
        self.consecutive_frames_threshold = (
            consecutive_frames_threshold
            if consecutive_frames_threshold is not None
            else config.consecutive_frames_threshold
        )
        self.timeout_seconds = (
            timeout_seconds
            if timeout_seconds is not None
            else config.timeout_seconds
        )
        self.cooldown_seconds = (
            cooldown_seconds
            if cooldown_seconds is not None
            else config.cooldown_seconds
        )
        if accident_confidence_threshold is not None:
            self.confidence_threshold = accident_confidence_threshold
        elif confidence_threshold is not None:
            self.confidence_threshold = confidence_threshold
        else:
            self.confidence_threshold = config.accident_confidence_threshold
        self.accident_classes = (
            accident_classes
            if accident_classes is not None
            else set(ACCIDENT_CLASSES)
        )
        self.output_dir = output_dir or config.report_images_dir
        self.reports_dir = reports_dir or config.reports_dir
        self.clips_dir = clips_dir or getattr(config, "clips_dir", "report_clips")
        self.save_clips = save_clips
        self.clip_pre_seconds = (
            clip_pre_seconds if clip_pre_seconds is not None else getattr(config, "clip_pre_incident_seconds", 8.0)
        )
        self.clip_post_seconds = (
            clip_post_seconds if clip_post_seconds is not None else getattr(config, "clip_post_incident_seconds", 10.0)
        )
        self.target_fps = target_fps
        self.csv_filename = csv_filename
        self.auto_save = auto_save
        self.auto_verify = auto_verify
        self.save_verification_frames = save_verification_frames

        # Second-Stage Temporal Model Gate
        self.temporal_model_enabled = (
            temporal_model_enabled
            if temporal_model_enabled is not None
            else getattr(config, "temporal_model_enabled", True)
        )
        self.temporal_model_path = (
            temporal_model_path
            or getattr(config, "temporal_model_path", "models/temporal_gate.pt")
        )
        self.temporal_threshold = (
            temporal_threshold
            if temporal_threshold is not None
            else getattr(config, "temporal_threshold", 0.50)
        )

        from acs.temporal import TemporalGate, TrackHistoryBuffer
        self.temporal_history_buffer = TrackHistoryBuffer(
            max_buffer_seconds=4.0,
            fps=self.target_fps,
        )
        self.temporal_gate = TemporalGate(
            model_path=self.temporal_model_path,
            threshold=self.temporal_threshold,
            enabled=self.temporal_model_enabled,
        )
        self.last_temporal_prob: Optional[float] = None
        self.last_temporal_confirmed: Optional[bool] = None
        self.last_temporal_diagnostics: Optional[Dict[str, Any]] = None

        # Clip Recorder with Bounded Rolling Ring Buffer
        from acs.clip_recorder import IncidentClipRecorder
        self.clip_recorder = IncidentClipRecorder(
            rolling_buffer_seconds=getattr(config, "clip_buffer_seconds", 10.0),
            pre_incident_seconds=self.clip_pre_seconds,
            post_incident_seconds=self.clip_post_seconds,
            output_dir=self.clips_dir,
            target_fps=self.target_fps,
        )

        # Internal state
        self.consecutive_count: int = 0
        self.is_incident_active: bool = False
        self.incident_counter: int = 0

        # Current active incident buffer
        self.current_incident_id: Optional[str] = None
        self.current_start_time: Optional[float] = None
        self.current_start_ts: Optional[str] = None
        self.current_class_name: str = ""
        self.current_incident_type: Optional[str] = None
        self.current_max_confidence: float = 0.0
        self.current_keyframe: Optional[np.ndarray] = None
        self.last_accident_time: Optional[float] = None
        self.last_incident_end_time: Optional[float] = None

        # Frame buffer and active history for verification and physics pre-check
        self._frame_buffer: Deque[Tuple[float, np.ndarray, DetectionResult]] = deque(maxlen=30)
        self._active_incident_history: List[Tuple[float, np.ndarray, DetectionResult]] = []
        self._active_peak_idx: int = 0

        # Incident history
        self.incidents: List[AccidentIncident] = []

    def _format_timestamp(self, ts: float) -> str:
        """Converts an epoch float to an ISO-formatted timestamp string."""
        return datetime.datetime.fromtimestamp(
            ts, tz=datetime.timezone.utc
        ).strftime("%Y-%m-%d %H:%M:%S UTC")

    def get_verification_frames(self, num_frames: int = 3) -> List[np.ndarray]:
        """Collects 3 to 5 chronological frames (before impact, at peak, and after) for active incident."""
        frames: List[np.ndarray] = []
        if self._active_incident_history:
            start_t = self.current_start_time if self.current_start_time is not None else 0.0
            pre_candidates = [item[1] for item in self._frame_buffer if item[0] < start_t and item[1] is not None]
            frames.append(pre_candidates[-1] if pre_candidates else self._active_incident_history[0][1])

            peak_idx = min(self._active_peak_idx, len(self._active_incident_history) - 1)
            peak_f = self.current_keyframe if self.current_keyframe is not None else self._active_incident_history[peak_idx][1]
            frames.append(peak_f)

            frames.append(self._active_incident_history[-1][1])
        elif self.current_keyframe is not None:
            frames = [self.current_keyframe, self.current_keyframe, self.current_keyframe]
        return frames

    def update(
        self,
        detection_result: DetectionResult,
        frame: Optional[np.ndarray] = None,
        timestamp: Optional[float] = None,
    ) -> Optional[AccidentIncident]:
        """Processes a new detection result and updates the incident state machine.

        Args:
            detection_result: Output from detector.predict_frame or track_frame.
            frame: Raw BGR frame or annotated frame for keyframe retention.
            timestamp: Optional external timestamp (e.g. video elapsed seconds).
                Defaults to time.time().

        Returns:
            Optional[AccidentIncident]: Completed incident object if an incident
                ended on this frame, otherwise None.
        """
        current_time = timestamp if timestamp is not None else time.time()

        # Filter accident detections in this frame meeting confidence threshold
        accident_boxes: List[DetectionBox] = []
        for box in detection_result.detections:
            is_acc = box.is_accident or (box.class_name.lower().strip() in self.accident_classes)
            if is_acc and box.confidence >= self.confidence_threshold:
                accident_boxes.append(box)

        # Fallback if detections list is empty but accident_detected flag is set
        if not accident_boxes and not detection_result.detections and detection_result.accident_detected:
            acc_names = [
                c for c in detection_result.accident_classes
                if c.lower().strip() in self.accident_classes
            ]
            cls_name = acc_names[0] if acc_names else "accident"
            dummy_box = DetectionBox(
                bbox=(0.0, 0.0, 0.0, 0.0),
                confidence=1.0,
                class_id=0,
                class_name=cls_name,
                is_accident=True,
            )
            accident_boxes.append(dummy_box)

        has_accident = len(accident_boxes) > 0

        # Safe keyframe buffer copy
        if frame is not None and isinstance(frame, np.ndarray) and frame.size > 0:
            frame_copy = frame.copy()
        else:
            frame_copy = np.zeros((480, 640, 3), dtype=np.uint8)

        # Buffer dimensions
        ann_w = frame_copy.shape[1]
        ann_h = frame_copy.shape[0]
        if getattr(detection_result, "annotated_frame", None) is not None:
            ann_h, ann_w = detection_result.annotated_frame.shape[:2]

        # Feed detections to second-stage temporal history buffer
        self.temporal_history_buffer.add_frame_detections(
            frame_idx=int(current_time * self.target_fps),
            timestamp=current_time,
            detections=detection_result.detections,
            frame_width=ann_w,
            frame_height=ann_h,
        )

        # Record to rolling frame buffer
        self._frame_buffer.append((current_time, frame_copy, detection_result))
        ann_frame = (
            detection_result.annotated_frame
            if (detection_result is not None and getattr(detection_result, "annotated_frame", None) is not None)
            else frame_copy
        )
        self.clip_recorder.feed_frame(current_time, raw_frame=frame_copy, annotated_frame=ann_frame)

        # Check cooldown status
        in_cooldown = (
            self.last_incident_end_time is not None
            and (current_time - self.last_incident_end_time < self.cooldown_seconds)
        )

        completed_incident: Optional[AccidentIncident] = None

        if self.is_incident_active:
            self._active_incident_history.append((current_time, frame_copy, detection_result))
            if has_accident:
                self.last_accident_time = current_time
                best_box = max(accident_boxes, key=lambda b: b.confidence)
                if best_box.confidence > self.current_max_confidence:
                    self.current_max_confidence = best_box.confidence
                    self.current_class_name = best_box.class_name
                    self.current_incident_type = best_box.incident_type
                    self.current_keyframe = frame_copy
                    self._active_peak_idx = len(self._active_incident_history) - 1
            else:
                # Check for incident timeout (M seconds without accident detections)
                if (
                    self.last_accident_time is not None
                    and (current_time - self.last_accident_time >= self.timeout_seconds)
                ):
                    completed_incident = self._terminate_active_incident(end_time=current_time)

        else:
            if in_cooldown:
                self.consecutive_count = 0
                self._active_incident_history.clear()
            else:
                if has_accident:
                    self.consecutive_count += 1
                    best_box = max(accident_boxes, key=lambda b: b.confidence)
                    self.last_accident_time = current_time
                    self._active_incident_history.append((current_time, frame_copy, detection_result))

                    # Initialize incident buffer on first consecutive detection
                    if self.consecutive_count == 1:
                        self.current_start_time = current_time
                        self.current_start_ts = self._format_timestamp(current_time)
                        self.current_max_confidence = best_box.confidence
                        self.current_class_name = best_box.class_name
                        self.current_incident_type = best_box.incident_type
                        self.current_keyframe = frame_copy
                        self._active_peak_idx = 0
                    else:
                        if best_box.confidence > self.current_max_confidence:
                            self.current_max_confidence = best_box.confidence
                            self.current_class_name = best_box.class_name
                            self.current_incident_type = best_box.incident_type
                            self.current_keyframe = frame_copy
                            self._active_peak_idx = len(self._active_incident_history) - 1

                    # Trigger incident start upon reaching consecutive frames threshold (N)
                    # Gate in AccidentEventManager: YOLO proposes, the temporal model confirms!
                    if self.consecutive_count >= self.consecutive_frames_threshold:
                        gate_confirmed = True
                        if self.temporal_model_enabled and self.temporal_gate is not None:
                            gate_confirmed, prob, diag = self.temporal_gate.verify(
                                history_buffer=self.temporal_history_buffer,
                                candidate_bbox=best_box.bbox,
                                frame_width=ann_w,
                                frame_height=ann_h,
                                window_seconds=1.5,
                            )
                            self.last_temporal_prob = prob
                            self.last_temporal_confirmed = gate_confirmed
                            self.last_temporal_diagnostics = diag
                        else:
                            self.last_temporal_prob = 1.0
                            self.last_temporal_confirmed = True
                            self.last_temporal_diagnostics = {"gate_disabled": True}

                        if gate_confirmed:
                            self.is_incident_active = True
                            self.incident_counter += 1
                            self.current_incident_id = (
                                f"INC-{int(self.current_start_time)}-{self.incident_counter:03d}"
                            )
                            if self.save_clips:
                                self.clip_recorder.start_incident(
                                    incident_id=self.current_incident_id,
                                    impact_time=self.current_start_time,
                                )
                        else:
                            logger.debug(
                                "Temporal gate rejected accident proposal at threshold N (prob=%.3f < %.2f)",
                                self.last_temporal_prob or 0.0, self.temporal_threshold,
                            )
                            self.consecutive_count = 0
                            self._active_incident_history.clear()
                else:
                    self.consecutive_count = 0
                    self.current_start_time = None
                    self.current_start_ts = None
                    self.current_class_name = ""
                    self.current_incident_type = None
                    self.current_max_confidence = 0.0
                    self.current_keyframe = None
                    self.last_accident_time = None
                    self._active_incident_history.clear()

        return completed_incident

    def end_active_incident(
        self,
        timestamp: Optional[float] = None,
    ) -> Optional[AccidentIncident]:
        """Forces the termination and persistence of any currently active incident.

        Useful at the end of a video stream to finalize an in-progress incident.

        Args:
            timestamp: Optional end timestamp. Defaults to time.time().

        Returns:
            Optional[AccidentIncident]: Completed incident if active, else None.
        """
        if not self.is_incident_active:
            return None
        end_time = timestamp if timestamp is not None else time.time()
        return self._terminate_active_incident(end_time=end_time)

    def _terminate_active_incident(self, end_time: float) -> AccidentIncident:
        """Internal helper to finalize, persist, and record an ended incident."""
        start_time = self.current_start_time if self.current_start_time is not None else end_time
        start_ts = self.current_start_ts or self._format_timestamp(start_time)
        end_ts = self._format_timestamp(end_time)
        incident_id = self.current_incident_id or f"INC-{int(start_time)}-{self.incident_counter:03d}"

        # Resolve image file path
        image_filename = f"{incident_id}_keyframe.jpg"
        image_path = os.path.join(self.output_dir, image_filename)

        # Collect 3 to 5 verification frames: before impact, at peak, and after impact
        v_frames: List[np.ndarray] = []
        if self._active_incident_history:
            pre_candidates = [item[1] for item in self._frame_buffer if item[0] < start_time and item[1] is not None]
            v_frames.append(pre_candidates[-1] if pre_candidates else self._active_incident_history[0][1])

            peak_idx = min(self._active_peak_idx, len(self._active_incident_history) - 1)
            if peak_idx > 1 and len(self._active_incident_history) >= 4:
                mid_before = self._active_incident_history[peak_idx // 2][1]
                v_frames.append(mid_before)

            peak_f = self.current_keyframe if self.current_keyframe is not None else self._active_incident_history[peak_idx][1]
            v_frames.append(peak_f)

            post_count = len(self._active_incident_history) - 1 - peak_idx
            if post_count > 1 and len(self._active_incident_history) >= 4:
                mid_after = self._active_incident_history[peak_idx + (post_count // 2)][1]
                v_frames.append(mid_after)

            v_frames.append(self._active_incident_history[-1][1])
        elif self.current_keyframe is not None:
            v_frames = [self.current_keyframe, self.current_keyframe, self.current_keyframe]

        if len(v_frames) < 3 and self.current_keyframe is not None:
            while len(v_frames) < 3:
                v_frames.append(self.current_keyframe)
        elif len(v_frames) > 5:
            v_frames = [v_frames[0], v_frames[len(v_frames)//4], v_frames[len(v_frames)//2], v_frames[3*len(v_frames)//4], v_frames[-1]]

        history_detections = [item[2] for item in self._active_incident_history]

        # Finalize incident H.264 video clips (8s pre to 10s post impact)
        raw_clip = None
        ann_clip = None
        impact_offset = self.clip_pre_seconds
        if self.save_clips:
            try:
                raw_clip, ann_clip, impact_offset = self.clip_recorder.finalize_incident(
                    incident_id, output_dir=self.clips_dir
                )
            except Exception as clip_err:
                logger.warning("Failed to finalize incident clips for %s: %s", incident_id, clip_err)

        incident = AccidentIncident(
            incident_id=incident_id,
            start_timestamp=start_ts,
            end_timestamp=end_ts,
            class_name=self.current_class_name or "accident",
            incident_type=self.current_incident_type,
            max_confidence=self.current_max_confidence,
            image_path=image_path,
            start_time=start_time,
            end_time=end_time,
            keyframe=self.current_keyframe,
            last_accident_time=self.last_accident_time,
            verification_frames=v_frames,
            detection_history=history_detections,
            clip_path=raw_clip,
            clip_annotated_path=ann_clip,
            impact_offset_seconds=impact_offset,
            temporal_confirmed=self.last_temporal_confirmed,
            temporal_prob=self.last_temporal_prob,
            temporal_diagnostics=self.last_temporal_diagnostics,
        )

        if self.auto_save:
            self._save_incident_artifacts(incident)

        if self.auto_verify:
            try:
                from acs.verifier import verify_incident
                decision = verify_incident(
                    incident,
                    detection_history=history_detections,
                    frames=v_frames,
                )
                incident.verification_status = decision.status
                incident.vlm_confidence = decision.confidence
                incident.vlm_reason = decision.reason
                incident.vlm_severity = decision.severity
                incident.vehicles_involved = decision.vehicles_involved
                incident.physics_passed = decision.physics_passed
            except Exception as v_err:
                logger.warning(f"Auto-verification failed for incident {incident_id}: {v_err}")
                incident.verification_status = "unverified"

        # Multi-factor collision severity scoring and unit routing
        try:
            from acs.severity import evaluate_incident_severity
            sev_result = evaluate_incident_severity(
                incident=incident,
                detection_history=history_detections,
                vlm_severity=incident.vlm_severity,
                vlm_confidence=incident.vlm_confidence,
            )
            incident.severity_score = sev_result.score
            incident.severity = sev_result.level
            incident.severity_breakdown = sev_result.to_dict()
            incident.recommended_units = sev_result.recommended_units
        except Exception as sev_err:
            logger.warning("Severity evaluation failed for incident %s: %s", incident_id, sev_err)
            incident.severity_score = 50
            incident.severity = incident.vlm_severity or "medium"
            incident.recommended_units = ["police", "tow"]

        self.incidents.append(incident)
        self.last_incident_end_time = end_time

        # Reset active tracking state
        self.is_incident_active = False
        self.consecutive_count = 0
        self.current_incident_id = None
        self.current_start_time = None
        self.current_start_ts = None
        self.current_class_name = ""
        self.current_incident_type = None
        self.current_max_confidence = 0.0
        self.current_keyframe = None
        self.last_accident_time = None
        self._active_incident_history.clear()
        self._active_peak_idx = 0

        return incident

    def _save_incident_artifacts(self, incident: AccidentIncident) -> None:
        """Persists the keyframe image, verification frames, and appends exactly one CSV row."""
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.reports_dir, exist_ok=True)

        # 1. Save keyframe image (with optional privacy blur & vault)
        if incident.keyframe is not None:
            if getattr(config, "privacy_blur_enabled", True):
                try:
                    from acs.privacy import protect_keyframe
                    boxes = incident.detection_history[-1].boxes if incident.detection_history and hasattr(incident.detection_history[-1], "boxes") else None
                    out_path, unredacted_path = protect_keyframe(
                        keyframe=incident.keyframe,
                        incident_id=incident.incident_id,
                        output_path=incident.image_path,
                        detected_boxes=boxes,
                        preserve_unblurred=getattr(config, "privacy_preserve_unblurred_original", True),
                        unredacted_dir=getattr(config, "privacy_unredacted_dir", "media/unredacted"),
                    )
                    incident.is_privacy_blurred = True
                    incident.unredacted_keyframe_path = unredacted_path
                except Exception as priv_err:
                    logger.warning("Privacy keyframe protection failed: %s", priv_err)
                    cv2.imwrite(incident.image_path, incident.keyframe)
            else:
                cv2.imwrite(incident.image_path, incident.keyframe)
        else:
            # Fallback placeholder if no keyframe frame was stored
            blank_frame = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.imwrite(incident.image_path, blank_frame)

        # 2. Save verification frames (before, peak, after) if requested
        v_paths = []
        if self.save_verification_frames and incident.verification_frames:
            roles = (
                ["before", "approach", "peak", "aftermath", "rest"]
                if len(incident.verification_frames) == 5
                else ["before", "peak", "after"]
            )
            for idx, vf in enumerate(incident.verification_frames[:5]):
                role = roles[idx] if idx < len(roles) else f"frame_{idx + 1}"
                v_fname = f"{incident.incident_id}_frame_{idx + 1:02d}_{role}.jpg"
                v_path = os.path.join(self.output_dir, v_fname)
                if vf is not None and isinstance(vf, np.ndarray) and vf.size > 0:
                    if getattr(config, "privacy_blur_enabled", True):
                        from acs.privacy import apply_privacy_blur
                        cv2.imwrite(v_path, apply_privacy_blur(vf))
                    else:
                        cv2.imwrite(v_path, vf)
                    v_paths.append(v_path)
            incident.verification_frame_paths = v_paths

        # 3. Append exactly one CSV row
        csv_path = os.path.join(self.reports_dir, self.csv_filename)
        file_exists = os.path.exists(csv_path) and os.path.getsize(csv_path) > 0

        with open(csv_path, "a", encoding="utf-8") as f:
            if not file_exists:
                f.write("incident_id,start_timestamp,end_timestamp,class,max_confidence,image_path\n")
            f.write(
                f"{incident.incident_id},"
                f"{incident.start_timestamp},"
                f"{incident.end_timestamp},"
                f"{incident.class_name},"
                f"{incident.max_confidence:.4f},"
                f"{incident.image_path}\n"
            )

    @property
    def current_incident(self) -> Optional[AccidentIncident]:
        """Returns a snapshot of the currently active in-progress incident, or None."""
        if not self.is_incident_active:
            return None
        start_time = self.current_start_time if self.current_start_time is not None else 0.0
        incident_id = self.current_incident_id or f"INC-{int(start_time)}-{self.incident_counter:03d}"
        image_path = os.path.join(self.output_dir, f"{incident_id}_keyframe.jpg")
        return AccidentIncident(
            incident_id=incident_id,
            start_timestamp=self.current_start_ts or self._format_timestamp(start_time),
            end_timestamp=self._format_timestamp(self.last_accident_time or start_time),
            class_name=self.current_class_name or "accident",
            incident_type=self.current_incident_type,
            max_confidence=self.current_max_confidence,
            image_path=image_path,
            start_time=start_time,
            end_time=self.last_accident_time or start_time,
            keyframe=self.current_keyframe,
            last_accident_time=self.last_accident_time,
        )

    def reset(self) -> None:
        """Resets the state machine and clears all history."""
        self.consecutive_count = 0
        self.is_incident_active = False
        self.current_incident_id = None
        self.current_start_time = None
        self.current_start_ts = None
        self.current_class_name = ""
        self.current_max_confidence = 0.0
        self.current_keyframe = None
        self.last_accident_time = None
        self.last_incident_end_time = None
        self.incident_counter = 0
        self.incidents.clear()
        self._active_incident_history.clear()
        self._active_peak_idx = 0
