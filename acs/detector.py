"""
acs.detector
Provides model loading, ByteTrack tracking inference, structured detections,
and the AccidentEventManager state machine for incident lifecycle tracking.
No Streamlit dependencies are included to ensure testability.
"""
from dataclasses import dataclass, field
import datetime
import logging
import os
import time
from typing import List, Optional, Set, Tuple
import cv2
import numpy as np
from ultralytics import YOLO

from acs.config import config, ACCIDENT_CLASSES

logger = logging.getLogger("acs.detector")


@dataclass
class DetectionBox:
    """Represents a single detected or tracked bounding box and its metadata."""

    bbox: Tuple[float, float, float, float]  # (x1, y1, x2, y2)
    confidence: float
    class_id: int
    class_name: str
    is_accident: bool
    track_id: Optional[int] = None


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
    keyframe: Optional[np.ndarray] = None
    last_accident_time: Optional[float] = None


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
                class_name = self.primary_model.names.get(class_id, f"class_{class_id}")
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
        accident_classes = {b.class_name for b in validated_accident_boxes}

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
        self.csv_filename = csv_filename
        self.auto_save = auto_save

        # Internal state
        self.consecutive_count: int = 0
        self.is_incident_active: bool = False
        self.incident_counter: int = 0

        # Current active incident buffer
        self.current_incident_id: Optional[str] = None
        self.current_start_time: Optional[float] = None
        self.current_start_ts: Optional[str] = None
        self.current_class_name: str = ""
        self.current_max_confidence: float = 0.0
        self.current_keyframe: Optional[np.ndarray] = None
        self.last_accident_time: Optional[float] = None
        self.last_incident_end_time: Optional[float] = None

        # Incident history
        self.incidents: List[AccidentIncident] = []

    def _format_timestamp(self, ts: float) -> str:
        """Converts an epoch float to an ISO-formatted timestamp string."""
        return datetime.datetime.fromtimestamp(
            ts, tz=datetime.timezone.utc
        ).strftime("%Y-%m-%d %H:%M:%S UTC")

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

        # Check cooldown status
        in_cooldown = (
            self.last_incident_end_time is not None
            and (current_time - self.last_incident_end_time < self.cooldown_seconds)
        )

        completed_incident: Optional[AccidentIncident] = None

        if self.is_incident_active:
            if has_accident:
                self.last_accident_time = current_time
                best_box = max(accident_boxes, key=lambda b: b.confidence)
                if best_box.confidence > self.current_max_confidence:
                    self.current_max_confidence = best_box.confidence
                    self.current_class_name = best_box.class_name
                    self.current_keyframe = frame_copy
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
            else:
                if has_accident:
                    self.consecutive_count += 1
                    best_box = max(accident_boxes, key=lambda b: b.confidence)
                    self.last_accident_time = current_time

                    # Initialize incident buffer on first consecutive detection
                    if self.consecutive_count == 1:
                        self.current_start_time = current_time
                        self.current_start_ts = self._format_timestamp(current_time)
                        self.current_max_confidence = best_box.confidence
                        self.current_class_name = best_box.class_name
                        self.current_keyframe = frame_copy
                    else:
                        if best_box.confidence > self.current_max_confidence:
                            self.current_max_confidence = best_box.confidence
                            self.current_class_name = best_box.class_name
                            self.current_keyframe = frame_copy

                    # Trigger incident start upon reaching consecutive frames threshold (N)
                    if self.consecutive_count >= self.consecutive_frames_threshold:
                        self.is_incident_active = True
                        self.incident_counter += 1
                        self.current_incident_id = (
                            f"INC-{int(self.current_start_time)}-{self.incident_counter:03d}"
                        )
                else:
                    self.consecutive_count = 0
                    self.current_start_time = None
                    self.current_start_ts = None
                    self.current_class_name = ""
                    self.current_max_confidence = 0.0
                    self.current_keyframe = None
                    self.last_accident_time = None

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

        incident = AccidentIncident(
            incident_id=incident_id,
            start_timestamp=start_ts,
            end_timestamp=end_ts,
            class_name=self.current_class_name or "accident",
            max_confidence=self.current_max_confidence,
            image_path=image_path,
            start_time=start_time,
            end_time=end_time,
            keyframe=self.current_keyframe,
            last_accident_time=self.last_accident_time,
        )

        if self.auto_save:
            self._save_incident_artifacts(incident)

        self.incidents.append(incident)
        self.last_incident_end_time = end_time

        # Reset active tracking state
        self.is_incident_active = False
        self.consecutive_count = 0
        self.current_incident_id = None
        self.current_start_time = None
        self.current_start_ts = None
        self.current_class_name = ""
        self.current_max_confidence = 0.0
        self.current_keyframe = None
        self.last_accident_time = None

        return incident

    def _save_incident_artifacts(self, incident: AccidentIncident) -> None:
        """Persists the keyframe image and appends exactly one CSV row."""
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.reports_dir, exist_ok=True)

        # 1. Save exactly one keyframe image
        if incident.keyframe is not None:
            cv2.imwrite(incident.image_path, incident.keyframe)
        else:
            # Fallback placeholder if no keyframe frame was stored
            blank_frame = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.imwrite(incident.image_path, blank_frame)

        # 2. Append exactly one CSV row
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
