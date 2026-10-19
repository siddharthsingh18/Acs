"""
acs.verifier
Verification stage between "YOLO flagged an incident" and "alert sent".

Key Responsibilities:
1. Multi-Frame Collection: Collects 3 to 5 chronological frames: before impact, at peak, and after.
2. Cheap Motion/Physics Check: Bounding-box velocity change, vehicle overlap, or stopped vehicles
   after overlap to quickly suppress non-physical false positives without incurring VLM API cost.
3. VLM Verification: Strict structured query to Google Gemini VLM:
   "Is a real traffic collision visible? Answer JSON: {is_accident: bool, confidence: 0-1, reason: str, severity: low|medium|high, vehicles_involved: int}"
4. Tri-State Decision Matrix:
   - auto_confirmed: YOLO and VLM agree -> alert allowed
   - needs_review: YOLO and VLM disagree (or physics check fails) -> alert held for human dispatch review
   - suppressed: both indicate no collision -> logged as suppressed, never alerted
   - unverified: API failure, timeout, or daily budget reached -> falls back to YOLO-only with unverified flag, never crashes
5. Operational Controls: Hard timeouts, exponential backoff retries, per-incident caching, and daily call budget.
"""
from dataclasses import asdict, dataclass
import json
import logging
import math
import os
import threading
import time
from typing import Any, Dict, List, Literal, Optional, Tuple

import numpy as np

from acs.config import ACCIDENT_CLASSES, config
from acs.detector import compute_box_iou, compute_intersection_ratio
from acs.llm import verify_incident_frames_vlm

logger = logging.getLogger("acs.verifier")


@dataclass
class PhysicsCheckResult:
    """Detailed evaluation from the cheap heuristic motion/physics pre-check."""

    passed: bool
    reason: str
    overlap_detected: bool = False
    sudden_velocity_change: bool = False
    stopped_after_overlap: bool = False
    max_iou: float = 0.0
    max_velocity_delta: float = 0.0


@dataclass
class VerificationDecision:
    """Final tri-state decision produced by the verification pipeline."""

    status: Literal["auto_confirmed", "needs_review", "suppressed", "unverified"]
    vlm_called: bool
    physics_passed: bool
    physics_result: Optional[PhysicsCheckResult]
    vlm_result: Optional[Dict[str, Any]]
    reason: str
    alert_allowed: bool
    confidence: float
    severity: Literal["low", "medium", "high"] = "medium"
    vehicles_involved: int = 1
    cached: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Serializes decision into a JSON-safe dictionary."""
        return {
            "status": self.status,
            "vlm_called": self.vlm_called,
            "physics_passed": self.physics_passed,
            "physics_result": asdict(self.physics_result) if self.physics_result else None,
            "vlm_result": self.vlm_result,
            "reason": self.reason,
            "alert_allowed": self.alert_allowed,
            "confidence": self.confidence,
            "severity": self.severity,
            "vehicles_involved": self.vehicles_involved,
            "cached": self.cached,
        }


def evaluate_physics_motion(
    detection_history: Optional[List[Any]],
    min_iou_overlap: float = 0.02,
    min_intersection_ratio: float = 0.05,
    min_speed_px: float = 8.0,
    sudden_decel_ratio: float = 0.40,
    sudden_delta_v: float = 18.0,
) -> PhysicsCheckResult:
    """Executes a cheap motion and physics heuristic check over incident frame history.

    Checks:
    1. Overlap between vehicle bounding boxes (IoU > threshold or intersection ratio).
    2. Sudden bounding-box velocity change (hard deceleration / abrupt stop on impact).
    3. Stopped vehicles after overlap (vehicles that collided and remain stationary).

    Args:
        detection_history: Sequence of per-frame records. Each element can be:
            - Tuple: (timestamp, detections)
            - Object with `.timestamp` and `.detections` attributes
            - Dict with 'timestamp' and 'detections' / 'boxes'
        min_iou_overlap: IoU threshold to consider boxes overlapping.
        min_intersection_ratio: Intersection area ratio threshold.
        min_speed_px: Minimum velocity to establish moving vehicle baseline.
        sudden_decel_ratio: Velocity drop percentage considered sudden deceleration.
        sudden_delta_v: Absolute velocity change magnitude in pixels/frame.

    Returns:
        PhysicsCheckResult: Assessment with passed flag and granular metrics.
    """
    if not detection_history or len(detection_history) < 2:
        return PhysicsCheckResult(
            passed=True,
            reason="Insufficient temporal history for motion check; defaulting to pass.",
            overlap_detected=False,
            sudden_velocity_change=False,
            stopped_after_overlap=False,
        )

    overlap_detected = False
    sudden_velocity_change = False
    stopped_after_overlap = False
    max_iou = 0.0
    max_velocity_delta = 0.0
    overlap_frame_idx: Optional[int] = None

    # Track box centroids across frames: dict of track_id or nearest box centroid
    # Key: track_id (or synthetic box index), Value: List of (frame_idx, (cx, cy))
    trajectory: Dict[int, List[Tuple[int, Tuple[float, float]]]] = {}

    for f_idx, record in enumerate(detection_history):
        # Normalize record format
        boxes: List[Any] = []
        if isinstance(record, (tuple, list)) and len(record) >= 2:
            raw_boxes = record[1]
        elif isinstance(record, dict):
            raw_boxes = record.get("detections", record.get("boxes", []))
        elif hasattr(record, "detections"):
            raw_boxes = record.detections
        else:
            raw_boxes = []

        if hasattr(raw_boxes, "detections"):
            raw_boxes = raw_boxes.detections

        # Extract normalized boxes
        normalized_boxes = []
        for b in raw_boxes or []:
            if hasattr(b, "bbox"):
                bbox = b.bbox
                cls_name = getattr(b, "class_name", "")
                trk_id = getattr(b, "track_id", -1)
                is_acc = getattr(b, "is_accident", False)
            elif isinstance(b, dict):
                bbox = b.get("bbox", (0, 0, 0, 0))
                cls_name = b.get("class_name", "")
                trk_id = b.get("track_id", -1)
                is_acc = b.get("is_accident", False)
            else:
                continue
            normalized_boxes.append({
                "bbox": bbox,
                "class_name": str(cls_name).lower(),
                "track_id": trk_id,
                "is_accident": is_acc or (str(cls_name).lower() in ACCIDENT_CLASSES),
            })

        # 1. Check Pairwise Overlap in this frame
        n_boxes = len(normalized_boxes)
        for i in range(n_boxes):
            box_i = normalized_boxes[i]["bbox"]
            for j in range(i + 1, n_boxes):
                box_j = normalized_boxes[j]["bbox"]
                iou = compute_box_iou(box_i, box_j)
                ratio_ij = compute_intersection_ratio(box_i, box_j)
                ratio_ji = compute_intersection_ratio(box_j, box_i)

                max_inter = max(ratio_ij, ratio_ji)
                if iou > max_iou:
                    max_iou = iou

                if iou >= min_iou_overlap or max_inter >= min_intersection_ratio:
                    overlap_detected = True
                    if overlap_frame_idx is None:
                        overlap_frame_idx = f_idx

        # 2. Centroid tracking for velocity change
        for b_idx, b_info in enumerate(normalized_boxes):
            x1, y1, x2, y2 = b_info["bbox"]
            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0
            tid = b_info["track_id"] if (b_info["track_id"] is not None and b_info["track_id"] > 0) else (1000 + b_idx)
            trajectory.setdefault(tid, []).append((f_idx, (cx, cy)))

    # Compute Velocity Changes per track
    for tid, pts in trajectory.items():
        if len(pts) < 2:
            continue
        speeds: List[float] = []
        for k in range(1, len(pts)):
            prev_f, (x1, y1) = pts[k - 1]
            curr_f, (x2, y2) = pts[k]
            df = max(1, curr_f - prev_f)
            dist = math.hypot(x2 - x1, y2 - y1)
            v = dist / df
            speeds.append(v)

        for k in range(1, len(speeds)):
            s_prev = speeds[k - 1]
            s_curr = speeds[k]
            delta_v = abs(s_curr - s_prev)
            if delta_v > max_velocity_delta:
                max_velocity_delta = delta_v

            # Check sudden deceleration or velocity jump
            if s_prev >= min_speed_px and (s_prev - s_curr) / s_prev >= sudden_decel_ratio:
                sudden_velocity_change = True
            elif delta_v >= sudden_delta_v:
                sudden_velocity_change = True

    # 3. Check for stopped vehicles after overlap
    if overlap_detected and overlap_frame_idx is not None:
        for tid, pts in trajectory.items():
            post_overlap_pts = [(f, pt) for f, pt in pts if f >= overlap_frame_idx]
            if len(post_overlap_pts) >= 2:
                recent_moves = [
                    math.hypot(post_overlap_pts[i][1][0] - post_overlap_pts[i - 1][1][0],
                               post_overlap_pts[i][1][1] - post_overlap_pts[i - 1][1][1])
                    for i in range(1, len(post_overlap_pts))
                ]
                avg_post_move = sum(recent_moves) / len(recent_moves)
                if avg_post_move < 4.0:  # Less than 4 px/frame motion after collision
                    stopped_after_overlap = True

    passed = bool(overlap_detected or sudden_velocity_change or stopped_after_overlap)

    if passed:
        reasons = []
        if overlap_detected:
            reasons.append(f"box overlap detected (max IoU {max_iou:.2f})")
        if sudden_velocity_change:
            reasons.append(f"sudden velocity deceleration/delta ({max_velocity_delta:.1f} px/f)")
        if stopped_after_overlap:
            reasons.append("stopped stationary vehicles post-overlap")
        reason_str = f"Motion/physics check passed: {', '.join(reasons)}."
    else:
        reason_str = (
            "Motion/physics check failed: no bounding-box overlap, sudden velocity change, "
            "or post-collision stopped vehicles observed."
        )

    return PhysicsCheckResult(
        passed=passed,
        reason=reason_str,
        overlap_detected=overlap_detected,
        sudden_velocity_change=sudden_velocity_change,
        stopped_after_overlap=stopped_after_overlap,
        max_iou=round(max_iou, 3),
        max_velocity_delta=round(max_velocity_delta, 1),
    )


class VLMBudgetManager:
    """Tracks and enforces daily API call limits per UTC day."""

    def __init__(self, budget_file: Optional[str] = None, daily_budget: Optional[int] = None):
        self.budget_file = budget_file or getattr(config, "vlm_budget_file", "reports/vlm_budget.json")
        self.daily_budget = daily_budget if daily_budget is not None else getattr(config, "vlm_daily_budget", 200)
        self._lock = threading.Lock()

    def _today_utc(self) -> str:
        return time.strftime("%Y-%m-%d", time.gmtime())

    def _load_data(self) -> Dict[str, Any]:
        if os.path.exists(self.budget_file):
            try:
                with open(self.budget_file, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                pass
        return {"date": self._today_utc(), "calls_today": 0, "daily_budget": self.daily_budget}

    def _save_data(self, data: Dict[str, Any]) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(self.budget_file)), exist_ok=True)
        tmp_path = f"{self.budget_file}.tmp"
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        os.replace(tmp_path, self.budget_file)

    def can_call(self) -> bool:
        """Returns True if the current daily budget allows another VLM call."""
        with self._lock:
            data = self._load_data()
            today = self._today_utc()
            if data.get("date") != today:
                return True
            return int(data.get("calls_today", 0)) < self.daily_budget

    def record_call(self) -> int:
        """Records an API call and returns the updated count for today."""
        with self._lock:
            data = self._load_data()
            today = self._today_utc()
            if data.get("date") != today:
                data["date"] = today
                data["calls_today"] = 1
            else:
                data["calls_today"] = int(data.get("calls_today", 0)) + 1
            data["daily_budget"] = self.daily_budget
            data["last_updated"] = time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())
            self._save_data(data)
            return data["calls_today"]

    def get_stats(self) -> Dict[str, Any]:
        """Returns current daily call stats."""
        with self._lock:
            data = self._load_data()
            today = self._today_utc()
            calls = int(data.get("calls_today", 0)) if data.get("date") == today else 0
            return {
                "date": today,
                "calls_today": calls,
                "daily_budget": self.daily_budget,
                "remaining": max(0, self.daily_budget - calls),
            }


class IncidentVerificationCache:
    """Thread-safe cache for incident verification decisions."""

    def __init__(self, cache_file: str = "reports/verification_cache.json"):
        self.cache_file = cache_file
        self._memory_cache: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.Lock()
        self._load()

    def _load(self) -> None:
        if os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, "r", encoding="utf-8") as f:
                    self._memory_cache = json.load(f)
            except Exception:
                self._memory_cache = {}

    def get(self, incident_id: str) -> Optional[Dict[str, Any]]:
        """Retrieves cached decision for incident ID if present."""
        with self._lock:
            return self._memory_cache.get(incident_id)

    def set(self, incident_id: str, decision_dict: Dict[str, Any]) -> None:
        """Stores decision in cache and persists to disk."""
        with self._lock:
            self._memory_cache[incident_id] = decision_dict
            try:
                os.makedirs(os.path.dirname(os.path.abspath(self.cache_file)), exist_ok=True)
                with open(f"{self.cache_file}.tmp", "w", encoding="utf-8") as f:
                    json.dump(self._memory_cache, f, indent=2)
                os.replace(f"{self.cache_file}.tmp", self.cache_file)
            except Exception as exc:
                logger.warning("Failed to persist verification cache: %s", exc)

    def clear(self) -> None:
        with self._lock:
            self._memory_cache.clear()
            if os.path.exists(self.cache_file):
                try:
                    os.remove(self.cache_file)
                except OSError:
                    pass


# Global singleton manager instances
global_budget_manager = VLMBudgetManager()
global_verification_cache = IncidentVerificationCache()


def collect_verification_frames(
    incident: Any,
    max_frames: int = 5,
) -> List[Any]:
    """Collects 3 to 5 chronological keyframes: before impact, at peak, and after impact.

    Args:
        incident: AccidentIncident object or similar record with keyframe and frame paths.
        max_frames: Target frame count (3 to 5).

    Returns:
        List[Any]: Ordered list of 3-5 images (file paths or numpy arrays).
    """
    frames: List[Any] = []

    # 1. Direct verification frames on incident object if available
    direct_frames = getattr(incident, "verification_frames", None)
    if direct_frames and isinstance(direct_frames, list) and len(direct_frames) >= 2:
        return direct_frames[:max_frames]

    direct_paths = getattr(incident, "verification_frame_paths", None)
    if direct_paths and isinstance(direct_paths, list) and len(direct_paths) >= 2:
        valid_paths = [p for p in direct_paths if os.path.exists(p)]
        if valid_paths:
            return valid_paths[:max_frames]

    # 2. Check surrounding frames on disk around the primary keyframe
    img_path = getattr(incident, "image_path", None)
    if img_path and os.path.exists(img_path):
        from acs.llm import find_surrounding_frames
        discovered = find_surrounding_frames(img_path, max_before=2, max_after=2)
        if len(discovered) >= 3:
            return discovered[:max_frames]
        frames.extend(discovered)

    # 3. Peak keyframe numpy array fallback
    kf = getattr(incident, "keyframe", None)
    if kf is not None and isinstance(kf, np.ndarray) and kf.size > 0:
        if not frames:
            # Build 3 sequential representations (before, peak, after)
            frames = [kf, kf, kf]

    return frames or [img_path] if img_path else []


def verify_incident(
    incident: Any,
    detection_history: Optional[List[Any]] = None,
    frames: Optional[List[Any]] = None,
    budget_manager: Optional[VLMBudgetManager] = None,
    cache: Optional[IncidentVerificationCache] = None,
    force_vlm: bool = False,
    skip_cache: bool = False,
    api_key: Optional[str] = None,
    model_name: Optional[str] = None,
) -> VerificationDecision:
    """Executes the full automated verification stage on an incident.

    Tri-State Decision Workflow:
    1. Check Cache: Return cached result for identical incident_id unless skip_cache=True.
    2. Cheap Motion/Physics Pre-Check: Skip VLM if physics heuristic clearly fails (saving cost).
    3. Daily Budget & Timeout Check: Gracefully fall back to YOLO-only with "unverified" flag.
    4. Multi-Frame VLM Query: Strict JSON format to Google Gemini via acs.llm.
    5. Tri-State Decision:
       - Auto-confirmed: YOLO and VLM agree -> alert allowed
       - Needs review: YOLO and VLM disagree (or physics check fails) -> alert held for human review
       - Suppressed: both reject -> logged as suppressed, never alerted
       - Unverified: API failure, timeout, or budget exceeded -> unverified YOLO alert allowed

    Args:
        incident: AccidentIncident or dict with incident metadata.
        detection_history: Sequence of detection boxes over incident lifetime.
        frames: Optional explicit list of 3-5 images (before, peak, after).
        budget_manager: Budget manager instance override.
        cache: Cache instance override.
        force_vlm: Skip physics pre-check and force VLM call.
        skip_cache: Bypass reading from cache.
        api_key: Optional Gemini API key override.
        model_name: Optional Gemini model override.

    Returns:
        VerificationDecision: Complete structured verification outcome.
    """
    bm = budget_manager or global_budget_manager
    vc = cache or global_verification_cache

    inc_id = getattr(incident, "incident_id", "INC-UNKNOWN")
    cls_name = getattr(incident, "class_name", "accident")
    yolo_conf = float(getattr(incident, "max_confidence", 0.70))

    # 1. Check cache
    if not skip_cache:
        cached_entry = vc.get(inc_id)
        if cached_entry:
            logger.info("Incident %s returned from verification cache: %s", inc_id, cached_entry.get("status"))
            p_res = None
            if cached_entry.get("physics_result"):
                p_res = PhysicsCheckResult(**cached_entry["physics_result"])
            return VerificationDecision(
                status=cached_entry["status"],
                vlm_called=cached_entry.get("vlm_called", False),
                physics_passed=cached_entry.get("physics_passed", True),
                physics_result=p_res,
                vlm_result=cached_entry.get("vlm_result"),
                reason=cached_entry.get("reason", "Cached verification decision"),
                alert_allowed=cached_entry.get("alert_allowed", False),
                confidence=float(cached_entry.get("confidence", yolo_conf)),
                severity=cached_entry.get("severity", "medium"),
                vehicles_involved=int(cached_entry.get("vehicles_involved", 1)),
                cached=True,
            )

    # 2. Extract or infer detection history
    history = detection_history or getattr(incident, "detection_history", None)

    # 3. Cheap Motion / Physics Pre-Check
    physics_enabled = getattr(config, "physics_check_enabled", True)
    physics_result = None

    if physics_enabled and not force_vlm:
        physics_result = evaluate_physics_motion(history)
        if not physics_result.passed:
            # Physics check clearly failed: SKIP VLM call to save cost!
            logger.info(
                "Incident %s failed cheap physics check (%s). Skipping VLM call.",
                inc_id,
                physics_result.reason,
            )
            # Decision logic: YOLO flagged but physics says no collision -> DISAGREE -> needs_review
            # Or if YOLO confidence is low -> suppressed
            is_suppressed = yolo_conf < getattr(config, "normal_confidence_threshold", 0.35)
            decision_status: Literal["needs_review", "suppressed"] = (
                "suppressed" if is_suppressed else "needs_review"
            )

            decision = VerificationDecision(
                status=decision_status,
                vlm_called=False,
                physics_passed=False,
                physics_result=physics_result,
                vlm_result=None,
                reason=f"Physics check failed: {physics_result.reason} (VLM skipped to save cost)",
                alert_allowed=False,  # Never alerted
                confidence=0.0,
                severity="low",
                vehicles_involved=0,
            )
            vc.set(inc_id, decision.to_dict())
            return decision

    # If physics passed or disabled:
    physics_passed_flag = physics_result.passed if physics_result else True

    # Check if VLM verification is enabled
    vlm_enabled = getattr(config, "vlm_verification_enabled", True)
    if not vlm_enabled:
        # VLM verification disabled in config -> default to unverified YOLO-only
        decision = VerificationDecision(
            status="unverified",
            vlm_called=False,
            physics_passed=physics_passed_flag,
            physics_result=physics_result,
            vlm_result=None,
            reason="VLM verification is disabled in configuration; YOLO unverified pass-through",
            alert_allowed=True,
            confidence=yolo_conf,
            severity="medium",
            vehicles_involved=2,
        )
        vc.set(inc_id, decision.to_dict())
        return decision

    # 4. Check Daily Budget
    if not bm.can_call() and not force_vlm:
        logger.warning(
            "Daily VLM call budget reached (%d calls). Marking incident %s as 'unverified'.",
            bm.daily_budget,
            inc_id,
        )
        decision = VerificationDecision(
            status="unverified",
            vlm_called=False,
            physics_passed=physics_passed_flag,
            physics_result=physics_result,
            vlm_result=None,
            reason=f"Daily VLM call budget reached ({bm.daily_budget}); falling back to unverified YOLO detection",
            alert_allowed=True,  # Allowed with unverified flag
            confidence=yolo_conf,
            severity="medium",
            vehicles_involved=2,
        )
        vc.set(inc_id, decision.to_dict())
        return decision

    # 5. Collect 3 to 5 frames
    collected_frames = frames or collect_verification_frames(incident, max_frames=5)

    # 6. Call Google Gemini VLM via acs.llm
    bm.record_call()
    vlm_output = verify_incident_frames_vlm(
        frames=collected_frames,
        detections=cls_name,
        api_key=api_key,
        model_name=model_name,
    )

    # 7. Apply Tri-State Decision Matrix
    # Case A: VLM API Failure / Fallback
    if vlm_output.get("verification_status") == "unverified":
        logger.warning(
            "VLM API verification returned unverified fallback for incident %s: %s",
            inc_id,
            vlm_output.get("reason"),
        )
        decision = VerificationDecision(
            status="unverified",
            vlm_called=True,
            physics_passed=physics_passed_flag,
            physics_result=physics_result,
            vlm_result=vlm_output,
            reason=f"VLM verification unverified: {vlm_output.get('reason')}",
            alert_allowed=True,  # Fall back to YOLO-only with unverified flag
            confidence=yolo_conf,
            severity="medium",
            vehicles_involved=2,
        )
        vc.set(inc_id, decision.to_dict())
        return decision

    vlm_is_accident = bool(vlm_output.get("is_accident", False))
    vlm_conf = float(vlm_output.get("confidence", 0.0))
    vlm_reason = str(vlm_output.get("reason", "VLM analysis complete"))
    vlm_severity = str(vlm_output.get("severity", "medium")).lower()
    if vlm_severity not in ("low", "medium", "high"):
        vlm_severity = "medium"
    vehicles_count = int(vlm_output.get("vehicles_involved", 1))

    conf_thresh = getattr(config, "vlm_confidence_threshold", 0.50)
    yolo_says_accident = cls_name.lower().strip() in ACCIDENT_CLASSES or yolo_conf >= getattr(
        config, "accident_confidence_threshold", 0.55
    )

    # Case B: Auto-Confirmed (YOLO and VLM AGREE on accident)
    if yolo_says_accident and vlm_is_accident and vlm_conf >= conf_thresh:
        decision = VerificationDecision(
            status="auto_confirmed",
            vlm_called=True,
            physics_passed=physics_passed_flag,
            physics_result=physics_result,
            vlm_result=vlm_output,
            reason=f"Auto-confirmed: YOLO and VLM agree. VLM: {vlm_reason} (conf {vlm_conf:.2f})",
            alert_allowed=True,
            confidence=max(yolo_conf, vlm_conf),
            severity=vlm_severity,  # type: ignore
            vehicles_involved=vehicles_count,
        )

    # Case C: Suppressed (Both say NO accident)
    elif not yolo_says_accident and not vlm_is_accident:
        decision = VerificationDecision(
            status="suppressed",
            vlm_called=True,
            physics_passed=physics_passed_flag,
            physics_result=physics_result,
            vlm_result=vlm_output,
            reason=f"Suppressed: Both YOLO and VLM agree no collision. VLM: {vlm_reason}",
            alert_allowed=False,  # Kept for review, never alerted
            confidence=vlm_conf,
            severity=vlm_severity,  # type: ignore
            vehicles_involved=vehicles_count,
        )

    # Case D: Disagreement -> Needs Human Review
    else:
        # Either YOLO said accident and VLM said no, or vice versa
        decision = VerificationDecision(
            status="needs_review",
            vlm_called=True,
            physics_passed=physics_passed_flag,
            physics_result=physics_result,
            vlm_result=vlm_output,
            reason=(
                f"Needs Human Review (YOLO & VLM disagree): YOLO={yolo_says_accident} "
                f"vs VLM={vlm_is_accident} (conf {vlm_conf:.2f}). Reason: {vlm_reason}"
            ),
            alert_allowed=False,  # Held for review, alert NOT sent
            confidence=vlm_conf,
            severity=vlm_severity,  # type: ignore
            vehicles_involved=vehicles_count,
        )

    vc.set(inc_id, decision.to_dict())
    return decision
