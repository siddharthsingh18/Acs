"""
acs.severity
Multi-factor collision severity scoring engine (0 to 100), severity level mapping (low/medium/high),
automated emergency unit routing (EMS, police, fire, tow), and transparent explanation breakdown.

Core Capabilities:
1. Feature Extraction:
   - Scales and vulnerable road users (pedestrians, motorcycles, bicycles, heavy vehicles).
   - Pre-impact closing speed kinematics (relative pixels/sec, normalized by camera calibration ppm).
   - Post-impact scene state (disabled stationary vehicles, people on ground, blocked travel lanes).
   - Multimodal VLM visual triage assessment (Gemini verification opinion and confidence).
2. Pluggable Scorer Architecture:
   - BaseSeverityScorer (ABC): standard protocol for pluggable replacement by a trained ML model.
   - RuleBasedSeverityScorer: transparent calibrated heuristic with clear sub-factor points.
   - MLSeverityScorer: supports scikit-learn / XGBoost / PyTorch models or custom estimators with
     fallback to rule-based explanation.
3. Configurable Thresholds:
   - Stored in AppConfig (severity_low_max, severity_medium_max).
4. Dynamic Emergency Unit Dispatch:
   - Routes recommended emergency units (EMS, police, fire, tow) based on score and hazard rules.
"""
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
import logging
import math
import os
from typing import Any, Callable, Dict, List, Literal, Optional, Sequence, Set, Tuple, Union

import numpy as np

from acs.config import ACCIDENT_CLASSES, config

logger = logging.getLogger("acs.severity")


@dataclass
class CameraCalibration:
    """Camera calibration metrics and spatial parameters for physical metric normalization."""

    pixels_per_meter: Optional[float] = None  # Scale factor to convert pixels to real-world meters
    fps: float = 30.0  # Frame rate of camera sensor
    lane_bounds: Optional[List[Tuple[float, float, float, float]]] = None  # Optional lane bounding boxes (x1, y1, x2, y2)


@dataclass
class SeverityFeatures:
    """Standardized feature vector capturing collision dynamics, entities, and scene aftermath."""

    total_entities: int = 1
    vehicles_count: int = 1
    pedestrians_count: int = 0
    motorcycles_count: int = 0
    bicycles_count: int = 0
    heavy_vehicles_count: int = 0

    closing_speed_px_per_sec: float = 0.0
    closing_speed_mps: Optional[float] = None

    stationary_vehicles_post_impact: int = 0
    people_on_ground: bool = False
    blocked_lanes: bool = False

    vlm_severity: Optional[str] = None  # "low", "medium", "high"
    vlm_confidence: Optional[float] = None

    class_name: str = "accident"
    peak_confidence: float = 0.5
    duration_seconds: float = 0.0

    def to_vector(self) -> List[float]:
        """Serializes features into a numerical feature vector suitable for machine learning estimators."""
        vlm_map = {"low": 0.2, "medium": 0.5, "high": 0.9}
        vlm_num = vlm_map.get(str(self.vlm_severity).lower(), 0.5)
        return [
            float(self.total_entities),
            float(self.vehicles_count),
            float(self.pedestrians_count),
            float(self.motorcycles_count),
            float(self.bicycles_count),
            float(self.heavy_vehicles_count),
            float(self.closing_speed_px_per_sec),
            float(self.closing_speed_mps or 0.0),
            float(self.stationary_vehicles_post_impact),
            1.0 if self.people_on_ground else 0.0,
            1.0 if self.blocked_lanes else 0.0,
            vlm_num,
            float(self.vlm_confidence if self.vlm_confidence is not None else 0.5),
            float(self.peak_confidence),
            float(self.duration_seconds),
        ]


@dataclass
class ScoreComponent:
    """Individual transparent breakdown component explaining points contributed to total score."""

    key: str
    label: str
    score: float
    max_score: float
    weight_pct: float
    explanation: str
    metrics: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SeverityResult:
    """Comprehensive output produced by the severity evaluation engine."""

    score: int  # Overall score strictly bounded between 0 and 100
    level: Literal["low", "medium", "high"]
    recommended_units: List[str]  # e.g. ["EMS", "police", "fire", "tow"]
    components: List[ScoreComponent]
    factors: List[str]  # High-level explanatory bullets for UI display
    scorer_version: str = "rule_based_v1"

    def to_dict(self) -> Dict[str, Any]:
        """Serializes severity evaluation into a JSON-compatible dictionary."""
        return {
            "score": self.score,
            "level": self.level,
            "recommended_units": self.recommended_units,
            "components": [
                {
                    "key": c.key,
                    "label": c.label,
                    "score": round(c.score, 1),
                    "max_score": round(c.max_score, 1),
                    "weight_pct": round(c.weight_pct, 1),
                    "explanation": c.explanation,
                    "metrics": c.metrics,
                }
                for c in self.components
            ],
            "factors": self.factors,
            "scorer_version": self.scorer_version,
        }


class BaseSeverityScorer(ABC):
    """Abstract base class establishing the contract for all collision severity scorers."""

    @abstractmethod
    def score(
        self,
        features: SeverityFeatures,
        low_max: int = 35,
        medium_max: int = 70,
    ) -> SeverityResult:
        """Evaluates severity score (0-100), level classification, units, and breakdown."""
        pass


class RuleBasedSeverityScorer(BaseSeverityScorer):
    """Transparent heuristic multi-factor severity scorer.

    Score Allocation:
    1. Vulnerable Road Users & Scale: up to 35 points
    2. Pre-Impact Closing Speed: up to 25 points
    3. Post-Impact Scene State: up to 20 points
    4. VLM Multimodal Assessment: up to 20 points
    Total Capacity: 100 points
    """

    def __init__(self, version: str = "rule_based_v1"):
        self.version = version

    def score(
        self,
        features: SeverityFeatures,
        low_max: int = 35,
        medium_max: int = 70,
    ) -> SeverityResult:
        components: List[ScoreComponent] = []
        factors: List[str] = []

        # -------------------------------------------------------------
        # 1. Vulnerable Road Users & Scale (Capacity: 35 points)
        # -------------------------------------------------------------
        vuln_pts = 0.0
        vuln_details: List[str] = []

        # Base entity count
        total_v = max(1, features.vehicles_count)
        if total_v >= 3:
            vuln_pts += 15.0
            vuln_details.append(f"{total_v} vehicles involved (+15)")
        elif total_v == 2:
            vuln_pts += 10.0
            vuln_details.append("2 vehicles involved (+10)")
        else:
            vuln_pts += 5.0
            vuln_details.append("1 vehicle involved (+5)")

        # Vulnerable road users multipliers
        if features.pedestrians_count > 0:
            vuln_pts += 20.0
            vuln_details.append(f"{features.pedestrians_count} pedestrian(s) involved (+20)")
        if features.motorcycles_count > 0:
            vuln_pts += 15.0
            vuln_details.append(f"{features.motorcycles_count} motorcycle(s) involved (+15)")
        if features.bicycles_count > 0:
            vuln_pts += 12.0
            vuln_details.append(f"{features.bicycles_count} bicycle(s) involved (+12)")
        if features.heavy_vehicles_count > 0:
            vuln_pts += 8.0
            vuln_details.append(f"{features.heavy_vehicles_count} heavy truck/bus (+8)")

        vuln_score = min(35.0, max(0.0, vuln_pts))
        vuln_explanation = "; ".join(vuln_details) if vuln_details else "Standard traffic scale"
        components.append(
            ScoreComponent(
                key="vulnerable_users",
                label="Entities & Vulnerable Road Users",
                score=vuln_score,
                max_score=35.0,
                weight_pct=35.0,
                explanation=vuln_explanation,
                metrics={
                    "total_entities": features.total_entities,
                    "vehicles": features.vehicles_count,
                    "pedestrians": features.pedestrians_count,
                    "motorcycles": features.motorcycles_count,
                    "bicycles": features.bicycles_count,
                    "heavy_vehicles": features.heavy_vehicles_count,
                },
            )
        )
        if features.pedestrians_count > 0 or features.motorcycles_count > 0:
            factors.append(
                f"Vulnerable road users involved: {features.pedestrians_count} pedestrian(s), {features.motorcycles_count} motorcycle(s)."
            )

        # -------------------------------------------------------------
        # 2. Pre-Impact Closing Speed (Capacity: 25 points)
        # -------------------------------------------------------------
        speed_pts = 0.0
        speed_explanation = ""

        if features.closing_speed_mps is not None:
            mps = features.closing_speed_mps
            kmh = mps * 3.6
            if mps >= 18.0:  # > 65 km/h
                speed_pts = 25.0
                speed_explanation = f"Severe highway closing speed: {mps:.1f} m/s (~{kmh:.0f} km/h)"
            elif mps >= 10.0:  # 36 - 65 km/h
                speed_pts = 18.0
                speed_explanation = f"Moderate arterial closing speed: {mps:.1f} m/s (~{kmh:.0f} km/h)"
            elif mps >= 4.0:  # 15 - 36 km/h
                speed_pts = 12.0
                speed_explanation = f"Low urban closing speed: {mps:.1f} m/s (~{kmh:.0f} km/h)"
            else:
                speed_pts = 5.0
                speed_explanation = f"Creeping/parking closing speed: {mps:.1f} m/s (~{kmh:.0f} km/h)"
            speed_explanation += " (calibrated ppm)"
        else:
            px_s = features.closing_speed_px_per_sec
            if px_s >= 600.0:
                speed_pts = 25.0
                speed_explanation = f"Very high track closing rate: {px_s:.0f} px/s"
            elif px_s >= 350.0:
                speed_pts = 18.0
                speed_explanation = f"Moderate track closing rate: {px_s:.0f} px/s"
            elif px_s >= 150.0:
                speed_pts = 12.0
                speed_explanation = f"Low track closing rate: {px_s:.0f} px/s"
            else:
                speed_pts = 5.0
                speed_explanation = f"Minimal track motion: {px_s:.0f} px/s"

        speed_score = min(25.0, max(0.0, speed_pts))
        components.append(
            ScoreComponent(
                key="closing_speed",
                label="Pre-Impact Closing Velocity",
                score=speed_score,
                max_score=25.0,
                weight_pct=25.0,
                explanation=speed_explanation,
                metrics={
                    "closing_speed_px_per_sec": round(features.closing_speed_px_per_sec, 1),
                    "closing_speed_mps": round(features.closing_speed_mps, 2) if features.closing_speed_mps is not None else None,
                },
            )
        )
        if speed_score >= 18.0:
            factors.append(f"High pre-impact closing velocity: {speed_explanation}.")

        # -------------------------------------------------------------
        # 3. Post-Impact Scene State (Capacity: 20 points)
        # -------------------------------------------------------------
        post_pts = 0.0
        post_details: List[str] = []

        if features.people_on_ground:
            post_pts += 12.0
            post_details.append("Person/rider prone on roadway (+12)")
        if features.blocked_lanes:
            post_pts += 5.0
            post_details.append("Traffic travel lane(s) blocked (+5)")

        if features.stationary_vehicles_post_impact >= 2:
            post_pts += 5.0
            post_details.append(f"{features.stationary_vehicles_post_impact} disabled vehicles (+5)")
        elif features.stationary_vehicles_post_impact == 1:
            post_pts += 3.0
            post_details.append("1 disabled vehicle (+3)")

        post_score = min(20.0, max(0.0, post_pts))
        post_explanation = "; ".join(post_details) if post_details else "No severe aftermath obstruction detected"
        components.append(
            ScoreComponent(
                key="post_impact",
                label="Post-Impact Scene State",
                score=post_score,
                max_score=20.0,
                weight_pct=20.0,
                explanation=post_explanation,
                metrics={
                    "people_on_ground": features.people_on_ground,
                    "blocked_lanes": features.blocked_lanes,
                    "stationary_vehicles": features.stationary_vehicles_post_impact,
                },
            )
        )
        if features.people_on_ground:
            factors.append("Critical hazard: individual detected down on the roadway.")
        if features.blocked_lanes:
            factors.append("Roadway hazard: active traffic travel lane is obstructed.")

        # -------------------------------------------------------------
        # 4. VLM AI Severity Assessment (Capacity: 20 points)
        # -------------------------------------------------------------
        vlm_pts = 0.0
        vlm_sev = str(features.vlm_severity or "").lower()
        conf = features.vlm_confidence if features.vlm_confidence is not None else 0.8
        conf_multiplier = max(0.6, min(1.0, conf))

        if vlm_sev == "high":
            vlm_pts = 20.0 * conf_multiplier
            vlm_explanation = f"Gemini VLM classified high severity ({conf * 100:.0f}% confidence)"
        elif vlm_sev == "medium":
            vlm_pts = 12.0 * conf_multiplier
            vlm_explanation = f"Gemini VLM classified medium severity ({conf * 100:.0f}% confidence)"
        elif vlm_sev == "low":
            vlm_pts = 5.0 * conf_multiplier
            vlm_explanation = f"Gemini VLM classified low severity ({conf * 100:.0f}% confidence)"
        else:
            vlm_pts = 10.0  # Neutral baseline when VLM is unverified or offline
            vlm_explanation = "Neutral baseline (VLM unverified or offline)"

        vlm_score = min(20.0, max(0.0, vlm_pts))
        components.append(
            ScoreComponent(
                key="vlm_opinion",
                label="VLM Visual AI Assessment",
                score=vlm_score,
                max_score=20.0,
                weight_pct=20.0,
                explanation=vlm_explanation,
                metrics={
                    "vlm_severity": features.vlm_severity,
                    "vlm_confidence": features.vlm_confidence,
                },
            )
        )
        if vlm_sev in ("high", "low"):
            factors.append(f"AI Vision Assessment: {vlm_explanation}.")

        # -------------------------------------------------------------
        # Final Total Score & Level Mapping
        # -------------------------------------------------------------
        total_raw = sum(c.score for c in components)
        final_score = int(round(max(0.0, min(100.0, total_raw))))

        if final_score <= low_max:
            level: Literal["low", "medium", "high"] = "low"
        elif final_score <= medium_max:
            level = "medium"
        else:
            level = "high"

        # -------------------------------------------------------------
        # Dynamic Emergency Unit Routing
        # (EMS, police, fire, tow)
        # -------------------------------------------------------------
        routed_units: Set[str] = set()

        # Baseline assignment based on level
        if level == "low":
            routed_units.update(["police", "tow"])
        elif level == "medium":
            routed_units.update(["EMS", "police", "tow"])
        else:  # "high"
            routed_units.update(["EMS", "police", "fire", "tow"])

        # Targeted escalation rules
        if features.pedestrians_count > 0 or features.people_on_ground or features.motorcycles_count > 0 or features.bicycles_count > 0:
            routed_units.add("EMS")

        if (
            (features.closing_speed_mps is not None and features.closing_speed_mps >= 18.0)
            or features.closing_speed_px_per_sec >= 600.0
            or (features.blocked_lanes and final_score >= 60)
            or final_score >= 75
        ):
            routed_units.add("fire")

        if features.vehicles_count > 0 or features.motorcycles_count > 0 or features.heavy_vehicles_count > 0:
            routed_units.add("tow")

        # Police is always required for traffic control and accident investigation
        routed_units.add("police")

        canonical_order = ["EMS", "police", "fire", "tow"]
        recommended_units = [u for u in canonical_order if u in routed_units]

        if not factors:
            factors.append(f"Evaluated as {level.upper()} severity based on collision telemetry.")

        return SeverityResult(
            score=final_score,
            level=level,
            recommended_units=recommended_units,
            components=components,
            factors=factors,
            scorer_version=self.version,
        )


class MLSeverityScorer(BaseSeverityScorer):
    """Machine learning severity scorer that utilizes a trained model when available."""

    def __init__(
        self,
        model: Optional[Any] = None,
        model_path: Optional[str] = None,
        version: str = "ml_severity_v1",
    ):
        self.model = model
        self.model_path = model_path
        self.version = version
        if self.model is None and self.model_path and os.path.exists(self.model_path):
            self._load_model()

    def _load_model(self) -> None:
        try:
            import joblib
            self.model = joblib.load(self.model_path)
            logger.info("Loaded trained severity model from %s", self.model_path)
        except Exception as exc:
            logger.warning("Could not load ML severity model from %s: %s", self.model_path, exc)

    def score(
        self,
        features: SeverityFeatures,
        low_max: int = 35,
        medium_max: int = 70,
    ) -> SeverityResult:
        rule_scorer = RuleBasedSeverityScorer()
        rule_res = rule_scorer.score(features, low_max=low_max, medium_max=medium_max)

        if self.model is None:
            rule_res.scorer_version = f"{self.version}_fallback"
            return rule_res

        try:
            vec = np.array(features.to_vector()).reshape(1, -1)
            if hasattr(self.model, "predict_proba"):
                proba = self.model.predict_proba(vec)[0]
                weights = [20.0, 55.0, 85.0][: len(proba)]
                raw_score = float(np.dot(proba, weights))
            elif hasattr(self.model, "predict"):
                raw_score = float(self.model.predict(vec)[0])
            elif callable(self.model):
                raw_score = float(self.model(vec))
            else:
                raw_score = float(rule_res.score)

            predicted_score = int(round(max(0.0, min(100.0, raw_score))))
        except Exception as exc:
            logger.warning("ML prediction failed; falling back to rule-based score: %s", exc)
            rule_res.scorer_version = f"{self.version}_error_fallback"
            return rule_res

        if predicted_score <= low_max:
            level: Literal["low", "medium", "high"] = "low"
        elif predicted_score <= medium_max:
            level = "medium"
        else:
            level = "high"

        # Blend recommended units
        units = set(rule_res.recommended_units)
        if level == "high":
            units.update(["EMS", "police", "fire", "tow"])
        elif level == "medium":
            units.update(["EMS", "police", "tow"])
        canonical = ["EMS", "police", "fire", "tow"]
        rec_units = [u for u in canonical if u in units]

        factors = [f"Trained model ({self.version}) evaluated severity at {predicted_score}/100."]
        factors.extend(rule_res.factors)

        return SeverityResult(
            score=predicted_score,
            level=level,
            recommended_units=rec_units,
            components=rule_res.components,
            factors=factors,
            scorer_version=self.version,
        )


# Global Scorer Registry
_ACTIVE_SCORER: BaseSeverityScorer = RuleBasedSeverityScorer()
_SCORER_REGISTRY: Dict[str, BaseSeverityScorer] = {
    "rule_based": _ACTIVE_SCORER,
    "ml_model": MLSeverityScorer(),
}


def register_severity_scorer(name: str, scorer: BaseSeverityScorer) -> None:
    """Registers a custom or trained severity scorer in the global registry."""
    _SCORER_REGISTRY[name] = scorer


def set_default_severity_scorer(scorer: BaseSeverityScorer) -> None:
    """Sets the active default severity scorer instance."""
    global _ACTIVE_SCORER
    _ACTIVE_SCORER = scorer


def get_severity_scorer(scorer_type: Optional[str] = None) -> BaseSeverityScorer:
    """Resolves and returns the configured or requested severity scorer."""
    resolved_type = scorer_type or getattr(config, "severity_scorer_type", "rule_based")
    return _SCORER_REGISTRY.get(resolved_type, _ACTIVE_SCORER)


def extract_severity_features(
    incident: Any,
    detection_history: Optional[List[Any]] = None,
    calibration: Optional[CameraCalibration] = None,
    vlm_severity: Optional[str] = None,
    vlm_confidence: Optional[float] = None,
) -> SeverityFeatures:
    """Extracts standardized SeverityFeatures from an incident record and detection telemetry.

    Args:
        incident: AccidentIncident object or dictionary.
        detection_history: Sequence of detection frames across the incident duration.
        calibration: Optional CameraCalibration object (for ppm normalization).
        vlm_severity: Optional VLM assessment override ("low", "medium", "high").
        vlm_confidence: Optional VLM confidence score.

    Returns:
        SeverityFeatures: Extracted quantitative collision features.
    """
    class_name = getattr(incident, "class_name", "") or ""
    class_name_lower = class_name.lower()
    peak_conf = float(getattr(incident, "max_confidence", 0.5) or 0.5)

    duration = 0.0
    if hasattr(incident, "duration"):
        duration = float(incident.duration)
    elif hasattr(incident, "start_time") and hasattr(incident, "end_time") and incident.start_time and incident.end_time:
        duration = float(incident.end_time - incident.start_time)

    # 1. Parse Entities and Track IDs across detection history
    track_classes: Dict[int, str] = {}
    track_boxes_history: Dict[int, List[Tuple[int, Tuple[float, float, float, float]]]] = {}
    track_centroids: Dict[int, List[Tuple[int, Tuple[float, float]]]] = {}

    max_concurrent_classes: Dict[str, int] = {
        "pedestrian": 0,
        "motorcycle": 0,
        "bicycle": 0,
        "vehicle": 0,
        "heavy_vehicle": 0,
    }

    synthetic_id_counter = 5000

    if detection_history:
        for f_idx, record in enumerate(detection_history):
            raw_boxes = []
            if isinstance(record, tuple) and len(record) == 3 and not hasattr(record[0], "bbox") and not isinstance(record[0], dict):
                raw_boxes = record[2]
            elif isinstance(record, tuple) and len(record) == 2 and not hasattr(record[0], "bbox") and not isinstance(record[0], dict):
                raw_boxes = record[1]
            elif isinstance(record, (list, tuple)):
                raw_boxes = record
            elif isinstance(record, dict):
                raw_boxes = record.get("detections", record.get("boxes", []))
            elif hasattr(record, "detections"):
                raw_boxes = record.detections

            if hasattr(raw_boxes, "detections"):
                raw_boxes = raw_boxes.detections

            frame_class_counts: Dict[str, int] = {k: 0 for k in max_concurrent_classes}

            for b in raw_boxes or []:
                if hasattr(b, "bbox"):
                    bbox = b.bbox
                    cls_str = str(getattr(b, "class_name", "")).lower()
                    trk_id = getattr(b, "track_id", None)
                elif isinstance(b, dict):
                    bbox = b.get("bbox", (0, 0, 0, 0))
                    cls_str = str(b.get("class_name", "")).lower()
                    trk_id = b.get("track_id", None)
                else:
                    continue

                if trk_id is None or trk_id <= 0:
                    trk_id = synthetic_id_counter
                    synthetic_id_counter += 1

                category = "vehicle"
                if cls_str in ("person", "pedestrian"):
                    category = "pedestrian"
                elif cls_str in ("motorcycle", "motorbike", "bike", "scooter"):
                    category = "motorcycle"
                elif cls_str == "bicycle":
                    category = "bicycle"
                elif any(h == cls_str for h in ("truck", "bus", "lorry")):
                    category = "heavy_vehicle"
                elif "bike_person" in cls_str:
                    category = "motorcycle"
                elif "car_person" in cls_str or "car_bike" in cls_str or "car_car" in cls_str:
                    category = "vehicle"
                elif "bike_bike" in cls_str or "bike_object" in cls_str:
                    category = "motorcycle"
                elif any(m in cls_str for m in ("motorcycle", "motorbike", "bike", "scooter")):
                    category = "motorcycle"
                elif any(p in cls_str for p in ("person", "pedestrian")):
                    category = "pedestrian"
                elif any(h in cls_str for h in ("truck", "bus", "lorry")):
                    category = "heavy_vehicle"

                frame_class_counts[category] += 1
                track_classes[trk_id] = category

                x1, y1, x2, y2 = bbox
                cx = (x1 + x2) / 2.0
                cy = (y1 + y2) / 2.0
                track_boxes_history.setdefault(trk_id, []).append((f_idx, bbox))
                track_centroids.setdefault(trk_id, []).append((f_idx, (cx, cy)))

            for cat, count in frame_class_counts.items():
                if count > max_concurrent_classes[cat]:
                    max_concurrent_classes[cat] = count

    # Summarize entity counts from tracks or concurrent detections
    ped_count = sum(1 for c in track_classes.values() if c == "pedestrian") or max_concurrent_classes["pedestrian"]
    moto_count = sum(1 for c in track_classes.values() if c == "motorcycle") or max_concurrent_classes["motorcycle"]
    bike_count = sum(1 for c in track_classes.values() if c == "bicycle") or max_concurrent_classes["bicycle"]
    heavy_count = sum(1 for c in track_classes.values() if c == "heavy_vehicle") or max_concurrent_classes["heavy_vehicle"]
    veh_count = sum(1 for c in track_classes.values() if c == "vehicle") or max_concurrent_classes["vehicle"]

    # Deduce from incident class_name if no fine-grained tracks available
    if "person" in class_name_lower and ped_count == 0:
        ped_count = 1
    if "bike" in class_name_lower and moto_count == 0:
        moto_count = 1
    if "car_car" in class_name_lower and veh_count < 2:
        veh_count = 2
    elif "car" in class_name_lower and veh_count == 0:
        veh_count = 1

    reported_involved = getattr(incident, "vehicles_involved", None)
    if reported_involved and reported_involved > (veh_count + moto_count + heavy_count):
        veh_count = max(veh_count, reported_involved)

    total_entities = max(1, ped_count + moto_count + bike_count + heavy_count + veh_count)

    # 2. Estimate Closing Speed Before Impact
    closing_speed_px = 0.0
    impact_frame_idx = len(detection_history) // 2 if detection_history else 0

    fps = calibration.fps if calibration else 30.0
    dt_per_frame = 1.0 / max(1.0, fps)

    # Calculate track velocities and inter-track closing rates prior to impact
    active_pre_tracks = {
        tid: [(f, pt) for f, pt in pts if f <= impact_frame_idx]
        for tid, pts in track_centroids.items()
    }

    # Inter-track convergence rates
    tids = list(active_pre_tracks.keys())
    for i in range(len(tids)):
        tid_a = tids[i]
        pts_a = dict(active_pre_tracks[tid_a])
        for j in range(i + 1, len(tids)):
            tid_b = tids[j]
            pts_b = dict(active_pre_tracks[tid_b])
            common_frames = sorted(set(pts_a.keys()) & set(pts_b.keys()))
            if len(common_frames) >= 2:
                for k in range(1, len(common_frames)):
                    f_prev = common_frames[k - 1]
                    f_curr = common_frames[k]
                    df = max(1, f_curr - f_prev)
                    d_prev = math.hypot(pts_a[f_prev][0] - pts_b[f_prev][0], pts_a[f_prev][1] - pts_b[f_prev][1])
                    d_curr = math.hypot(pts_a[f_curr][0] - pts_b[f_curr][0], pts_a[f_curr][1] - pts_b[f_curr][1])
                    if d_curr < d_prev:
                        # Approaching
                        rate = (d_prev - d_curr) / (df * dt_per_frame)
                        if rate > closing_speed_px:
                            closing_speed_px = rate

    # If no converging pair found, use maximum individual speed
    if closing_speed_px == 0.0:
        for tid, pts in active_pre_tracks.items():
            if len(pts) >= 2:
                for k in range(1, len(pts)):
                    f_prev, (x1, y1) = pts[k - 1]
                    f_curr, (x2, y2) = pts[k]
                    df = max(1, f_curr - f_prev)
                    dist = math.hypot(x2 - x1, y2 - y1)
                    speed = dist / (df * dt_per_frame)
                    if speed > closing_speed_px:
                        closing_speed_px = speed

    closing_speed_mps: Optional[float] = None
    ppm = calibration.pixels_per_meter if calibration else getattr(config, "camera_pixels_per_meter", None)
    if ppm and ppm > 0:
        closing_speed_mps = closing_speed_px / ppm

    # 3. Post-Impact State (stationary vehicles, people on ground, blocked lanes)
    stationary_count = 0
    people_on_ground = False
    blocked_lanes = False

    active_post_tracks = {
        tid: [(f, pt) for f, pt in pts if f >= impact_frame_idx]
        for tid, pts in track_centroids.items()
    }

    for tid, pts in active_post_tracks.items():
        if len(pts) >= 2:
            moves = [
                math.hypot(pts[k][1][0] - pts[k - 1][1][0], pts[k][1][1] - pts[k - 1][1][1])
                for k in range(1, len(pts))
            ]
            avg_move_per_frame = sum(moves) / len(moves)
            if avg_move_per_frame < 5.0 and track_classes.get(tid) in ("vehicle", "heavy_vehicle", "motorcycle"):
                stationary_count += 1

    # Check for people on ground
    for tid, boxes in track_boxes_history.items():
        if track_classes.get(tid) == "pedestrian":
            for f, bbox in boxes:
                if f >= impact_frame_idx:
                    x1, y1, x2, y2 = bbox
                    w = x2 - x1
                    h = y2 - y1
                    # A horizontal/prone person has aspect ratio width >= height * 1.05
                    if h > 0 and (w / h) >= 1.05:
                        people_on_ground = True
                        break

    # Check for blocked lanes: stationary vehicles in lane bounds or central road corridor
    if stationary_count > 0:
        if calibration and calibration.lane_bounds:
            for tid, boxes in track_boxes_history.items():
                if track_classes.get(tid) in ("vehicle", "heavy_vehicle"):
                    latest_box = boxes[-1][1] if boxes else None
                    if latest_box:
                        for lb in calibration.lane_bounds:
                            # Intersection test
                            if not (latest_box[2] < lb[0] or latest_box[0] > lb[2] or latest_box[3] < lb[1] or latest_box[1] > lb[3]):
                                blocked_lanes = True
                                break
        else:
            # Heuristic default: stationary vehicles occupying central travel roadway
            blocked_lanes = stationary_count >= 1

    # 4. Multimodal VLM Assessment
    resolved_vlm_sev = vlm_severity or getattr(incident, "vlm_severity", None)
    resolved_vlm_conf = vlm_confidence if vlm_confidence is not None else getattr(incident, "vlm_confidence", None)

    return SeverityFeatures(
        total_entities=total_entities,
        vehicles_count=veh_count,
        pedestrians_count=ped_count,
        motorcycles_count=moto_count,
        bicycles_count=bike_count,
        heavy_vehicles_count=heavy_count,
        closing_speed_px_per_sec=closing_speed_px,
        closing_speed_mps=closing_speed_mps,
        stationary_vehicles_post_impact=stationary_count,
        people_on_ground=people_on_ground,
        blocked_lanes=blocked_lanes,
        vlm_severity=resolved_vlm_sev,
        vlm_confidence=resolved_vlm_conf,
        class_name=class_name,
        peak_confidence=peak_conf,
        duration_seconds=duration,
    )


def evaluate_incident_severity(
    incident: Any,
    detection_history: Optional[List[Any]] = None,
    calibration: Optional[CameraCalibration] = None,
    vlm_severity: Optional[str] = None,
    vlm_confidence: Optional[float] = None,
    scorer_type: Optional[str] = None,
) -> SeverityResult:
    """Convenience pipeline function to extract features and compute severity for an incident.

    Args:
        incident: AccidentIncident object or dictionary.
        detection_history: Detection records sequence.
        calibration: Optional CameraCalibration data.
        vlm_severity: Optional VLM severity level override.
        vlm_confidence: Optional VLM confidence score.
        scorer_type: Optional scorer name from registry.

    Returns:
        SeverityResult: Evaluation result containing score, level, units, and breakdown.
    """
    features = extract_severity_features(
        incident=incident,
        detection_history=detection_history,
        calibration=calibration,
        vlm_severity=vlm_severity,
        vlm_confidence=vlm_confidence,
    )
    scorer = get_severity_scorer(scorer_type)
    low_max = getattr(config, "severity_low_max", 35)
    medium_max = getattr(config, "severity_medium_max", 70)
    return scorer.score(features, low_max=low_max, medium_max=medium_max)
