"""
backend.app.services.drift_service
Model drift monitoring and camera anomaly detection service.

Monitors:
1. Weekly distribution of confidence scores per camera (binned 0.0-1.0 into 10 deciles).
2. Weekly detection counts per camera.
3. Statistical drift algorithms:
   - Population Stability Index (PSI) on confidence score distributions.
   - Sudden count variance ratio (detects camera occlusion, dirty lens, or sudden perspective shifts).
   - Mean confidence delta (blurry/dirty lens indicator).
4. Automatic alert dispatch when sharp drift is detected.
"""
import json
import logging
import math
import os
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from sqlmodel import Session, select

from acs.config import config
from backend.app.core.metrics import record_model_drift
from backend.app.db.database import engine
from backend.app.db.models import AuditLog, Camera, Incident

logger = logging.getLogger("acs.drift_service")

# Drift file location
DRIFT_STATE_PATH = os.path.join(getattr(config, "reports_dir", "reports"), "model_drift_state.json")

# Statistical thresholds
PSI_WARNING_THRESHOLD = 0.15
PSI_ALERT_THRESHOLD = 0.25          # PSI >= 0.25 represents significant population shift
COUNT_DROP_THRESHOLD = 0.60         # >60% drop suggests lens occlusion / dirty / disconnected
COUNT_SURGE_THRESHOLD = 2.00        # >200% surge suggests camera shake / false positive burst
CONFIDENCE_DROP_THRESHOLD = 0.15    # Drop of >0.15 in mean confidence indicates dirty lens blur


@dataclass
class DistributionMetrics:
    """Confidence score distribution and count metrics for a time window."""

    window_name: str
    sample_count: int
    mean_confidence: float
    confidence_bins: List[float]  # 10 bins: [0.0-0.1, ..., 0.9-1.0] proportions
    detection_count: int
    timestamp: float = field(default_factory=time.time)


@dataclass
class CameraDriftReport:
    """Drift evaluation report for a single CCTV camera."""

    camera_id: str
    camera_name: str
    psi_score: float
    count_delta_ratio: float
    mean_confidence_delta: float
    drift_detected: bool
    drift_severity: str  # "none", "warning", "critical"
    reasons: List[str]
    suggested_action: str
    baseline: Optional[Dict[str, Any]] = None
    current_week: Optional[Dict[str, Any]] = None
    checked_at: float = field(default_factory=time.time)


def compute_distribution(confidences: List[float], total_detections: int, window_name: str = "current") -> DistributionMetrics:
    """Calculates 10-decile histogram distribution and summary statistics."""
    if not confidences:
        return DistributionMetrics(
            window_name=window_name,
            sample_count=0,
            mean_confidence=0.0,
            confidence_bins=[0.1] * 10,
            detection_count=total_detections,
        )

    arr = np.clip(np.array(confidences, dtype=np.float64), 0.0, 1.0)
    mean_conf = float(np.mean(arr))
    counts, _ = np.histogram(arr, bins=10, range=(0.0, 1.0))
    total = len(arr)
    # Proportions with small epsilon smoothing
    proportions = [(c / total) for c in counts]

    return DistributionMetrics(
        window_name=window_name,
        sample_count=total,
        mean_confidence=round(mean_conf, 4),
        confidence_bins=[round(p, 4) for p in proportions],
        detection_count=total_detections,
    )


def compute_psi(baseline_bins: List[float], current_bins: List[float], epsilon: float = 1e-4) -> float:
    """Calculates Population Stability Index (PSI) between baseline and current distributions."""
    if not baseline_bins or not current_bins or len(baseline_bins) != len(current_bins):
        return 0.0

    psi = 0.0
    for b_prop, c_prop in zip(baseline_bins, current_bins):
        b_safe = max(b_prop, epsilon)
        c_safe = max(c_prop, epsilon)
        psi += (c_safe - b_safe) * math.log(c_safe / b_safe)

    return round(float(psi), 4)


class ModelDriftService:
    """Coordinates camera drift tracking, persistence, and automated alert dispatch."""

    def __init__(self, state_file: str = DRIFT_STATE_PATH) -> None:
        self.state_file = state_file
        self.state: Dict[str, Any] = self._load_state()

    def _load_state(self) -> Dict[str, Any]:
        """Loads historical baseline and check state from disk."""
        if os.path.exists(self.state_file):
            try:
                with open(self.state_file, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                logger.warning("Failed loading drift state file (%s): %s", self.state_file, e)
        return {"baselines": {}, "last_reports": {}}

    def _save_state(self) -> None:
        """Persists drift state to disk."""
        try:
            os.makedirs(os.path.dirname(os.path.abspath(self.state_file)), exist_ok=True)
            with open(self.state_file, "w", encoding="utf-8") as f:
                json.dump(self.state, f, indent=2)
        except Exception as e:
            logger.warning("Failed saving drift state to %s: %s", self.state_file, e)

    def extract_camera_observations(
        self,
        camera_id: str,
        start_time: float,
        end_time: float,
    ) -> Tuple[List[float], int]:
        """Extracts confidence scores and incident detection counts from database within a time window."""
        confidences: List[float] = []
        detection_count = 0

        with Session(engine) as session:
            incidents = session.exec(
                select(Incident).where(
                    Incident.camera_id == camera_id,
                    Incident.created_at >= start_time,
                    Incident.created_at <= end_time,
                )
            ).all()

            for inc in incidents:
                detection_count += 1
                if inc.max_confidence is not None:
                    confidences.append(float(inc.max_confidence))
                if inc.vlm_confidence is not None:
                    confidences.append(float(inc.vlm_confidence))

        return confidences, detection_count

    def set_camera_baseline(
        self,
        camera_id: str,
        custom_metrics: Optional[DistributionMetrics] = None,
    ) -> DistributionMetrics:
        """Establishes or updates baseline distribution for a camera."""
        now = time.time()
        # Default baseline: past 14 days of data, or synthetic standard traffic calibration
        if custom_metrics:
            baseline_metrics = custom_metrics
        else:
            confidences, count = self.extract_camera_observations(
                camera_id=camera_id,
                start_time=now - (14 * 86400),
                end_time=now,
            )
            if not confidences:
                # Default canonical baseline for calibrated YOLO accident detector
                confidences = [0.45, 0.55, 0.65, 0.72, 0.78, 0.82, 0.88, 0.91, 0.94] * 5
                count = len(confidences)
            baseline_metrics = compute_distribution(confidences, count, window_name="baseline")

        if "baselines" not in self.state:
            self.state["baselines"] = {}
        self.state["baselines"][camera_id] = asdict(baseline_metrics)
        self._save_state()
        logger.info("Established baseline distribution for camera %s (mean conf: %.3f, count: %d)",
                    camera_id, baseline_metrics.mean_confidence, baseline_metrics.detection_count)
        return baseline_metrics

    def check_camera_drift(
        self,
        camera_id: str,
        camera_name: str = "",
        window_days: int = 7,
    ) -> CameraDriftReport:
        """Evaluates whether confidence scores or detection counts shifted sharply over the window."""
        now = time.time()
        start_time = now - (window_days * 86400)

        # 1. Fetch or initialize baseline
        baseline_dict = self.state.get("baselines", {}).get(camera_id)
        if not baseline_dict:
            baseline_metrics = self.set_camera_baseline(camera_id)
            baseline_dict = asdict(baseline_metrics)

        baseline_bins = baseline_dict.get("confidence_bins", [0.1] * 10)
        baseline_mean_conf = baseline_dict.get("mean_confidence", 0.75)
        baseline_count = baseline_dict.get("detection_count", 10)

        # 2. Extract current week observations
        current_confidences, current_count = self.extract_camera_observations(
            camera_id=camera_id,
            start_time=start_time,
            end_time=now,
        )

        current_metrics = compute_distribution(
            current_confidences,
            current_count,
            window_name=f"last_{window_days}d",
        )

        # 3. Compute drift metrics
        psi_score = compute_psi(baseline_bins, current_metrics.confidence_bins)

        # Count delta ratio relative to baseline
        denom = max(1.0, float(baseline_count))
        count_delta_ratio = (current_count - baseline_count) / denom

        # Mean confidence shift
        mean_conf_delta = baseline_mean_conf - current_metrics.mean_confidence

        # 4. Diagnose Drift Anomaly Causes
        reasons: List[str] = []
        is_drifted = False
        severity = "none"

        # Check PSI threshold
        if psi_score >= PSI_ALERT_THRESHOLD:
            is_drifted = True
            severity = "critical"
            reasons.append(
                f"Severe confidence score distribution shift (PSI: {psi_score:.3f} >= {PSI_ALERT_THRESHOLD}). "
                "Possible camera repositioning, altered field of view, or significant scene change."
            )
        elif psi_score >= PSI_WARNING_THRESHOLD:
            severity = "warning"
            reasons.append(
                f"Moderate confidence score distribution shift (PSI: {psi_score:.3f})."
            )

        # Check Mean Confidence Drop (Dirty lens / Blur / Occlusion)
        if mean_conf_delta >= CONFIDENCE_DROP_THRESHOLD:
            is_drifted = True
            severity = "critical"
            reasons.append(
                f"Sharp drop in average confidence (-{mean_conf_delta:.3f}). "
                "Likely dirty lens, smudge, spiderweb, rain streak, or optical occlusion."
            )

        # Check Count Anomaly
        if count_delta_ratio <= -COUNT_DROP_THRESHOLD:
            is_drifted = True
            severity = "critical"
            reasons.append(
                f"Severe detection volume drop ({count_delta_ratio * 100:.1f}% vs baseline). "
                "Camera may be partially blocked, dirty, or experiencing stream dropouts."
            )
        elif count_delta_ratio >= COUNT_SURGE_THRESHOLD:
            is_drifted = True
            severity = "critical"
            reasons.append(
                f"Abnormal detection surge (+{count_delta_ratio * 100:.1f}% vs baseline). "
                "Camera may have vibrated/moved, capturing repetitive false alarms or glares."
            )

        # Suggested remediation action
        if "dirty" in " ".join(reasons).lower() or mean_conf_delta >= CONFIDENCE_DROP_THRESHOLD:
            action = "Inspect physical camera lens for dirt, water spots, or cobwebs; clean casing."
        elif "repositioning" in " ".join(reasons).lower() or psi_score >= PSI_ALERT_THRESHOLD:
            action = "Verify camera angle alignment against reference preset; recalibrate detection zones."
        elif is_drifted:
            action = "Review camera video feed and update baseline distribution if scene permanently changed."
        else:
            action = "Camera operating within normal historical distribution."

        # Update Prometheus metrics
        record_model_drift(camera_id, "confidence", psi_score >= PSI_ALERT_THRESHOLD)
        record_model_drift(camera_id, "count", abs(count_delta_ratio) >= COUNT_DROP_THRESHOLD)
        record_model_drift(camera_id, "composite", is_drifted)

        report = CameraDriftReport(
            camera_id=camera_id,
            camera_name=camera_name or camera_id,
            psi_score=psi_score,
            count_delta_ratio=round(count_delta_ratio, 3),
            mean_confidence_delta=round(mean_conf_delta, 3),
            drift_detected=is_drifted,
            drift_severity=severity,
            reasons=reasons,
            suggested_action=action,
            baseline=baseline_dict,
            current_week=asdict(current_metrics),
            checked_at=now,
        )

        # Store in state
        if "last_reports" not in self.state:
            self.state["last_reports"] = {}
        self.state["last_reports"][camera_id] = asdict(report)
        self._save_state()

        # If critical drift detected, dispatch notification alert and record audit log
        if is_drifted:
            self._dispatch_drift_alert(report)

        return report

    def _dispatch_drift_alert(self, report: CameraDriftReport) -> None:
        """Dispatches automated alert through ACS notification manager."""
        logger.warning(
            "MODEL DRIFT ALERT for camera '%s' (%s): %s",
            report.camera_name,
            report.camera_id,
            " | ".join(report.reasons),
        )
        try:
            with Session(engine) as session:
                audit = AuditLog(
                    timestamp=time.time(),
                    user_id=None,
                    username="system_drift_monitor",
                    user_role="system",
                    action="model_drift_alert_triggered",
                    resource_type="camera",
                    resource_id=report.camera_id,
                    details=json.dumps({
                        "camera_id": report.camera_id,
                        "camera_name": report.camera_name,
                        "psi_score": report.psi_score,
                        "count_delta": report.count_delta_ratio,
                        "reasons": report.reasons,
                        "suggested_action": report.suggested_action,
                    }),
                    ip_address="127.0.0.1",
                )
                session.add(audit)
                session.commit()
        except Exception as e:
            logger.debug("Failed saving drift audit log: %s", e)

        # Send alert via acs.alerts
        try:
            from acs.alerts import AlertPayload, alert_manager
            alert_manager.send(
                AlertPayload(
                    incident_id=f"DRIFT-{report.camera_id}-{int(time.time())}",
                    class_name="camera_anomaly_drift",
                    timestamp=time.strftime("%Y-%m-%d %H:%M:%S UTC"),
                    confidence=report.psi_score,
                    report_text=(
                        f"🚨 CAMERA MODEL DRIFT ALERT\n"
                        f"Camera: {report.camera_name} ({report.camera_id})\n"
                        f"Status: {report.drift_severity.upper()}\n"
                        f"PSI Score: {report.psi_score:.3f}\n"
                        f"Count Shift: {report.count_delta_ratio * 100:+.1f}%\n"
                        f"Mean Conf Delta: {report.mean_confidence_delta:+.3f}\n\n"
                        f"Reasons:\n- " + "\n- ".join(report.reasons) + "\n\n"
                        f"Action: {report.suggested_action}"
                    ),
                    dry_run=False,
                    severity="high" if report.drift_severity == "critical" else "medium",
                    severity_score=85 if report.drift_severity == "critical" else 60,
                    camera_id=report.camera_id,
                    metadata={"is_drift_alert": True, "reasons": report.reasons},
                )
            )
        except Exception as alert_err:
            logger.debug("Failed to dispatch drift alert notification: %s", alert_err)

    def check_all_cameras(self) -> List[CameraDriftReport]:
        """Runs drift checks across all cameras configured in the database."""
        reports: List[CameraDriftReport] = []
        with Session(engine) as session:
            cameras = session.exec(select(Camera)).all()
            for cam in cameras:
                report = self.check_camera_drift(cam.id, camera_name=cam.name)
                reports.append(report)
        return reports


# Global singleton drift service
drift_service = ModelDriftService()
