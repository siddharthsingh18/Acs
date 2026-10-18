"""
backend.app.services.incident_service
Service layer for incident CRUD operations, filtering, CSV migration, and stats.
"""
import csv
import datetime
import io
import json
import logging
import os
import time
from typing import Any, Dict, List, Optional, Tuple, Union

from sqlmodel import Session, func, select

from acs.llm import generate_dispatch_report
from acs.reporter import read_incidents_csv
from backend.app.db.database import engine
from backend.app.db.models import Incident, ReviewFeedback

logger = logging.getLogger("acs.backend.incident_service")


def parse_timestamp_to_epoch(ts_str: str) -> float:
    """Parses various timestamp formats into a Unix epoch float."""
    if not ts_str:
        return time.time()
    for fmt in (
        "%Y-%m-%d %H:%M:%S UTC",
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%dT%H:%M:%S%z",
        "%Y-%m-%dT%H:%M:%S",
    ):
        try:
            cleaned = ts_str.strip()
            dt = datetime.datetime.strptime(cleaned, fmt)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=datetime.timezone.utc)
            return dt.timestamp()
        except Exception:
            continue
    return time.time()


def import_csv_to_db(reports_dir: Optional[str] = None) -> int:
    """One-time migration helper importing existing CSV records into SQLite.

    Deduplicates by incident_id so running repeatedly is safe and idempotent.

    Returns:
        int: Number of new records imported.
    """
    df = read_incidents_csv(reports_dir=reports_dir)
    if df.empty:
        return 0

    imported_count = 0
    with Session(engine) as session:
        existing_ids = set(session.exec(select(Incident.incident_id)).all())

        for _, row in df.iterrows():
            inc_id = str(row.get("incident_id", "")).strip()
            if not inc_id or inc_id in existing_ids:
                continue

            start_ts = str(row.get("start_timestamp", ""))
            end_ts = str(row.get("end_timestamp", ""))
            cls_name = str(row.get("class", "accident"))
            conf_val = float(row.get("max_confidence", 0.0))
            img_path = str(row.get("image_path", ""))
            keyframe_name = os.path.basename(img_path)
            kf_url = f"/media/keyframes/{keyframe_name}" if keyframe_name else ""

            epoch_time = parse_timestamp_to_epoch(start_ts)
            end_epoch = parse_timestamp_to_epoch(end_ts)
            duration = max(0.1, end_epoch - epoch_time) if end_epoch >= epoch_time else 0.0

            incident = Incident(
                incident_id=inc_id,
                start_timestamp=start_ts,
                end_timestamp=end_ts,
                class_name=cls_name,
                max_confidence=conf_val,
                image_path=img_path,
                keyframe_url=kf_url,
                severity="medium",
                status="pending",
                created_at=epoch_time,
                duration=duration,
            )
            session.add(incident)
            existing_ids.add(inc_id)
            imported_count += 1

        if imported_count > 0:
            session.commit()
            logger.info("Imported %d historical incidents from CSV into SQLite.", imported_count)

    return imported_count


def save_incident_to_db(
    incident_id: str,
    start_timestamp: str,
    end_timestamp: str,
    class_name: str,
    max_confidence: float,
    image_path: str,
    duration: float = 0.0,
    status: str = "pending",
    verification_status: Optional[str] = None,
    vlm_confidence: Optional[float] = None,
    vlm_reason: Optional[str] = None,
    physics_passed: Optional[bool] = None,
    clip_path: Optional[str] = None,
    clip_annotated_path: Optional[str] = None,
    impact_offset_seconds: Optional[float] = None,
    severity: Optional[str] = None,
    severity_score: Optional[int] = None,
    severity_breakdown: Optional[Union[str, dict]] = None,
    recommended_units: Optional[Union[str, list]] = None,
    camera_id: Optional[str] = None,
    session: Optional[Session] = None,
) -> Incident:
    """Inserts or updates an incident record in SQLite."""
    keyframe_name = os.path.basename(image_path)
    kf_url = f"/media/keyframes/{keyframe_name}" if keyframe_name else ""
    created_epoch = parse_timestamp_to_epoch(start_timestamp)

    breakdown_json = (
        json.dumps(severity_breakdown)
        if isinstance(severity_breakdown, dict)
        else severity_breakdown
    )
    units_json = (
        json.dumps(recommended_units)
        if isinstance(recommended_units, list)
        else recommended_units
    )

    def _persist(s: Session) -> Incident:
        existing = s.exec(select(Incident).where(Incident.incident_id == incident_id)).first()
        if existing:
            existing.end_timestamp = end_timestamp
            existing.max_confidence = max(existing.max_confidence, max_confidence)
            existing.duration = max(existing.duration, duration)
            if camera_id is not None:
                existing.camera_id = camera_id
            if verification_status is not None:
                existing.verification_status = verification_status
            if vlm_confidence is not None:
                existing.vlm_confidence = vlm_confidence
            if vlm_reason is not None:
                existing.vlm_reason = vlm_reason
            if physics_passed is not None:
                existing.physics_passed = physics_passed
            if clip_path is not None:
                existing.clip_path = clip_path
            if clip_annotated_path is not None:
                existing.clip_annotated_path = clip_annotated_path
            if impact_offset_seconds is not None:
                existing.impact_offset_seconds = impact_offset_seconds
            if severity is not None:
                existing.severity = severity
            if severity_score is not None:
                existing.severity_score = severity_score
            if breakdown_json is not None:
                existing.severity_breakdown = breakdown_json
            if units_json is not None:
                existing.recommended_units = units_json
            if status != "pending":
                existing.status = status
            s.add(existing)
            s.commit()
            s.refresh(existing)
            return existing

        new_inc = Incident(
            incident_id=incident_id,
            camera_id=camera_id,
            start_timestamp=start_timestamp,
            end_timestamp=end_timestamp,
            class_name=class_name,
            max_confidence=max_confidence,
            image_path=image_path,
            keyframe_url=kf_url,
            severity=severity or "medium",
            severity_score=severity_score if severity_score is not None else 50,
            severity_breakdown=breakdown_json,
            recommended_units=units_json,
            status=status,
            verification_status=verification_status,
            vlm_confidence=vlm_confidence,
            vlm_reason=vlm_reason,
            physics_passed=physics_passed,
            clip_path=clip_path,
            clip_annotated_path=clip_annotated_path,
            impact_offset_seconds=impact_offset_seconds,
            created_at=created_epoch,
            duration=duration,
        )
        s.add(new_inc)
        s.commit()
        s.refresh(new_inc)
        return new_inc

    if session is not None:
        return _persist(session)
    with Session(engine) as s:
        return _persist(s)


def get_incidents(
    session: Session,
    page: int = 1,
    page_size: int = 20,
    class_name: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    severity: Optional[str] = None,
    status: Optional[str] = None,
    sort_by: str = "created_at",
    order: str = "desc",
) -> Tuple[List[Incident], int]:
    """Retrieves paginated, filtered, and sorted incidents from SQLite."""
    stmt = select(Incident)

    if class_name:
        stmt = stmt.where(Incident.class_name == class_name)
    if severity:
        stmt = stmt.where(Incident.severity == severity.lower())
    if status:
        stmt = stmt.where(Incident.status == status.lower())
    if start_date:
        stmt = stmt.where(Incident.start_timestamp >= start_date)
    if end_date:
        stmt = stmt.where(Incident.start_timestamp <= end_date)

    # Count total matching rows
    count_stmt = select(func.count()).select_from(stmt.subquery())
    total_count = session.exec(count_stmt).one()

    # Apply sorting
    sort_col = getattr(Incident, sort_by, Incident.created_at)
    if order.lower() == "asc":
        stmt = stmt.order_by(sort_col.asc())
    else:
        stmt = stmt.order_by(sort_col.desc())

    # Apply pagination
    offset = max(0, (page - 1) * page_size)
    stmt = stmt.offset(offset).limit(page_size)

    items = list(session.exec(stmt).all())
    return items, total_count


def get_incident_by_id(session: Session, incident_id: str) -> Optional[Incident]:
    """Finds an incident by its ID."""
    return session.exec(select(Incident).where(Incident.incident_id == incident_id)).first()


def update_incident_status(
    session: Session,
    incident_id: str,
    new_status: str,
    reviewer_name: Optional[str] = None,
    notes: Optional[str] = None,
    corrected_class: Optional[str] = None,
    corrected_boxes: Optional[Union[str, list]] = None,
    decision: Optional[str] = None,
) -> Optional[Incident]:
    """Updates human-in-the-loop review status on an incident and logs training feedback."""
    incident = get_incident_by_id(session, incident_id)
    if not incident:
        return None

    normalized_status = new_status.lower()
    incident.status = normalized_status
    incident.reviewer_name = reviewer_name or "Dispatch Supervisor"
    reviewed_at_str = datetime.datetime.now(datetime.timezone.utc).isoformat()
    incident.reviewed_at = reviewed_at_str

    # Determine feedback decision label
    if decision is not None:
        actual_decision = decision.lower()
    elif normalized_status in ("approved", "dispatched"):
        actual_decision = "accident"
    elif normalized_status == "rejected":
        actual_decision = "not_accident"
    elif normalized_status == "wrong_class":
        actual_decision = "wrong_class"
    else:
        actual_decision = normalized_status

    incident.reviewed_label = actual_decision
    if corrected_class:
        incident.corrected_class = corrected_class
    if corrected_boxes:
        incident.corrected_boxes = (
            json.dumps(corrected_boxes)
            if not isinstance(corrected_boxes, str)
            else corrected_boxes
        )

    session.add(incident)

    # Store or update ReviewFeedback record capturing label with keyframe and clips
    boxes_str = incident.corrected_boxes
    existing_fb = session.exec(
        select(ReviewFeedback).where(ReviewFeedback.incident_id == incident_id)
    ).first()

    if existing_fb:
        existing_fb.decision = actual_decision
        existing_fb.original_class = incident.class_name
        existing_fb.corrected_class = incident.corrected_class
        existing_fb.confidence = incident.max_confidence
        existing_fb.image_path = incident.image_path
        existing_fb.clip_path = incident.clip_path
        existing_fb.clip_annotated_path = incident.clip_annotated_path
        existing_fb.reviewer_name = incident.reviewer_name
        existing_fb.reviewed_at = reviewed_at_str
        if notes is not None:
            existing_fb.notes = notes
        if boxes_str is not None:
            existing_fb.corrected_boxes = boxes_str
        existing_fb.is_exported = False  # Mark for staging re-export
        session.add(existing_fb)
    else:
        new_fb = ReviewFeedback(
            incident_id=incident.incident_id,
            decision=actual_decision,
            original_class=incident.class_name,
            corrected_class=incident.corrected_class,
            confidence=incident.max_confidence,
            image_path=incident.image_path,
            clip_path=incident.clip_path,
            clip_annotated_path=incident.clip_annotated_path,
            reviewer_name=incident.reviewer_name,
            reviewed_at=reviewed_at_str,
            notes=notes,
            corrected_boxes=boxes_str,
            is_exported=False,
            created_at=time.time(),
        )
        session.add(new_fb)

    session.commit()
    session.refresh(incident)
    return incident


def record_review_decision(
    session: Session,
    incident_id: str,
    decision: str,
    corrected_class: Optional[str] = None,
    corrected_boxes: Optional[Union[str, list]] = None,
    reviewer_name: str = "Operator",
    notes: Optional[str] = None,
) -> Optional[ReviewFeedback]:
    """Records a review decision (accident, not_accident, wrong_class) from the review queue."""
    incident = get_incident_by_id(session, incident_id)
    if not incident:
        return None

    decision_norm = decision.lower()
    if decision_norm == "accident":
        new_status = "approved"
    elif decision_norm == "not_accident":
        new_status = "rejected"
    elif decision_norm == "wrong_class":
        new_status = "approved"
    else:
        new_status = "approved"

    update_incident_status(
        session=session,
        incident_id=incident_id,
        new_status=new_status,
        reviewer_name=reviewer_name,
        notes=notes,
        corrected_class=corrected_class,
        corrected_boxes=corrected_boxes,
        decision=decision_norm,
    )

    fb = session.exec(
        select(ReviewFeedback).where(ReviewFeedback.incident_id == incident_id)
    ).first()
    return fb


def undo_review_decision(session: Session, incident_id: str) -> Optional[Incident]:
    """Reverts an incident back to pending status and clears unexported review feedback."""
    incident = get_incident_by_id(session, incident_id)
    if not incident:
        return None

    incident.status = "pending"
    incident.reviewed_label = None
    incident.reviewer_name = None
    incident.reviewed_at = None
    incident.corrected_class = None
    incident.corrected_boxes = None
    session.add(incident)

    fb = session.exec(
        select(ReviewFeedback).where(ReviewFeedback.incident_id == incident_id)
    ).first()
    if fb and not fb.is_exported:
        session.delete(fb)

    session.commit()
    session.refresh(incident)
    return incident


def get_review_queue_items(
    session: Session, include_reviewed: bool = False
) -> List[Dict[str, Any]]:
    """Fetches uncertain detections qualifying for fast swipe review.

    Criteria:
    1. Confidence is in the borderline window: 0.30 to 0.60, OR
    2. YOLO and VLM disagreement (e.g. verification_status == 'needs_review', or significant conf delta).
    """
    stmt = select(Incident)
    if not include_reviewed:
        stmt = stmt.where(Incident.reviewed_label == None)  # noqa: E711
        stmt = stmt.where(Incident.status == "pending")

    incidents = list(session.exec(stmt.order_by(Incident.created_at.desc())).all())
    queue_items: List[Dict[str, Any]] = []

    for inc in incidents:
        conf = inc.max_confidence
        vlm_conf = inc.vlm_confidence
        verif_status = inc.verification_status

        # Condition A: Uncertain confidence between 0.30 and 0.60
        is_uncertain_conf = 0.30 <= conf <= 0.60

        # Condition B: YOLO / VLM disagreement
        is_disagreement = False
        reasons: List[str] = []

        if is_uncertain_conf:
            reasons.append(f"Borderline YOLO confidence ({conf:.0%})")

        if verif_status == "needs_review":
            is_disagreement = True
            reasons.append("YOLO/VLM verification flagged for human dispatch review")
        elif verif_status == "unverified":
            if conf >= 0.30:
                is_disagreement = True
                reasons.append("Verification unverified (fallback from API/budget)")
        elif vlm_conf is not None:
            # Significant disagreement between YOLO and VLM
            delta = abs(conf - vlm_conf)
            if delta >= 0.25:
                is_disagreement = True
                reasons.append(f"YOLO/VLM divergence: YOLO={conf:.0%} vs VLM={vlm_conf:.0%}")
            elif conf >= 0.35 and vlm_conf < 0.40:
                is_disagreement = True
                reasons.append(f"VLM did not confirm collision (VLM={vlm_conf:.0%})")

        if not inc.physics_passed and inc.physics_passed is not None:
            reasons.append("Pre-check motion/physics heuristic failed")

        if is_uncertain_conf or is_disagreement:
            reason_str = " • ".join(reasons) if reasons else f"Borderline confidence ({conf:.0%})"
            keyframe_name = os.path.basename(inc.image_path) if inc.image_path else ""
            kf_url = f"/media/keyframes/{keyframe_name}" if keyframe_name else inc.keyframe_url
            clip_url = f"/api/incidents/{inc.incident_id}/clip" if inc.clip_path and os.path.exists(inc.clip_path) else None
            clip_ann_url = (
                f"/api/incidents/{inc.incident_id}/clip?annotated=true"
                if inc.clip_annotated_path and os.path.exists(inc.clip_annotated_path)
                else None
            )

            queue_items.append({
                "incident_id": inc.incident_id,
                "class_name": inc.class_name,
                "max_confidence": inc.max_confidence,
                "image_path": inc.image_path,
                "keyframe_url": kf_url,
                "clip_path": inc.clip_path,
                "clip_url": clip_url,
                "clip_annotated_path": inc.clip_annotated_path,
                "clip_annotated_url": clip_ann_url,
                "verification_status": inc.verification_status,
                "vlm_confidence": inc.vlm_confidence,
                "vlm_reason": inc.vlm_reason,
                "physics_passed": inc.physics_passed,
                "uncertainty_reason": reason_str,
                "start_timestamp": inc.start_timestamp,
                "severity": inc.severity,
                "severity_score": inc.severity_score if inc.severity_score is not None else 50,
                "duration": inc.duration,
            })

    return queue_items


def get_review_stats(session: Session) -> Dict[str, Any]:
    """Calculates statistics on operator review decisions and export readiness."""
    feedbacks = list(session.exec(select(ReviewFeedback)).all())
    total_reviewed = len(feedbacks)
    accidents_approved = sum(1 for fb in feedbacks if fb.decision == "accident")
    hard_negatives = sum(1 for fb in feedbacks if fb.decision == "not_accident")
    wrong_class_count = sum(1 for fb in feedbacks if fb.decision == "wrong_class")
    staging_pending = sum(1 for fb in feedbacks if not fb.is_exported)

    return {
        "total_reviewed": total_reviewed,
        "accidents_approved": accidents_approved,
        "hard_negatives": hard_negatives,
        "wrong_class_count": wrong_class_count,
        "staging_pending_export": staging_pending,
    }


def get_or_generate_dispatch_report(
    session: Session,
    incident_id: str,
    lang: str = "en",
) -> Optional[Dict[str, Any]]:
    """Retrieves cached multimodal dispatch report or generates and caches a new one."""
    incident = get_incident_by_id(session, incident_id)
    if not incident:
        return None

    # Return cached report if already available
    if incident.dispatch_report:
        try:
            cached_data = json.loads(incident.dispatch_report)
            return cached_data
        except Exception:
            pass

    # Call Gemini multimodal dispatch generation
    report_dict = generate_dispatch_report(
        image_path=incident.image_path,
        detections=incident.class_name,
        timestamp=incident.start_timestamp,
        lang=lang,
    )

    # Cache into SQLite
    incident.dispatch_report = json.dumps(report_dict)
    incident.dispatch_text = report_dict.get("rendered_text")
    if "severity" in report_dict and report_dict["severity"]:
        incident.severity = str(report_dict["severity"]).lower()

    session.add(incident)
    session.commit()
    session.refresh(incident)

    return report_dict


def export_incidents_csv(session: Session) -> str:
    """Exports all incidents from the SQLite database to a CSV formatted string."""
    incidents = list(session.exec(select(Incident).order_by(Incident.created_at.desc())).all())
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow([
        "incident_id",
        "start_timestamp",
        "end_timestamp",
        "class",
        "max_confidence",
        "image_path",
        "severity",
        "status",
        "reviewer_name",
        "reviewed_at",
    ])
    for inc in incidents:
        writer.writerow([
            inc.incident_id,
            inc.start_timestamp,
            inc.end_timestamp,
            inc.class_name,
            f"{inc.max_confidence:.4f}",
            inc.image_path,
            inc.severity,
            inc.status,
            inc.reviewer_name or "",
            inc.reviewed_at or "",
        ])
    return output.getvalue()


def get_incident_stats(session: Session) -> Dict[str, Any]:
    """Calculates summary KPIs across all recorded incidents."""
    incidents = list(session.exec(select(Incident)).all())
    total_count = len(incidents)

    if total_count == 0:
        return {
            "total_incidents": 0,
            "incidents_today": 0,
            "by_class": {},
            "by_status": {},
            "average_confidence": 0.0,
            "average_review_time_seconds": None,
        }

    # 1. Incidents today
    now_utc = datetime.datetime.now(datetime.timezone.utc)
    today_start = datetime.datetime(now_utc.year, now_utc.month, now_utc.day, tzinfo=datetime.timezone.utc).timestamp()
    today_end = today_start + 86400.0

    today_count = sum(1 for inc in incidents if today_start <= inc.created_at < today_end)

    # 2. Breakdown by class
    by_class: Dict[str, int] = {}
    for inc in incidents:
        by_class[inc.class_name] = by_class.get(inc.class_name, 0) + 1

    # 3. Breakdown by status
    by_status: Dict[str, int] = {}
    for inc in incidents:
        by_status[inc.status] = by_status.get(inc.status, 0) + 1

    # 4. Average confidence
    avg_conf = sum(inc.max_confidence for inc in incidents) / total_count

    # 5. Average review time
    review_times: List[float] = []
    for inc in incidents:
        if inc.reviewed_at:
            rev_epoch = parse_timestamp_to_epoch(inc.reviewed_at)
            diff = rev_epoch - inc.created_at
            if diff >= 0:
                review_times.append(diff)

    avg_review_time = (sum(review_times) / len(review_times)) if review_times else None

    return {
        "total_incidents": total_count,
        "incidents_today": today_count,
        "by_class": by_class,
        "by_status": by_status,
        "average_confidence": round(avg_conf, 4),
        "average_review_time_seconds": round(avg_review_time, 2) if avg_review_time is not None else None,
    }
