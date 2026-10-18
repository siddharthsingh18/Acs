"""
backend.app.services.retention_service
Incident retention policy enforcement, automatic expiration cleanup,
individual incident ZIP bundle export, and full incident erasure.
"""
import glob
import io
import json
import logging
import os
import shutil
import time
import zipfile
from typing import Any, Dict, List, Optional, Tuple

from sqlmodel import Session, select

from acs.clip_recorder import cleanup_expired_clips
from acs.config import config
from acs.reporter import generate_report_pdf
from backend.app.db.models import AuditLog, Incident, ReviewFeedback

logger = logging.getLogger("acs.retention_service")


def run_retention_cleanup(
    session: Session,
    retention_days: Optional[int] = None,
    max_disk_usage_mb: Optional[int] = None,
) -> Dict[str, int]:
    """Enforces retention policy across database records and media files.

    Purges incidents older than `retention_days` and caps total media storage.

    Args:
        session: Active SQLModel session.
        retention_days: Maximum age in days before purging (defaults to config).
        max_disk_usage_mb: Maximum disk size cap in MB.

    Returns:
        Dict detailing purged incidents, clips, keyframes, and freed bytes.
    """
    days = retention_days if retention_days is not None else getattr(config, "incident_retention_days", 30)
    now = time.time()
    cutoff_time = now - (days * 86400)

    # 1. Purge expired clips via clip_recorder
    clip_stats = cleanup_expired_clips(
        clips_dir=getattr(config, "clips_dir", "report_clips"),
        retention_days=days,
        max_disk_usage_mb=max_disk_usage_mb,
    )

    # 2. Find expired incidents in DB
    statement = select(Incident).where(Incident.created_at < cutoff_time)
    expired_incidents = session.exec(statement).all()

    deleted_incidents_count = 0
    deleted_media_files = 0
    freed_media_bytes = 0

    for inc in expired_incidents:
        # Delete associated media files
        file_candidates = [
            inc.image_path,
            inc.clip_path,
            inc.clip_annotated_path,
            getattr(inc, "unredacted_keyframe_path", None),
            getattr(inc, "unredacted_clip_path", None),
        ]
        # Verification frame patterns
        if inc.image_path:
            img_dir = os.path.dirname(inc.image_path)
            file_candidates.extend(glob.glob(os.path.join(img_dir, f"{inc.incident_id}_*")))

        for path in file_candidates:
            if path and os.path.exists(path):
                try:
                    freed_media_bytes += os.path.getsize(path)
                    os.remove(path)
                    deleted_media_files += 1
                except OSError as err:
                    logger.warning("Failed to remove expired file %s: %s", path, err)

        # Delete database records
        # Associated review feedback
        fb_statement = select(ReviewFeedback).where(ReviewFeedback.incident_id == inc.incident_id)
        for fb in session.exec(fb_statement).all():
            session.delete(fb)

        session.delete(inc)
        deleted_incidents_count += 1

    session.commit()

    return {
        "deleted_incidents": deleted_incidents_count,
        "deleted_clips": clip_stats.get("deleted_count", 0),
        "deleted_media_files": deleted_media_files,
        "freed_bytes": freed_media_bytes + clip_stats.get("freed_bytes", 0),
        "retention_days": days,
    }


def export_incident_bundle(session: Session, incident_id: str) -> Tuple[bytes, str]:
    """Compiles a complete forensic export bundle as a ZIP archive for an incident.

    Contains:
    - metadata.json (Incident attributes, severity breakdown, dispatch details)
    - keyframe.jpg (Privacy-blurred peak keyframe)
    - clip.mp4 (Raw or annotated incident video clip, if available)
    - report.pdf (Official summary PDF document)
    - audit_trail.json (Immutable audit logs referencing this incident)

    Args:
        session: Active SQLModel session.
        incident_id: Target incident ID.

    Returns:
        Tuple of (zip_bytes, filename).
    """
    statement = select(Incident).where(Incident.incident_id == incident_id)
    inc = session.exec(statement).first()
    if not inc:
        raise ValueError(f"Incident with ID '{incident_id}' not found.")

    zip_buffer = io.BytesIO()

    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        # 1. metadata.json
        meta_dict = {
            "incident_id": inc.incident_id,
            "camera_id": getattr(inc, "camera_id", None),
            "start_timestamp": inc.start_timestamp,
            "end_timestamp": inc.end_timestamp,
            "class_name": inc.class_name,
            "max_confidence": inc.max_confidence,
            "severity": inc.severity,
            "severity_score": inc.severity_score,
            "severity_breakdown": json.loads(inc.severity_breakdown) if inc.severity_breakdown else None,
            "recommended_units": json.loads(inc.recommended_units) if inc.recommended_units else None,
            "status": inc.status,
            "reviewer_name": inc.reviewer_name,
            "reviewed_at": inc.reviewed_at,
            "dispatch_report": json.loads(inc.dispatch_report) if inc.dispatch_report else None,
            "dispatch_text": inc.dispatch_text,
            "verification_status": inc.verification_status,
            "vlm_confidence": inc.vlm_confidence,
            "vlm_reason": inc.vlm_reason,
            "physics_passed": inc.physics_passed,
            "is_privacy_blurred": getattr(inc, "is_privacy_blurred", True),
            "created_at": inc.created_at,
            "duration": inc.duration,
            "exported_at": time.strftime("%Y-%m-%d %H:%M:%S UTC"),
        }
        zf.writestr("metadata.json", json.dumps(meta_dict, indent=2))

        # 2. Keyframe image (privacy protected)
        if inc.image_path and os.path.exists(inc.image_path):
            zf.write(inc.image_path, arcname="keyframe.jpg")

        # 3. Incident Video Clip
        clip_path = inc.clip_annotated_path if (inc.clip_annotated_path and os.path.exists(inc.clip_annotated_path)) else inc.clip_path
        if clip_path and os.path.exists(clip_path):
            zf.write(clip_path, arcname="clip.mp4")

        # 4. PDF summary report
        try:
            pdf_bytes = generate_report_pdf(
                incident_id=inc.incident_id,
                class_name=inc.class_name,
                start_timestamp=inc.start_timestamp,
                end_timestamp=inc.end_timestamp,
                max_confidence=inc.max_confidence,
                image_path=inc.image_path,
                dispatch_text=inc.dispatch_text,
                lang="en",
            )
            zf.writestr(f"{inc.incident_id}_report.pdf", pdf_bytes)
        except Exception as pdf_err:
            logger.warning("Failed to bundle PDF report for export: %s", pdf_err)

        # 5. Audit Trail
        audit_stmt = select(AuditLog).where(AuditLog.target_id == incident_id)
        audit_logs = session.exec(audit_stmt).all()
        audit_list = [
            {
                "timestamp": a.timestamp,
                "created_at": a.created_at,
                "username": a.username,
                "role": a.role,
                "action": a.action,
                "target_type": a.target_type,
                "target_id": a.target_id,
                "ip_address": a.ip_address,
                "details": json.loads(a.details_json) if a.details_json else None,
            }
            for a in audit_logs
        ]
        zf.writestr("audit_trail.json", json.dumps(audit_list, indent=2))

    zip_bytes = zip_buffer.getvalue()
    filename = f"{incident_id}_export.zip"
    return zip_bytes, filename


def delete_incident_and_media(session: Session, incident_id: str) -> Dict[str, Any]:
    """Deletes an incident row, associated review feedback, and all disk media.

    Args:
        session: Active SQLModel session.
        incident_id: Target incident ID.

    Returns:
        Dict confirmation summary.
    """
    statement = select(Incident).where(Incident.incident_id == incident_id)
    inc = session.exec(statement).first()
    if not inc:
        raise ValueError(f"Incident with ID '{incident_id}' not found.")

    deleted_files = 0
    file_candidates = [
        inc.image_path,
        inc.clip_path,
        inc.clip_annotated_path,
        getattr(inc, "unredacted_keyframe_path", None),
        getattr(inc, "unredacted_clip_path", None),
    ]

    # Include keyframe directory artifacts (e.g. verification frames)
    if inc.image_path:
        img_dir = os.path.dirname(inc.image_path)
        file_candidates.extend(glob.glob(os.path.join(img_dir, f"{incident_id}_*")))

    # Include unredacted vault artifacts if present
    vault_dir = getattr(config, "privacy_unredacted_dir", "media/unredacted")
    file_candidates.extend(glob.glob(os.path.join(vault_dir, "**", f"{incident_id}_*"), recursive=True))

    for path in file_candidates:
        if path and os.path.exists(path) and os.path.isfile(path):
            try:
                os.remove(path)
                deleted_files += 1
            except OSError as err:
                logger.warning("Failed to remove media file %s: %s", path, err)

    # Delete ReviewFeedback
    fb_statement = select(ReviewFeedback).where(ReviewFeedback.incident_id == incident_id)
    for fb in session.exec(fb_statement).all():
        session.delete(fb)

    # Delete Incident
    session.delete(inc)
    session.commit()

    return {
        "success": True,
        "incident_id": incident_id,
        "deleted_files_count": deleted_files,
        "message": f"Incident '{incident_id}' and all associated media files have been permanently deleted.",
    }
