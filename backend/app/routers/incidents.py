"""
backend.app.routers.incidents
Route handlers for incident management, filtering, dispatch reports, PDF, video clips, and CSV export.
"""
import json
import logging
import os
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query, Request, Response, status
from fastapi.responses import StreamingResponse
from sqlmodel import Session

from acs.config import config
from acs.reporter import generate_report_pdf
from backend.app.db.database import get_session
from backend.app.db.models import Incident
from backend.app.schemas import (
    IncidentListResponse,
    IncidentResponse,
    IncidentStatusUpdateRequest,
)
from backend.app.services.incident_service import (
    export_incidents_csv,
    get_incident_by_id,
    get_incidents,
    get_or_generate_dispatch_report,
    update_incident_status,
)

logger = logging.getLogger("acs.backend.routers.incidents")
router = APIRouter(prefix="/api/incidents", tags=["Incidents"])


def _to_incident_response(inc: Incident) -> IncidentResponse:
    """Helper formatting Incident SQLModel row into IncidentResponse schema."""
    parsed_dispatch = None
    if inc.dispatch_report:
        try:
            parsed_dispatch = json.loads(inc.dispatch_report)
        except Exception:
            pass

    parsed_breakdown = None
    if inc.severity_breakdown:
        try:
            parsed_breakdown = json.loads(inc.severity_breakdown)
        except Exception:
            pass

    parsed_units = None
    if inc.recommended_units:
        try:
            parsed_units = json.loads(inc.recommended_units)
        except Exception:
            pass

    clip_url = (
        f"/api/incidents/{inc.incident_id}/clip"
        if inc.clip_path and os.path.exists(inc.clip_path)
        else None
    )
    clip_ann_url = (
        f"/api/incidents/{inc.incident_id}/clip?annotated=true"
        if inc.clip_annotated_path and os.path.exists(inc.clip_annotated_path)
        else None
    )

    return IncidentResponse(
        incident_id=inc.incident_id,
        camera_id=getattr(inc, "camera_id", None),
        start_timestamp=inc.start_timestamp,
        end_timestamp=inc.end_timestamp,
        class_name=inc.class_name,
        max_confidence=inc.max_confidence,
        image_path=inc.image_path,
        keyframe_url=inc.keyframe_url,
        severity=inc.severity,
        severity_score=inc.severity_score if inc.severity_score is not None else 50,
        severity_breakdown=parsed_breakdown,
        recommended_units=parsed_units,
        status=inc.status,
        reviewer_name=inc.reviewer_name,
        reviewed_at=inc.reviewed_at,
        dispatch_report=parsed_dispatch,
        dispatch_text=inc.dispatch_text,
        verification_status=inc.verification_status,
        vlm_confidence=inc.vlm_confidence,
        vlm_reason=inc.vlm_reason,
        physics_passed=inc.physics_passed,
        clip_path=inc.clip_path,
        clip_url=clip_url,
        clip_annotated_path=inc.clip_annotated_path,
        clip_annotated_url=clip_ann_url,
        impact_offset_seconds=inc.impact_offset_seconds if inc.impact_offset_seconds is not None else 8.0,
        reviewed_label=inc.reviewed_label,
        corrected_class=inc.corrected_class,
        corrected_boxes=inc.corrected_boxes,
        created_at=inc.created_at,
        duration=inc.duration,
        is_privacy_blurred=getattr(inc, "is_privacy_blurred", True),
        has_unredacted_original=bool(
            getattr(inc, "unredacted_keyframe_path", None)
            or getattr(inc, "unredacted_clip_path", None)
            or os.path.exists(os.path.join(getattr(config, "privacy_unredacted_dir", "media/unredacted"), "keyframes", f"{inc.incident_id}_unredacted.jpg"))
            or os.path.exists(os.path.join(getattr(config, "privacy_unredacted_dir", "media/unredacted"), "clips", f"{inc.incident_id}_unredacted.mp4"))
        ),
    )


@router.get("", response_model=IncidentListResponse)
def list_incidents_endpoint(
    page: int = Query(1, ge=1, description="1-based page number"),
    page_size: int = Query(20, ge=1, le=100, description="Items per page"),
    class_name: Optional[str] = Query(None, alias="class", description="Filter by collision class"),
    start_date: Optional[str] = Query(None, description="Filter incidents starting on/after date"),
    end_date: Optional[str] = Query(None, description="Filter incidents starting on/before date"),
    severity: Optional[str] = Query(None, description="Filter by severity ('low', 'medium', 'high')"),
    status: Optional[str] = Query(None, description="Filter by review status ('pending', 'approved', 'rejected', 'dispatched')"),
    sort_by: str = Query("created_at", description="Field to sort by ('created_at', 'start_timestamp', 'max_confidence', 'class_name')"),
    order: str = Query("desc", pattern="^(asc|desc)$", description="Sort order ('asc' or 'desc')"),
    session: Session = Depends(get_session),
) -> IncidentListResponse:
    """Returns paginated, filtered, and sorted incidents list."""
    items, total = get_incidents(
        session=session,
        page=page,
        page_size=page_size,
        class_name=class_name,
        start_date=start_date,
        end_date=end_date,
        severity=severity,
        status=status,
        sort_by=sort_by,
        order=order,
    )

    total_pages = max(1, (total + page_size - 1) // page_size) if total > 0 else 1
    return IncidentListResponse(
        items=[_to_incident_response(item) for item in items],
        total=total,
        page=page,
        page_size=page_size,
        total_pages=total_pages,
    )


@router.get("/export.csv")
def export_incidents_csv_endpoint(
    session: Session = Depends(get_session),
) -> Response:
    """Exports all incidents as a downloadable CSV file."""
    csv_content = export_incidents_csv(session=session)
    return Response(
        content=csv_content,
        media_type="text/csv",
        headers={"Content-Disposition": 'attachment; filename="incidents_export.csv"'},
    )


@router.get("/{incident_id}", response_model=IncidentResponse)
def get_incident_endpoint(
    incident_id: str,
    session: Session = Depends(get_session),
) -> IncidentResponse:
    """Retrieves full details for a single incident by ID."""
    inc = get_incident_by_id(session, incident_id)
    if not inc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Incident with ID '{incident_id}' not found.",
        )
    return _to_incident_response(inc)


@router.post("/{incident_id}/dispatch-report")
def generate_dispatch_report_endpoint(
    incident_id: str,
    lang: str = Query("en", description="Target report language ('en' or 'ar')"),
    session: Session = Depends(get_session),
) -> dict:
    """Generates structured Gemini dispatch report and caches it in SQLite."""
    inc = get_incident_by_id(session, incident_id)
    if not inc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Incident with ID '{incident_id}' not found.",
        )

    report_data = get_or_generate_dispatch_report(session, incident_id=incident_id, lang=lang)
    if not report_data:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to generate dispatch report.",
        )
    return report_data


@router.get("/{incident_id}/pdf")
def get_incident_pdf_endpoint(
    incident_id: str,
    lang: str = Query("en", description="Language for generated PDF ('en' or 'ar')"),
    session: Session = Depends(get_session),
) -> Response:
    """Generates and returns an official PDF incident report."""
    inc = get_incident_by_id(session, incident_id)
    if not inc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Incident with ID '{incident_id}' not found.",
        )

    # Generate PDF bytes
    pdf_bytes = generate_report_pdf(
        incident_id=inc.incident_id,
        class_name=inc.class_name,
        start_timestamp=inc.start_timestamp,
        end_timestamp=inc.end_timestamp,
        max_confidence=inc.max_confidence,
        image_path=inc.image_path,
        dispatch_text=inc.dispatch_text,
        lang=lang,
    )

    filename = f"{inc.incident_id}_report.pdf"
    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={"Content-Disposition": f'inline; filename="{filename}"'},
    )


from backend.app.db.models import Incident, User
from backend.app.deps import extract_client_ip, require_admin, require_operator, require_viewer
from backend.app.services.audit_service import log_audit_event


@router.post("/{incident_id}/status", response_model=IncidentResponse)
def update_incident_status_endpoint(
    incident_id: str,
    body: IncidentStatusUpdateRequest,
    request: Request,
    session: Session = Depends(get_session),
    operator_user: User = Depends(require_operator),
) -> IncidentResponse:
    """Updates human-in-the-loop review status (approved, rejected, or dispatched).

    Enforces operator role and replaces free-text reviewer name with authenticated operator identity.
    """
    from backend.app.deps import is_auth_enforced
    if is_auth_enforced() or not body.reviewer_name:
        authenticated_reviewer = operator_user.full_name or operator_user.username
    else:
        authenticated_reviewer = body.reviewer_name

    inc = update_incident_status(
        session=session,
        incident_id=incident_id,
        new_status=body.status,
        reviewer_name=authenticated_reviewer,
        notes=body.notes,
        corrected_class=body.corrected_class,
        corrected_boxes=body.corrected_boxes,
    )

    if not inc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Incident with ID '{incident_id}' not found.",
        )

    # Append-only audit trail logging
    client_ip = extract_client_ip(request)
    log_audit_event(
        session=session,
        username=operator_user.username,
        user_role=operator_user.role,
        action=f"incident_{body.status}",
        target_type="incident",
        target_id=incident_id,
        ip_address=client_ip,
        details={
            "status": body.status,
            "reviewer": authenticated_reviewer,
            "notes": body.notes,
            "corrected_class": body.corrected_class,
        },
        user_id=operator_user.id,
    )

    return _to_incident_response(inc)



def serve_file_with_range_support(file_path: str, request: Optional[Request] = None) -> Response:
    """Streams a video file supporting RFC 7233 HTTP Byte Range requests for smooth scrubbing."""
    if not os.path.isfile(file_path):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Video clip file not found on disk.",
        )

    file_size = os.path.getsize(file_path)

    # HEAD request handling for browser media player probing
    if request and request.method == "HEAD":
        return Response(
            status_code=status.HTTP_200_OK,
            headers={
                "Accept-Ranges": "bytes",
                "Content-Length": str(file_size),
                "Content-Type": "video/mp4",
            },
        )

    range_header = request.headers.get("Range") if request else None

    if not range_header:
        def full_iter():
            with open(file_path, "rb") as f:
                while chunk := f.read(64 * 1024):
                    yield chunk

        return StreamingResponse(
            full_iter(),
            status_code=status.HTTP_200_OK,
            media_type="video/mp4",
            headers={
                "Accept-Ranges": "bytes",
                "Content-Length": str(file_size),
                "Content-Type": "video/mp4",
            },
        )

    try:
        range_str = range_header.strip()
        if not range_str.startswith("bytes="):
            raise ValueError("Invalid range unit")

        range_val = range_str[len("bytes="):].strip()
        parts = range_val.split("-")

        if parts[0] == "":
            suffix_len = int(parts[1])
            start = max(0, file_size - suffix_len)
            end = file_size - 1
        elif len(parts) > 1 and parts[1] != "":
            start = int(parts[0])
            end = min(file_size - 1, int(parts[1]))
        else:
            start = int(parts[0])
            end = file_size - 1

        if start > end or start >= file_size:
            return Response(
                status_code=status.HTTP_416_RANGE_NOT_SATISFIABLE,
                headers={
                    "Content-Range": f"bytes */{file_size}",
                    "Accept-Ranges": "bytes",
                },
            )

        chunk_size = end - start + 1

        def chunk_iter():
            with open(file_path, "rb") as f:
                f.seek(start)
                bytes_left = chunk_size
                while bytes_left > 0:
                    read_len = min(64 * 1024, bytes_left)
                    data = f.read(read_len)
                    if not data:
                        break
                    bytes_left -= len(data)
                    yield data

        return StreamingResponse(
            chunk_iter(),
            status_code=status.HTTP_206_PARTIAL_CONTENT,
            media_type="video/mp4",
            headers={
                "Content-Range": f"bytes {start}-{end}/{file_size}",
                "Accept-Ranges": "bytes",
                "Content-Length": str(chunk_size),
                "Content-Type": "video/mp4",
            },
        )
    except Exception:
        return Response(
            status_code=status.HTTP_416_REQUESTED_RANGE_NOT_SATISFIABLE,
            headers={
                "Content-Range": f"bytes */{file_size}",
                "Accept-Ranges": "bytes",
            },
        )


@router.get("/{incident_id}/clip")
@router.head("/{incident_id}/clip")
def get_incident_clip_endpoint(
    incident_id: str,
    request: Request,
    annotated: bool = Query(False, description="Whether to stream the annotated clip with bounding boxes burned in"),
    session: Session = Depends(get_session),
) -> Response:
    """Streams the raw or annotated H.264 MP4 clip with full HTTP Range request support."""
    inc = get_incident_by_id(session, incident_id)
    if not inc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Incident with ID '{incident_id}' not found.",
        )

    file_path = None
    if annotated and inc.clip_annotated_path and os.path.isfile(inc.clip_annotated_path):
        file_path = inc.clip_annotated_path
    elif inc.clip_path and os.path.isfile(inc.clip_path):
        file_path = inc.clip_path
    elif inc.clip_annotated_path and os.path.isfile(inc.clip_annotated_path):
        file_path = inc.clip_annotated_path

    if not file_path or not os.path.isfile(file_path):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No video clip available on disk for incident '{incident_id}'.",
        )

    return serve_file_with_range_support(file_path, request)


@router.get("/{incident_id}/export")
def export_incident_bundle_endpoint(
    incident_id: str,
    request: Request,
    session: Session = Depends(get_session),
    current_user: User = Depends(require_operator),
) -> Response:
    """Exports a specific incident bundle as a ZIP file including metadata, blurred media, PDF report, and audit logs."""
    from backend.app.services.retention_service import export_incident_bundle
    try:
        zip_bytes, filename = export_incident_bundle(session, incident_id)
        log_audit_event(
            session=session,
            username=current_user.username,
            user_role=current_user.role,
            action="incident.exported",
            target_type="incident",
            target_id=incident_id,
            ip_address=extract_client_ip(request),
            details={"filename": filename, "size_bytes": len(zip_bytes)},
            user_id=current_user.id,
        )
        return Response(
            content=zip_bytes,
            media_type="application/zip",
            headers={"Content-Disposition": f'attachment; filename="{filename}"'},
        )
    except ValueError as val_err:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(val_err))
    except Exception as exc:
        logger.error("Failed to export incident bundle for %s: %s", incident_id, exc)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Export failed: {exc}")


@router.delete("/{incident_id}")
def delete_incident_endpoint(
    incident_id: str,
    request: Request,
    session: Session = Depends(get_session),
    admin_user: User = Depends(require_admin),
) -> dict:
    """Permanently deletes an incident, associated review data, and all disk media. Enforces Admin role."""
    from backend.app.services.retention_service import delete_incident_and_media
    try:
        result = delete_incident_and_media(session, incident_id)
        log_audit_event(
            session=session,
            username=admin_user.username,
            user_role=admin_user.role,
            action="incident.deleted",
            target_type="incident",
            target_id=incident_id,
            ip_address=extract_client_ip(request),
            details={"deleted_files_count": result.get("deleted_files_count", 0)},
            user_id=admin_user.id,
        )
        return result
    except ValueError as val_err:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(val_err))
    except Exception as exc:
        logger.error("Failed to delete incident %s: %s", incident_id, exc)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Deletion failed: {exc}")


@router.get("/{incident_id}/unredacted-keyframe")
def get_unredacted_keyframe_endpoint(
    incident_id: str,
    request: Request,
    session: Session = Depends(get_session),
    admin_user: User = Depends(require_admin),
) -> Response:
    """Serves the unredacted original keyframe from the access-controlled vault. Enforces Admin role."""
    from fastapi.responses import FileResponse
    from acs.config import config
    inc = get_incident_by_id(session, incident_id)
    if not inc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Incident '{incident_id}' not found.")

    vault_dir = getattr(config, "privacy_unredacted_dir", "media/unredacted")
    path = getattr(inc, "unredacted_keyframe_path", None) or os.path.join(vault_dir, "keyframes", f"{incident_id}_unredacted.jpg")

    if not path or not os.path.isfile(path):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No unredacted original keyframe stored for incident '{incident_id}'.",
        )

    log_audit_event(
        session=session,
        username=admin_user.username,
        user_role=admin_user.role,
        action="privacy.unredacted_keyframe.accessed",
        target_type="incident",
        target_id=incident_id,
        ip_address=extract_client_ip(request),
        details={"path": path},
        user_id=admin_user.id,
    )
    return FileResponse(path, media_type="image/jpeg", filename=f"{incident_id}_unredacted.jpg")


@router.get("/{incident_id}/unredacted-clip")
def get_unredacted_clip_endpoint(
    incident_id: str,
    request: Request,
    session: Session = Depends(get_session),
    admin_user: User = Depends(require_admin),
) -> Response:
    """Serves the unredacted original video clip from the access-controlled vault. Enforces Admin role."""
    from acs.config import config
    inc = get_incident_by_id(session, incident_id)
    if not inc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Incident '{incident_id}' not found.")

    vault_dir = getattr(config, "privacy_unredacted_dir", "media/unredacted")
    path = getattr(inc, "unredacted_clip_path", None) or os.path.join(vault_dir, "clips", f"{incident_id}_unredacted.mp4")

    if not path or not os.path.isfile(path):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No unredacted original clip stored for incident '{incident_id}'.",
        )

    log_audit_event(
        session=session,
        username=admin_user.username,
        user_role=admin_user.role,
        action="privacy.unredacted_clip.accessed",
        target_type="incident",
        target_id=incident_id,
        ip_address=extract_client_ip(request),
        details={"path": path},
        user_id=admin_user.id,
    )
    return serve_file_with_range_support(path, request)


@router.post("/retention/cleanup")
def run_retention_cleanup_endpoint(
    request: Request,
    retention_days: Optional[int] = Query(None, ge=1, description="Override retention days"),
    max_disk_mb: Optional[int] = Query(None, ge=10, description="Override max disk usage in MB"),
    session: Session = Depends(get_session),
    admin_user: User = Depends(require_admin),
) -> dict:
    """Executes automatic retention cleanup policy. Enforces Admin role."""
    from backend.app.services.retention_service import run_retention_cleanup
    stats = run_retention_cleanup(session, retention_days=retention_days, max_disk_usage_mb=max_disk_mb)
    log_audit_event(
        session=session,
        username=admin_user.username,
        user_role=admin_user.role,
        action="retention.cleanup.executed",
        target_type="system",
        target_id="retention_policy",
        ip_address=extract_client_ip(request),
        details=stats,
        user_id=admin_user.id,
    )
    return stats


