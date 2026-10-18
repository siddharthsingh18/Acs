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
        created_at=inc.created_at,
        duration=inc.duration,
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


@router.post("/{incident_id}/status", response_model=IncidentResponse)
def update_incident_status_endpoint(
    incident_id: str,
    body: IncidentStatusUpdateRequest,
    session: Session = Depends(get_session),
) -> IncidentResponse:
    """Updates human-in-the-loop review status (approved, rejected, or dispatched)."""
    inc = update_incident_status(
        session=session,
        incident_id=incident_id,
        new_status=body.status,
        reviewer_name=body.reviewer_name,
    )
    if not inc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Incident with ID '{incident_id}' not found.",
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

