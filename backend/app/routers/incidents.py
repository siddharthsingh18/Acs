"""
backend.app.routers.incidents
Route handlers for incident management, filtering, dispatch reports, PDF, and CSV export.
"""
import json
import logging
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query, Response, status
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

    return IncidentResponse(
        incident_id=inc.incident_id,
        start_timestamp=inc.start_timestamp,
        end_timestamp=inc.end_timestamp,
        class_name=inc.class_name,
        max_confidence=inc.max_confidence,
        image_path=inc.image_path,
        keyframe_url=inc.keyframe_url,
        severity=inc.severity,
        status=inc.status,
        reviewer_name=inc.reviewer_name,
        reviewed_at=inc.reviewed_at,
        dispatch_report=parsed_dispatch,
        dispatch_text=inc.dispatch_text,
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
