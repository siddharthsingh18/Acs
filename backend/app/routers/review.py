"""
backend.app.routers.review
Fast swipe-style review queue API for triage of uncertain detections and feedback collection.
"""
import logging

from fastapi import APIRouter, Depends, HTTPException, Query, Request, status
from sqlmodel import Session

from backend.app.db.database import get_session
from backend.app.db.models import User
from backend.app.deps import extract_client_ip, require_operator, require_viewer
from backend.app.services.audit_service import log_audit_event
from backend.app.schemas import (
    IncidentResponse,
    ReviewDecisionRequest,
    ReviewFeedbackResponse,
    ReviewQueueItem,
    ReviewQueueResponse,
    ReviewStatsResponse,
)
from backend.app.services.incident_service import (
    get_incident_by_id,
    get_review_queue_items,
    get_review_stats,
    record_review_decision,
    undo_review_decision,
)
from backend.app.routers.incidents import _to_incident_response

logger = logging.getLogger("acs.backend.routers.review")
router = APIRouter(prefix="/api/review-queue", tags=["Review Queue"])


@router.get("", response_model=ReviewQueueResponse)
def get_review_queue_endpoint(
    include_reviewed: bool = Query(
        False, description="Whether to include already reviewed items"
    ),
    session: Session = Depends(get_session),
    viewer: User = Depends(require_viewer),
) -> ReviewQueueResponse:
    """Retrieves uncertain detections (confidence 0.3-0.6 or YOLO/VLM disagreement) for review."""
    raw_items = get_review_queue_items(session=session, include_reviewed=include_reviewed)
    items = [ReviewQueueItem(**it) for it in raw_items]
    return ReviewQueueResponse(items=items, total=len(items))


@router.get("/stats", response_model=ReviewStatsResponse)
def get_review_stats_endpoint(
    session: Session = Depends(get_session),
    viewer: User = Depends(require_viewer),
) -> ReviewStatsResponse:
    """Returns analytics on operator decisions and training dataset staging readiness."""
    stats = get_review_stats(session=session)
    return ReviewStatsResponse(**stats)


@router.post("/{incident_id}/decision", response_model=ReviewFeedbackResponse)
def submit_review_decision_endpoint(
    incident_id: str,
    payload: ReviewDecisionRequest,
    request: Request,
    session: Session = Depends(get_session),
    operator_user: User = Depends(require_operator),
) -> ReviewFeedbackResponse:
    """Submits a fast review decision: Accident (approved), Not accident (hard negative), or Wrong class.

    Enforces operator role, overwrites reviewer name with logged-in user, and logs audit trail.
    """
    inc = get_incident_by_id(session, incident_id)
    if not inc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Incident with ID '{incident_id}' not found.",
        )

    from backend.app.deps import is_auth_enforced
    if is_auth_enforced() or not payload.reviewer_name:
        authenticated_reviewer = operator_user.full_name or operator_user.username
    else:
        authenticated_reviewer = payload.reviewer_name

    feedback = record_review_decision(
        session=session,
        incident_id=incident_id,
        decision=payload.decision,
        corrected_class=payload.corrected_class,
        corrected_boxes=payload.corrected_boxes,
        reviewer_name=authenticated_reviewer,
        notes=payload.notes,
    )
    if not feedback:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to record review decision feedback.",
        )

    client_ip = extract_client_ip(request)
    log_audit_event(
        session=session,
        username=operator_user.username,
        user_role=operator_user.role,
        action="review_decision",
        target_type="incident",
        target_id=incident_id,
        ip_address=client_ip,
        details={
            "decision": payload.decision,
            "reviewer": authenticated_reviewer,
            "corrected_class": payload.corrected_class,
        },
        user_id=operator_user.id,
    )

    return ReviewFeedbackResponse(
        id=feedback.id,
        incident_id=feedback.incident_id,
        decision=feedback.decision,
        original_class=feedback.original_class,
        corrected_class=feedback.corrected_class,
        confidence=feedback.confidence,
        image_path=feedback.image_path,
        clip_path=feedback.clip_path,
        clip_annotated_path=feedback.clip_annotated_path,
        reviewer_name=feedback.reviewer_name,
        reviewed_at=feedback.reviewed_at,
        notes=feedback.notes,
        corrected_boxes=feedback.corrected_boxes,
        is_exported=feedback.is_exported,
        exported_at=feedback.exported_at,
        created_at=feedback.created_at,
    )


@router.post("/{incident_id}/undo", response_model=IncidentResponse)
def undo_review_decision_endpoint(
    incident_id: str,
    request: Request,
    session: Session = Depends(get_session),
    operator_user: User = Depends(require_operator),
) -> IncidentResponse:
    """Reverts an incident's review status back to pending and removes unexported feedback."""
    inc = undo_review_decision(session=session, incident_id=incident_id)
    if not inc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Incident with ID '{incident_id}' not found.",
        )

    client_ip = extract_client_ip(request)
    log_audit_event(
        session=session,
        username=operator_user.username,
        user_role=operator_user.role,
        action="review_undo",
        target_type="incident",
        target_id=incident_id,
        ip_address=client_ip,
        details={"status_reverted_to": "pending"},
        user_id=operator_user.id,
    )

    return _to_incident_response(inc)

