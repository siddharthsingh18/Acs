"""
backend.app.routers.audit
Audit Trail Query endpoints (Admin only).
Provides immutable history of all operational and administrative system activities.
"""
from typing import Optional

from fastapi import APIRouter, Depends, Query
from sqlmodel import Session

from backend.app.db.database import get_session
from backend.app.db.models import User
from backend.app.deps import require_admin
from backend.app.schemas import AuditLogListResponse, AuditLogResponse
from backend.app.services.audit_service import get_audit_logs

router = APIRouter(prefix="/api/audit-logs", tags=["Audit Log"])


@router.get("", response_model=AuditLogListResponse)
def list_audit_logs(
    skip: int = Query(0, ge=0, description="Offset for pagination"),
    limit: int = Query(50, ge=1, le=200, description="Page limit"),
    action: Optional[str] = Query(None, description="Filter by action name"),
    target_type: Optional[str] = Query(None, description="Filter by target type"),
    username: Optional[str] = Query(None, description="Filter by actor username"),
    session: Session = Depends(get_session),
    admin_user: User = Depends(require_admin),
):
    """Retrieves immutable audit trail logs with chronological ordering and filters (Admin only)."""
    items, total = get_audit_logs(
        session=session,
        skip=skip,
        limit=limit,
        action=action,
        target_type=target_type,
        username=username,
    )

    return AuditLogListResponse(
        items=[
            AuditLogResponse(
                id=log.id,
                timestamp=log.timestamp,
                epoch=log.epoch,
                user_id=log.user_id,
                username=log.username,
                user_role=log.user_role,
                action=log.action,
                target_type=log.target_type,
                target_id=log.target_id,
                ip_address=log.ip_address,
                user_agent=log.user_agent,
                details=log.details,
            )
            for log in items
        ],
        total=total,
        skip=skip,
        limit=limit,
    )
