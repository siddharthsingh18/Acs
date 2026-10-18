"""
backend.app.services.audit_service
Append-only audit trail service recording sensitive operational and configuration actions.
"""
import json
import logging
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from sqlmodel import Session, desc, func, select

from backend.app.db.models import AuditLog

logger = logging.getLogger("acs.audit")


def log_audit_event(
    session: Session,
    username: str,
    user_role: str,
    action: str,
    target_type: str,
    target_id: Optional[str] = None,
    ip_address: Optional[str] = None,
    user_agent: Optional[str] = None,
    details: Optional[Dict[str, Any] | str] = None,
    user_id: Optional[str] = None,
) -> AuditLog:
    """Creates and persists an append-only audit log entry."""
    now_epoch = time.time()
    now_iso = datetime.now(timezone.utc).isoformat()

    details_str: Optional[str] = None
    if details is not None:
        if isinstance(details, dict):
            details_str = json.dumps(details, default=str)
        else:
            details_str = str(details)

    log_entry = AuditLog(
        timestamp=now_iso,
        epoch=now_epoch,
        user_id=user_id,
        username=username or "system",
        user_role=user_role or "unknown",
        action=action,
        target_type=target_type,
        target_id=target_id,
        ip_address=ip_address,
        user_agent=user_agent,
        details=details_str,
    )

    try:
        session.add(log_entry)
        session.commit()
        session.refresh(log_entry)
        logger.info(
            "AUDIT: [%s] User=%s Role=%s Action=%s Target=%s:%s IP=%s",
            now_iso,
            username,
            user_role,
            action,
            target_type,
            target_id,
            ip_address,
        )
        return log_entry
    except Exception as e:
        logger.error("Failed to commit audit log entry: %s", e)
        session.rollback()
        raise


def get_audit_logs(
    session: Session,
    skip: int = 0,
    limit: int = 50,
    action: Optional[str] = None,
    target_type: Optional[str] = None,
    username: Optional[str] = None,
) -> Tuple[List[AuditLog], int]:
    """Retrieves paginated audit log entries with optional filtering."""
    query = select(AuditLog)
    count_query = select(func.count(AuditLog.id))

    if action:
        query = query.where(AuditLog.action == action)
        count_query = count_query.where(AuditLog.action == action)
    if target_type:
        query = query.where(AuditLog.target_type == target_type)
        count_query = count_query.where(AuditLog.target_type == target_type)
    if username:
        query = query.where(AuditLog.username == username)
        count_query = count_query.where(AuditLog.username == username)

    total = session.exec(count_query).one() or 0
    query = query.order_by(desc(AuditLog.epoch)).offset(skip).limit(limit)
    items = list(session.exec(query).all())

    return items, total
