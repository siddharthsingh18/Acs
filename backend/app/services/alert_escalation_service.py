"""
backend.app.services.alert_escalation_service
Service for monitoring unacknowledged incidents and executing automated multi-tier escalation.
If an incident is not acknowledged within X minutes, alerts the next contact tier/channel.
"""
import asyncio
import logging
import time
from typing import Any, Dict, List, Optional
from sqlmodel import Session, or_, select

from acs.alerts import send_incident_alert
from acs.config import config
from backend.app.db.database import engine
from backend.app.db.models import Incident
from backend.app.services.audit_service import log_audit_event

logger = logging.getLogger("acs.alert_escalation")


def check_and_escalate_unacknowledged_incidents(
    session: Optional[Session] = None,
    delay_minutes_override: Optional[float] = None,
) -> List[Dict[str, Any]]:
    """Inspects pending incidents and escalates unacknowledged ones exceeding delay threshold.

    Args:
        session: Optional SQLModel Session.
        delay_minutes_override: Optional threshold override in minutes.

    Returns:
        List[Dict[str, Any]]: Summary of escalated incidents.
    """
    if not getattr(config, "escalation_enabled", True):
        return []

    delay_minutes = (
        delay_minutes_override
        if delay_minutes_override is not None
        else getattr(config, "escalation_delay_minutes", 5.0)
    )
    delay_seconds = delay_minutes * 60.0
    now = time.time()
    cutoff_time = now - delay_seconds

    def _execute(sess: Session) -> List[Dict[str, Any]]:
        stmt = select(Incident).where(
            or_(Incident.is_acknowledged == False, Incident.is_acknowledged.is_(None)),
            or_(Incident.status.is_(None), Incident.status.notin_(["rejected", "suppressed"])),
            or_(Incident.verification_status.is_(None), Incident.verification_status != "suppressed"),
            Incident.escalation_tier == 1,
            Incident.created_at <= cutoff_time,
        )
        candidates = sess.exec(stmt).all()
        escalated_results = []

        for inc in candidates:
            elapsed_minutes = round((now - inc.created_at) / 60.0, 1)
            logger.warning(
                "Incident %s unacknowledged for %.1f min (threshold: %.1f min). Escalating to Tier 2.",
                inc.incident_id,
                elapsed_minutes,
                delay_minutes,
            )

            report_summary = inc.dispatch_text or "Automated surveillance alert requires immediate human acknowledgment."
            escalation_text = (
                f"[ESCALATION NOTICE - TIER 2: UNACKNOWLEDGED FOR {int(elapsed_minutes)} MINUTES]\n"
                f"Incident {inc.incident_id} ({inc.class_name.upper()}) has not been acknowledged within policy limits.\n\n"
                f"{report_summary}"
            )

            # Tier 2 escalation channel: email and telegram
            tier2_channels = ["email", "telegram"]
            results = send_incident_alert(
                incident=inc,
                report_text=escalation_text,
                image_path=inc.image_path,
                channels=tier2_channels,
                dry_run=getattr(config, "alerts_dry_run", False),
                tier=2,
                is_escalation=True,
            )

            inc.escalation_tier = 2
            inc.last_escalated_at = now
            sess.add(inc)

            # Log audit trail
            try:
                log_audit_event(
                    session=sess,
                    username="system.escalation",
                    user_role="system",
                    action="incident_escalated",
                    target_type="incident",
                    target_id=inc.incident_id,
                    ip_address="127.0.0.1",
                    details={
                        "delay_minutes": delay_minutes,
                        "elapsed_minutes": elapsed_minutes,
                        "tier": 2,
                        "channels": tier2_channels,
                        "results": results,
                    },
                )
            except Exception as audit_err:
                logger.debug("Could not record escalation audit log: %s", audit_err)

            escalated_results.append({
                "incident_id": inc.incident_id,
                "elapsed_minutes": elapsed_minutes,
                "tier": 2,
                "delivery_results": results,
            })

        if escalated_results:
            sess.commit()
            logger.info("Successfully escalated %d unacknowledged incident(s).", len(escalated_results))

        return escalated_results

    if session:
        return _execute(session)
    else:
        with Session(engine) as sess:
            return _execute(sess)


async def run_escalation_worker(interval_seconds: float = 30.0) -> None:
    """Background async worker periodically checking unacknowledged incidents."""
    logger.info("Alert escalation background worker started (interval: %.1fs).", interval_seconds)
    while True:
        try:
            check_and_escalate_unacknowledged_incidents()
        except Exception as exc:
            logger.error("Error in alert escalation worker cycle: %s", exc)
        await asyncio.sleep(interval_seconds)
