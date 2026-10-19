"""
backend.app.routers.metrics
Prometheus metrics exposition endpoint for ACS backend.
"""
import logging
from fastapi import APIRouter, Response
from sqlmodel import Session, select, func

from backend.app.core.metrics import (
    CONTENT_TYPE_LATEST,
    get_prometheus_metrics,
    record_queue_depth,
    record_camera_fps,
    record_camera_connection,
)
from backend.app.db.database import engine
from backend.app.db.models import Incident
from backend.app.services.camera_manager import camera_manager
from backend.app.services.job_manager import job_manager

logger = logging.getLogger("acs.backend.routers.metrics")
router = APIRouter(tags=["Metrics"])


def _update_dynamic_gauges() -> None:
    """Updates dynamic queue and camera gauges right before scraping."""
    # 1. Human Review Queue depth
    try:
        with Session(engine) as session:
            pending_count = session.exec(
                select(func.count(Incident.incident_id)).where(
                    Incident.status == "pending",
                    Incident.is_reviewed == False,
                )
            ).one()
            record_queue_depth("review_queue", int(pending_count))
    except Exception as e:
        logger.debug("Failed querying review queue depth for metrics: %s", e)

    # 2. Background Video Analysis Job Queue depth
    try:
        active_jobs = sum(
            1 for j in job_manager.jobs.values()
            if j.status in ("pending", "running") and (j.thread is None or j.thread.is_alive())
        )
        record_queue_depth("job_queue", active_jobs)
    except Exception as e:
        logger.debug("Failed querying job queue depth for metrics: %s", e)

    # 3. Active Camera Worker FPS & Connectivity
    try:
        total_subscribers = 0
        with camera_manager.lock:
            for cam_id, worker in camera_manager.workers.items():
                record_camera_fps(cam_id, worker.name, worker.current_fps)
                record_camera_connection(cam_id, worker.name, worker.connected)
                total_subscribers += len(worker.subscribers)
        record_queue_depth("camera_subscribers", total_subscribers)
    except Exception as e:
        logger.debug("Failed updating camera worker metrics: %s", e)


@router.get("/metrics", response_class=Response, include_in_schema=False)
def metrics_endpoint() -> Response:
    """Exposes Prometheus text format metrics scraped by Prometheus."""
    _update_dynamic_gauges()
    content = get_prometheus_metrics()
    return Response(content=content, media_type=CONTENT_TYPE_LATEST)
