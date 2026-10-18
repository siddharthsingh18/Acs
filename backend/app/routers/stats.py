"""
backend.app.routers.stats
Route handler for aggregated incident analytics and KPIs.
"""
from fastapi import APIRouter, Depends
from sqlmodel import Session

from backend.app.db.database import get_session
from backend.app.db.models import User
from backend.app.deps import require_viewer
from backend.app.schemas import StatsResponse
from backend.app.services.incident_service import get_incident_stats

router = APIRouter(prefix="/api/stats", tags=["Stats"])


@router.get("", response_model=StatsResponse)
def get_stats_endpoint(
    session: Session = Depends(get_session),
    viewer: User = Depends(require_viewer),
) -> StatsResponse:
    """Returns aggregated incident analytics (incidents today, class distribution, confidence, review time)."""
    stats_data = get_incident_stats(session=session)
    return StatsResponse(**stats_data)
