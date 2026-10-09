"""
backend.app.routers.health
Health check endpoint.
"""
import datetime
from fastapi import APIRouter
from backend.app.schemas import HealthResponse

router = APIRouter(prefix="/api", tags=["Health"])


@router.get("/health", response_model=HealthResponse)
def get_health() -> HealthResponse:
    """Returns application health status and UTC timestamp."""
    utc_now = datetime.datetime.now(datetime.timezone.utc).isoformat()
    return HealthResponse(
        status="ok",
        service="acs-backend",
        version="1.0.0",
        timestamp=utc_now,
    )
