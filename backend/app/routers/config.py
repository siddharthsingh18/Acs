"""
backend.app.routers.config
System configuration and capabilities inspection endpoints.
"""
from fastapi import APIRouter, Depends
from acs.config import AppConfig
from backend.app.deps import get_config
from backend.app.schemas import SystemConfigResponse
from backend.app.services.system_service import get_system_config

router = APIRouter(prefix="/api", tags=["Config"])


@router.get("/config", response_model=SystemConfigResponse)
def get_config_endpoint(
    cfg: AppConfig = Depends(get_config),
) -> SystemConfigResponse:
    """Returns available models, device capabilities, class listings, and defaults."""
    return get_system_config(cfg)
