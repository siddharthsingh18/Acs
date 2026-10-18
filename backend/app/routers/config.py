"""
backend.app.routers.config
System configuration, retention settings, and capabilities inspection endpoints.
"""
from fastapi import APIRouter, Depends, Request
from sqlmodel import Session

from acs.clip_recorder import cleanup_expired_clips, get_clip_storage_stats
from acs.config import AppConfig, config
from backend.app.db.database import get_session
from backend.app.db.models import User
from backend.app.deps import extract_client_ip, get_config, require_admin, require_viewer
from backend.app.services.audit_service import log_audit_event
from backend.app.schemas import (
    RetentionSettingsResponse,
    RetentionSettingsUpdateRequest,
    SystemConfigResponse,
)
from backend.app.services.system_service import get_system_config

router = APIRouter(prefix="/api", tags=["Config"])


@router.get("/config", response_model=SystemConfigResponse)
def get_config_endpoint(
    cfg: AppConfig = Depends(get_config),
    viewer: User = Depends(require_viewer),
) -> SystemConfigResponse:
    """Returns available models, device capabilities, class listings, and defaults."""
    return get_system_config(cfg)


@router.get("/retention", response_model=RetentionSettingsResponse)
def get_retention_settings_endpoint(
    cfg: AppConfig = Depends(get_config),
    viewer: User = Depends(require_viewer),
) -> RetentionSettingsResponse:
    """Returns current video clip retention policies and storage usage metrics."""
    stats = get_clip_storage_stats(cfg.clips_dir)
    return RetentionSettingsResponse(
        retention_days=cfg.clip_retention_days,
        max_disk_usage_mb=cfg.clip_max_disk_usage_mb,
        current_disk_usage_mb=stats["total_mb"],
        total_clips_count=stats["total_files"],
        clips_dir=cfg.clips_dir,
    )


@router.post("/retention", response_model=RetentionSettingsResponse)
def update_retention_settings_endpoint(
    body: RetentionSettingsUpdateRequest,
    request: Request,
    cfg: AppConfig = Depends(get_config),
    session: Session = Depends(get_session),
    admin_user: User = Depends(require_admin),
) -> RetentionSettingsResponse:
    """Updates video clip retention settings and optionally triggers storage cleanup (Admin only)."""
    changes = {}
    if body.retention_days is not None:
        cfg.clip_retention_days = body.retention_days
        changes["clip_retention_days"] = body.retention_days
    if body.max_disk_usage_mb is not None:
        cfg.clip_max_disk_usage_mb = body.max_disk_usage_mb
        changes["clip_max_disk_usage_mb"] = body.max_disk_usage_mb

    if body.trigger_cleanup:
        cleanup_expired_clips(
            clips_dir=cfg.clips_dir,
            retention_days=cfg.clip_retention_days,
            max_disk_usage_mb=cfg.clip_max_disk_usage_mb,
        )
        changes["cleanup_triggered"] = True

    client_ip = extract_client_ip(request)
    log_audit_event(
        session=session,
        username=admin_user.username,
        user_role=admin_user.role,
        action="retention_settings_updated",
        target_type="settings",
        target_id="retention",
        ip_address=client_ip,
        details=changes,
        user_id=admin_user.id,
    )

    stats = get_clip_storage_stats(cfg.clips_dir)
    return RetentionSettingsResponse(
        retention_days=cfg.clip_retention_days,
        max_disk_usage_mb=cfg.clip_max_disk_usage_mb,
        current_disk_usage_mb=stats["total_mb"],
        total_clips_count=stats["total_files"],
        clips_dir=cfg.clips_dir,
    )


