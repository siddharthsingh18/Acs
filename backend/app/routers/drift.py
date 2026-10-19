"""
backend.app.routers.drift
API endpoints for inspecting model drift signals and triggering camera drift checks.
"""
from dataclasses import asdict
from typing import Any, Dict, Optional
from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel

from backend.app.db.models import User
from backend.app.deps import require_operator, require_viewer
from backend.app.services.drift_service import drift_service

router = APIRouter(prefix="/api/drift", tags=["Model Drift"])


class DriftCheckRequest(BaseModel):
    camera_id: Optional[str] = None
    window_days: int = 7


class SetBaselineRequest(BaseModel):
    camera_id: str


@router.get("/status")
def get_drift_status(
    camera_id: Optional[str] = Query(None, description="Optional camera ID filter"),
    viewer: User = Depends(require_viewer),
) -> Dict[str, Any]:
    """Returns the latest drift evaluation reports and distribution baselines."""
    last_reports = drift_service.state.get("last_reports", {})
    baselines = drift_service.state.get("baselines", {})

    if camera_id:
        if camera_id not in last_reports:
            # Run on-demand check if no report exists
            report = drift_service.check_camera_drift(camera_id)
            return {"report": asdict(report)}
        return {
            "report": last_reports[camera_id],
            "baseline": baselines.get(camera_id),
        }

    return {
        "cameras": list(last_reports.values()),
        "baselines_count": len(baselines),
    }


@router.post("/check")
def trigger_drift_check(
    req: DriftCheckRequest,
    operator: User = Depends(require_operator),
) -> Dict[str, Any]:
    """Manually triggers weekly drift analysis for a single camera or all cameras."""
    if req.camera_id:
        report = drift_service.check_camera_drift(req.camera_id, window_days=req.window_days)
        return {"report": asdict(report)}

    reports = drift_service.check_all_cameras()
    return {
        "reports": [asdict(r) for r in reports],
        "total_checked": len(reports),
        "drift_detected_count": sum(1 for r in reports if r.drift_detected),
    }


@router.post("/baseline")
def reset_camera_baseline(
    req: SetBaselineRequest,
    operator: User = Depends(require_operator),
) -> Dict[str, Any]:
    """Recalibrates and sets the current distribution as the new baseline for a camera."""
    metrics = drift_service.set_camera_baseline(req.camera_id)
    return {
        "camera_id": req.camera_id,
        "message": "Baseline updated successfully",
        "baseline": asdict(metrics),
    }
