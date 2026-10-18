"""
backend.app.routers.edge
FastAPI router for edge-to-cloud telemetry, incident uplink, and edge node management.
"""
import json
import logging
import os
import shutil
import time
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, File, Form, Header, HTTPException, Request, UploadFile, status
from sqlmodel import Session, select

from acs.config import config
from backend.app.db.database import get_session
from backend.app.db.models import AuditLog, Incident, User
from backend.app.deps import extract_client_ip, get_current_user_optional, is_auth_enforced
from backend.app.services.audit_service import log_audit_event

logger = logging.getLogger("acs.backend.edge_router")

router = APIRouter(prefix="/api/edge", tags=["edge"])

# In-memory edge node registry for telemetry heartbeats
_EDGE_NODES_REGISTRY: Dict[str, Dict[str, Any]] = {}


def verify_edge_api_key(
    x_edge_api_key: Optional[str] = Header(None, alias="X-Edge-API-Key"),
    authorization: Optional[str] = Header(None),
) -> bool:
    """Verifies that the edge uplink request is authenticated with the configured edge secret key."""
    expected_key = os.getenv("ACS_EDGE_API_KEY", os.getenv("EDGE_API_KEY", "dev_edge_secret_key"))

    token = x_edge_api_key
    if not token and authorization:
        parts = authorization.split(" ")
        if len(parts) == 2 and parts[0].lower() in ("bearer", "apikey"):
            token = parts[1]

    if is_auth_enforced():
        if not token or token != expected_key:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid or missing edge API key for uplink transmission.",
            )
    return True


@router.post("/incidents", status_code=status.HTTP_201_CREATED)
async def receive_edge_incident(
    request: Request,
    metadata: str = Form(..., description="JSON serialized incident metadata from edge node"),
    keyframe: Optional[UploadFile] = File(None, description="Optional peak keyframe image file"),
    clip: Optional[UploadFile] = File(None, description="Optional incident video clip"),
    session: Session = Depends(get_session),
    _authorized: bool = Depends(verify_edge_api_key),
) -> Dict[str, Any]:
    """Receives collision events and media from lightweight edge nodes and persists to central storage."""
    try:
        data = json.loads(metadata)
    except Exception as parse_err:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Malformed metadata JSON payload: {parse_err}",
        )

    incident_id = data.get("incident_id") or f"INC-EDGE-{int(time.time() * 1000)}"
    camera_id = data.get("camera_id", "edge-cam")
    edge_node_id = data.get("edge_node_id", "unknown-edge-node")

    # 1. Save Keyframe Image if provided
    keyframe_dir = getattr(config, "report_images_dir", "report_images")
    os.makedirs(keyframe_dir, exist_ok=True)
    saved_keyframe_path = None
    keyframe_url = None

    if keyframe and keyframe.filename:
        safe_kf_name = f"{incident_id}_{os.path.basename(keyframe.filename)}"
        saved_keyframe_path = os.path.join(keyframe_dir, safe_kf_name)
        with open(saved_keyframe_path, "wb") as buffer:
            shutil.copyfileobj(keyframe.file, buffer)
        keyframe_url = f"/media/keyframes/{safe_kf_name}"
    else:
        # Default fallback keyframe path to satisfy database constraint
        saved_keyframe_path = os.path.join(keyframe_dir, f"{incident_id}_placeholder.jpg")
        if not os.path.exists(saved_keyframe_path):
            with open(saved_keyframe_path, "wb") as f:
                f.write(b"\xFF\xD8\xFF\xE0\x00\x10JFIF" + b"\x00" * 50)
        keyframe_url = f"/media/keyframes/{os.path.basename(saved_keyframe_path)}"

    # 2. Save Video Clip if provided
    clip_dir = getattr(config, "clips_dir", "report_clips")
    os.makedirs(clip_dir, exist_ok=True)
    saved_clip_path = None

    if clip and clip.filename:
        safe_clip_name = f"{incident_id}_{os.path.basename(clip.filename)}"
        saved_clip_path = os.path.join(clip_dir, safe_clip_name)
        with open(saved_clip_path, "wb") as buffer:
            shutil.copyfileobj(clip.file, buffer)

    # 3. Create or update Incident row in SQLite
    existing = session.exec(select(Incident).where(Incident.incident_id == incident_id)).first()
    if existing:
        inc = existing
    else:
        inc = Incident(
            incident_id=incident_id,
            camera_id=camera_id,
            start_timestamp=data.get("start_timestamp", time.strftime("%Y-%m-%d %H:%M:%S UTC")),
            end_timestamp=data.get("end_timestamp", time.strftime("%Y-%m-%d %H:%M:%S UTC")),
            class_name=data.get("class_name", "car_car_accident"),
            max_confidence=float(data.get("max_confidence", 0.85)),
            image_path=saved_keyframe_path,
            keyframe_url=keyframe_url,
            clip_path=saved_clip_path,
            severity=data.get("severity", "medium"),
            severity_score=int(data.get("severity_score", 50)),
            status="pending",
            is_privacy_blurred=data.get("is_privacy_blurred", True),
            created_at=float(data.get("created_at", time.time())),
        )
        session.add(inc)

    session.commit()
    session.refresh(inc)

    # 4. Audit Log
    client_ip = extract_client_ip(request)
    log_audit_event(
        session=session,
        username=f"edge.{edge_node_id}",
        user_role="edge_node",
        action="edge.incident.ingested",
        target_type="incident",
        target_id=incident_id,
        ip_address=client_ip,
        details={
            "camera_id": camera_id,
            "edge_node_id": edge_node_id,
            "has_keyframe": bool(keyframe),
            "has_clip": bool(clip),
            "severity": inc.severity,
        },
    )

    logger.info("Successfully ingested edge incident %s from node %s", incident_id, edge_node_id)
    return {
        "success": True,
        "incident_id": incident_id,
        "message": f"Incident '{incident_id}' received from edge node '{edge_node_id}' and persisted to central server.",
    }


@router.post("/heartbeat")
def receive_edge_heartbeat(
    telemetry: Dict[str, Any],
    request: Request,
    _authorized: bool = Depends(verify_edge_api_key),
) -> Dict[str, Any]:
    """Records real-time heartbeat and telemetry from deployed edge nodes."""
    node_id = telemetry.get("node_id", "unnamed-node")
    telemetry["last_seen"] = time.time()
    telemetry["client_ip"] = extract_client_ip(request)
    _EDGE_NODES_REGISTRY[node_id] = telemetry
    return {"status": "ok", "node_id": node_id, "timestamp": time.time()}


@router.get("/nodes")
def list_edge_nodes(
    _authorized: bool = Depends(verify_edge_api_key),
) -> List[Dict[str, Any]]:
    """Returns list of currently registered edge nodes and their telemetry."""
    now = time.time()
    nodes = []
    for node_id, data in _EDGE_NODES_REGISTRY.items():
        is_online = (now - data.get("last_seen", 0)) < 45.0
        nodes.append({
            **data,
            "is_online": is_online,
            "age_seconds": round(now - data.get("last_seen", 0), 1),
        })
    return nodes
