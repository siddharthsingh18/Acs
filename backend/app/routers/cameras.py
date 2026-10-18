"""
backend.app.routers.cameras
FastAPI router providing full CRUD and WebSocket live streaming for CCTV cameras.

Endpoints:
- GET /api/cameras: List all cameras with masked URLs and live health metrics.
- POST /api/cameras: Register a new camera (encrypts credentials).
- GET /api/cameras/{id}: Retrieve a single camera record and telemetry.
- PUT /api/cameras/{id}: Update camera configuration or toggle state.
- DELETE /api/cameras/{id}: Remove a camera and stop its worker.
- GET /api/cameras/{id}/health: Direct health telemetry check.
- GET /api/cameras/{id}/stream: WebSocket streaming real-time annotated frames and incidents.
"""
import asyncio
import logging
import time
import uuid
from typing import List

from fastapi import APIRouter, Depends, HTTPException, Request, WebSocket, WebSocketDisconnect, status
from sqlmodel import Session, select

from backend.app.db.database import get_session
from backend.app.db.models import Camera, User
from backend.app.deps import extract_client_ip, require_admin, require_viewer
from backend.app.services.audit_service import log_audit_event
from backend.app.schemas import (
    CameraCreateRequest,
    CameraHealth,
    CameraResponse,
    CameraUpdateRequest,
)
from backend.app.services.camera_manager import camera_manager
from backend.app.services.camera_source import (
    create_camera_source,
    decrypt_credentials,
    encrypt_credentials,
    mask_credentials,
)

logger = logging.getLogger("acs.backend.routers.cameras")

router = APIRouter(prefix="/api/cameras", tags=["Live Cameras"])


def _to_camera_response(cam: Camera) -> CameraResponse:
    """Helper formatting a Camera DB record into a CameraResponse with masked credentials."""
    decrypted = decrypt_credentials(cam.source_url_encrypted)
    masked = mask_credentials(decrypted)
    health = camera_manager.get_health(cam.id) if cam.enabled else None

    # Detect source type if not populated
    source_type = cam.source_type
    if not source_type:
        dummy_source = create_camera_source(decrypted)
        source_type = dummy_source.source_type

    return CameraResponse(
        id=cam.id,
        name=cam.name,
        source_type=source_type,
        source_url=masked,
        location_lat=cam.location_lat,
        location_lng=cam.location_lng,
        enabled=cam.enabled,
        confidence_override=cam.confidence_override,
        created_at=cam.created_at,
        updated_at=cam.updated_at,
        health=health,
    )




@router.get("", response_model=List[CameraResponse])
def list_cameras(
    session: Session = Depends(get_session),
    viewer: User = Depends(require_viewer),
) -> List[CameraResponse]:
    """Retrieves all registered CCTV cameras with masked credentials and live health."""
    cameras = session.exec(select(Camera).order_by(Camera.created_at.desc())).all()
    return [_to_camera_response(c) for c in cameras]


@router.post("", response_model=CameraResponse, status_code=status.HTTP_201_CREATED)
def create_camera(
    payload: CameraCreateRequest,
    request: Request,
    session: Session = Depends(get_session),
    admin_user: User = Depends(require_admin),
) -> CameraResponse:
    """Registers a new CCTV camera feed (Admin only).

    Credentials in RTSP/HTTP URLs are securely encrypted at rest.
    """
    raw_source = payload.source_url.strip()
    if not raw_source:
        raise HTTPException(status_code=400, detail="Source URL or device index is required.")

    # Determine source type
    sample_source = create_camera_source(raw_source, name=payload.name)
    source_type = sample_source.source_type

    cam_id = f"cam-{uuid.uuid4().hex[:8]}"
    now = time.time()

    encrypted_source = encrypt_credentials(raw_source)

    cam = Camera(
        id=cam_id,
        name=payload.name.strip(),
        source_type=source_type,
        source_url_encrypted=encrypted_source,
        location_lat=payload.location_lat,
        location_lng=payload.location_lng,
        enabled=payload.enabled if payload.enabled is not None else True,
        confidence_override=payload.confidence_override,
        created_at=now,
        updated_at=now,
    )
    session.add(cam)
    session.commit()
    session.refresh(cam)

    # Start dedicated worker if enabled
    if cam.enabled:
        camera_manager.start_worker_for_camera(cam)

    client_ip = extract_client_ip(request)
    log_audit_event(
        session=session,
        username=admin_user.username,
        user_role=admin_user.role,
        action="camera_created",
        target_type="camera",
        target_id=cam.id,
        ip_address=client_ip,
        details={"name": cam.name, "source_type": cam.source_type},
        user_id=admin_user.id,
    )

    logger.info("Registered new camera '%s' (%s)", cam.name, cam.id)
    return _to_camera_response(cam)


@router.get("/{camera_id}", response_model=CameraResponse)
def get_camera(
    camera_id: str,
    session: Session = Depends(get_session),
    viewer: User = Depends(require_viewer),
) -> CameraResponse:
    """Retrieves single camera metadata and live health status."""
    cam = session.get(Camera, camera_id)
    if not cam:
        raise HTTPException(status_code=404, detail=f"Camera '{camera_id}' not found.")
    return _to_camera_response(cam)


@router.put("/{camera_id}", response_model=CameraResponse)
def update_camera(
    camera_id: str,
    payload: CameraUpdateRequest,
    request: Request,
    session: Session = Depends(get_session),
    admin_user: User = Depends(require_admin),
) -> CameraResponse:
    """Updates camera parameters, toggles worker state, or alters confidence threshold (Admin only)."""
    cam = session.get(Camera, camera_id)
    if not cam:
        raise HTTPException(status_code=404, detail=f"Camera '{camera_id}' not found.")

    source_changed = False
    if payload.name is not None:
        cam.name = payload.name.strip()
    if payload.location_lat is not None:
        cam.location_lat = payload.location_lat
    if payload.location_lng is not None:
        cam.location_lng = payload.location_lng
    if payload.confidence_override is not None:
        cam.confidence_override = payload.confidence_override

    if payload.source_url is not None and payload.source_url.strip():
        raw_source = payload.source_url.strip()
        cam.source_url_encrypted = encrypt_credentials(raw_source)
        sample_source = create_camera_source(raw_source, name=cam.name)
        cam.source_type = sample_source.source_type
        source_changed = True

    if payload.enabled is not None:
        cam.enabled = payload.enabled

    cam.updated_at = time.time()
    session.add(cam)
    session.commit()
    session.refresh(cam)

    # Restart or stop worker according to updated state
    if not cam.enabled:
        camera_manager.stop_worker(cam.id)
    else:
        camera_manager.start_worker_for_camera(cam)

    client_ip = extract_client_ip(request)
    log_audit_event(
        session=session,
        username=admin_user.username,
        user_role=admin_user.role,
        action="camera_updated",
        target_type="camera",
        target_id=cam.id,
        ip_address=client_ip,
        details={"name": cam.name, "enabled": cam.enabled},
        user_id=admin_user.id,
    )

    logger.info("Updated camera '%s' (%s)", cam.name, cam.id)
    return _to_camera_response(cam)


@router.delete("/{camera_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_camera(
    camera_id: str,
    request: Request,
    session: Session = Depends(get_session),
    admin_user: User = Depends(require_admin),
) -> None:
    """Deletes camera registration and terminates its dedicated background worker (Admin only)."""
    cam = session.get(Camera, camera_id)
    if not cam:
        raise HTTPException(status_code=404, detail=f"Camera '{camera_id}' not found.")

    camera_manager.stop_worker(camera_id)
    session.delete(cam)
    session.commit()

    client_ip = extract_client_ip(request)
    log_audit_event(
        session=session,
        username=admin_user.username,
        user_role=admin_user.role,
        action="camera_deleted",
        target_type="camera",
        target_id=camera_id,
        ip_address=client_ip,
        details={"name": cam.name},
        user_id=admin_user.id,
    )

    logger.info("Deleted camera '%s' (%s)", cam.name, camera_id)



@router.get("/{camera_id}/health", response_model=CameraHealth)
def get_camera_health(
    camera_id: str,
    session: Session = Depends(get_session),
    viewer: User = Depends(require_viewer),
) -> CameraHealth:
    """Returns direct health indicators (connected, fps, last frame age, dropped frames)."""
    cam = session.get(Camera, camera_id)
    if not cam:
        raise HTTPException(status_code=404, detail=f"Camera '{camera_id}' not found.")
    return camera_manager.get_health(camera_id)


# -----------------------------------------------------------------------------
# Real-Time WebSocket Streaming Endpoint
# -----------------------------------------------------------------------------
@router.websocket("/{camera_id}/stream")
async def camera_stream_websocket(websocket: WebSocket, camera_id: str):
    """Real-time WebSocket delivering live annotated JPEG frames, incidents, and health telemetry.

    Frame-skip policy ensures that client disconnects or network lag never queues
    frames.
    """
    await websocket.accept()

    # Create subscriber queue with buffer size 2 to enforce frame-skip
    subscriber_queue: asyncio.Queue = asyncio.Queue(maxsize=2)
    subscribed = camera_manager.subscribe(camera_id, subscriber_queue)

    if not subscribed:
        # Camera worker may not be running or camera disabled
        health = camera_manager.get_health(camera_id)
        await websocket.send_json(
            {
                "type": "status",
                "camera_id": camera_id,
                "connected": False,
                "message": "Camera worker is offline or starting.",
                "health": health.model_dump(),
            }
        )

    logger.info("Client connected to stream for camera '%s'", camera_id)
    try:
        while True:
            # Await message with heartbeat timeout
            try:
                msg = await asyncio.wait_for(subscriber_queue.get(), timeout=2.0)
                await websocket.send_json(msg)
            except asyncio.TimeoutError:
                # Periodic health heartbeat when stream is idle or waiting
                health = camera_manager.get_health(camera_id)
                await websocket.send_json(
                    {
                        "type": "heartbeat",
                        "camera_id": camera_id,
                        "health": health.model_dump(),
                    }
                )
    except WebSocketDisconnect:
        logger.info("Client disconnected from stream for camera '%s'", camera_id)
    except Exception as exc:
        logger.warning("Stream error on camera '%s': %s", camera_id, exc)
    finally:
        camera_manager.unsubscribe(camera_id, subscriber_queue)
