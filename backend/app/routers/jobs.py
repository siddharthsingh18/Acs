"""
backend.app.routers.jobs
Route handlers for video analysis jobs:
- POST /api/jobs: Create job (multipart upload or JSON {"source": "sample"})
- GET /api/jobs/{id}/stream: WebSocket stream for annotated frames, incidents, and completion
- POST /api/jobs/{id}/cancel: Cancel active job
- GET /api/jobs/{id}: Inspect job status and summary
"""
import logging
import os
import shutil
from typing import Optional

import subprocess
from fastapi import (
    APIRouter,
    Depends,
    HTTPException,
    Request,
    WebSocket,
    WebSocketDisconnect,
    status,
)
from sqlmodel import Session

from acs.config import config
from backend.app.db.database import get_session
from backend.app.db.models import User
from backend.app.deps import extract_client_ip, require_operator, require_viewer
from backend.app.services.audit_service import log_audit_event
from backend.app.schemas import (
    JobCancelResponse,
    JobCreatedResponse,
    JobStatusResponse,
)
from backend.app.services.job_manager import job_manager

logger = logging.getLogger("acs.backend.routers.jobs")
router = APIRouter(prefix="/api/jobs", tags=["Jobs"])

MAX_UPLOAD_SIZE = 500 * 1024 * 1024  # 500 MB max video upload size


def is_valid_video_magic_bytes(header: bytes) -> bool:
    """Checks header bytes against known signatures for MP4, MOV, AVI, MKV, WebM, and MPEG-TS."""
    if len(header) < 8:
        return False

    # MP4 / MOV / ISO Base Media: box type at offset 4..8
    box_type = header[4:8]
    if box_type in (b"ftyp", b"moov", b"wide", b"mdat", b"free", b"skip"):
        return True

    # AVI: starts with RIFF and has AVI in bytes 8..12
    if header.startswith(b"RIFF") and b"AVI " in header[:16]:
        return True

    # Matroska / WebM: starts with \x1a\x45\xdf\xa3
    if header.startswith(b"\x1a\x45\xdf\xa3"):
        return True

    # MPEG Transport Stream (sync byte 0x47)
    if header[0] == 0x47:
        return True

    return False


def validate_video_file(file_path: str) -> None:
    """Validates container magic bytes, followed by ffprobe / cv2 decode sanity check."""
    # 1. Magic bytes check
    try:
        with open(file_path, "rb") as f:
            header = f.read(64)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unable to read uploaded file: {str(e)}",
        )

    if not is_valid_video_magic_bytes(header):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid video format: magic bytes header does not match a supported video container (MP4, AVI, MKV/WebM).",
        )

    # 2. Container decode sanity check (ffprobe if available, otherwise OpenCV)
    ffprobe_path = shutil.which("ffprobe")
    if ffprobe_path:
        try:
            res = subprocess.run(
                [
                    ffprobe_path,
                    "-v",
                    "error",
                    "-select_streams",
                    "v:0",
                    "-show_entries",
                    "stream=codec_type",
                    "-of",
                    "default=noprint_wrappers=1:nokey=1",
                    file_path,
                ],
                capture_output=True,
                text=True,
                timeout=8,
            )
            if res.returncode != 0 or "video" not in res.stdout.lower():
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Corrupted or invalid video: ffprobe could not detect a valid video stream.",
                )
        except HTTPException:
            raise
        except Exception as e:
            logger.warning("ffprobe check encountered an error (%s), falling back to OpenCV check", e)

    # Secondary / Fallback sanity check using OpenCV VideoCapture
    try:
        import cv2
        cap = cv2.VideoCapture(file_path)
        if not cap.isOpened():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Corrupted video file: unable to initialize video decoder.",
            )
        ret, frame = cap.read()
        cap.release()
        if not ret or frame is None or frame.size == 0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Corrupted video file: decoder failed to extract initial video frame.",
            )
    except HTTPException:
        raise
    except Exception as cv_err:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Video decode verification error: {str(cv_err)}",
        )


@router.post("", response_model=JobCreatedResponse, status_code=status.HTTP_201_CREATED)
async def create_job_endpoint(
    request: Request,
    session: Session = Depends(get_session),
    operator_user: User = Depends(require_operator),
) -> JobCreatedResponse:
    """Creates and starts a background video analysis job (Operator or Admin only).

    Enforces:
    - 500 MB max upload size limit
    - Magic bytes container verification
    - ffprobe / cv2 initial frame decode sanity check
    - Append-only audit logging
    """
    if job_manager.has_active_job():
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="A video analysis job is currently running. Please wait or cancel the active job.",
        )

    content_type = request.headers.get("content-type", "").lower()
    source: Optional[str] = None
    model_name: Optional[str] = None
    confidence: Optional[float] = None
    use_injury_model: bool = False
    max_frames: Optional[int] = None
    uploaded_file_path: Optional[str] = None
    video_source_label: str = "custom_upload"

    if "multipart/form-data" in content_type:
        form = await request.form()
        source = form.get("source")  # e.g. "sample"
        model_name = form.get("model_name")
        conf_raw = form.get("confidence")
        if conf_raw is not None:
            try:
                confidence = float(conf_raw)
            except ValueError:
                pass
        uim_raw = form.get("use_injury_model")
        if uim_raw is not None:
            use_injury_model = str(uim_raw).lower() in ("true", "1", "yes")

        mf_raw = form.get("max_frames")
        if mf_raw is not None:
            try:
                max_frames = int(mf_raw)
            except ValueError:
                pass

        file_obj = form.get("file")
        if file_obj is not None and hasattr(file_obj, "filename") and file_obj.filename:
            # Save uploaded video file into temp directory with size limit check
            config.ensure_directories()
            safe_filename = os.path.basename(file_obj.filename)
            temp_path = os.path.join(config.temp_dir, f"upload_{safe_filename}")

            total_written = 0
            try:
                with open(temp_path, "wb") as buffer:
                    while True:
                        chunk = file_obj.file.read(1024 * 1024)
                        if not chunk:
                            break
                        total_written += len(chunk)
                        if total_written > MAX_UPLOAD_SIZE:
                            raise HTTPException(
                                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                                detail=f"File exceeds maximum allowed upload size of 500 MB (read {total_written} bytes).",
                            )
                        buffer.write(chunk)
            except Exception:
                if os.path.exists(temp_path):
                    try:
                        os.remove(temp_path)
                    except OSError:
                        pass
                raise

            # Validate magic bytes and sanity check
            try:
                validate_video_file(temp_path)
            except Exception:
                if os.path.exists(temp_path):
                    try:
                        os.remove(temp_path)
                    except OSError:
                        pass
                raise

            uploaded_file_path = temp_path
            video_source_label = safe_filename
    else:
        # Parse JSON body
        try:
            body = await request.json()
        except Exception:
            body = {}

        source = body.get("source")
        model_name = body.get("model_name")
        confidence = body.get("confidence")
        use_injury_model = bool(body.get("use_injury_model", False))
        max_frames = body.get("max_frames")

    # Determine final video path
    final_video_path: Optional[str] = None
    if source == "sample":
        sample_path = "acc_ex.mp4"
        if not os.path.exists(sample_path):
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Sample video clip '{sample_path}' was not found in the repository root.",
            )
        final_video_path = sample_path
        video_source_label = "sample (acc_ex.mp4)"
    elif uploaded_file_path:
        final_video_path = uploaded_file_path
    else:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Must provide an uploaded video file or set source='sample'.",
        )

    # Launch background job
    try:
        job = job_manager.create_job(
            video_path=final_video_path,
            video_source=video_source_label,
            model_name=model_name,
            confidence=confidence,
            use_injury_model=use_injury_model,
            max_frames=max_frames,
        )
    except RuntimeError as r_err:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(r_err))
    except Exception as exc:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc))

    # Audit log job creation
    client_ip = extract_client_ip(request)
    log_audit_event(
        session=session,
        username=operator_user.username,
        user_role=operator_user.role,
        action="job_started",
        target_type="job",
        target_id=job.job_id,
        ip_address=client_ip,
        details={
            "source": video_source_label,
            "model_name": model_name,
            "confidence": confidence,
        },
        user_id=operator_user.id,
    )

    return JobCreatedResponse(

        job_id=job.job_id,
        status=job.status,
        video_source=job.video_source,
        model_name=job.model_name,
        confidence=job.confidence,
    )


@router.websocket("/{job_id}/stream")
async def websocket_job_stream(websocket: WebSocket, job_id: str) -> None:
    """WebSocket stream for real-time video frames, incidents, and completion notifications.

    Sends JSON messages conforming to:
    - {type: 'frame', index, total, fps, jpeg_base64}
    - {type: 'incident', incident_id, class_name, max_confidence, keyframe_url, ...}
    - {type: 'done', summary}
    - {type: 'error', message}
    """
    job = job_manager.get_job(job_id)
    if not job:
        await websocket.close(code=1008, reason="Job not found")
        return

    await websocket.accept()
    queue = job_manager.subscribe(job_id)
    if queue is None:
        await websocket.close(code=1008, reason="Failed to subscribe to job")
        return

    try:
        # Send historical incidents to late-connecting clients
        for past_inc in job.incidents:
            await websocket.send_json(past_inc)

        # If job already finished before connection, emit done/error immediately
        if job.status in ("completed", "cancelled") and job.summary:
            await websocket.send_json({"type": "done", "summary": job.summary})
            return
        elif job.status == "failed":
            await websocket.send_json({"type": "error", "message": job.error or "Job failed"})
            return

        # Stream real-time events as they arrive
        while True:
            message = await queue.get()
            await websocket.send_json(message)
            if message.get("type") in ("done", "error"):
                break

    except WebSocketDisconnect:
        logger.info("Client cleanly disconnected from WebSocket stream for job %s", job_id)
    except Exception as exc:
        logger.warning("WebSocket streaming terminated for job %s: %s", job_id, exc)
    finally:
        job_manager.unsubscribe(job_id, queue)
        try:
            await websocket.close()
        except Exception:
            pass


@router.post("/{job_id}/cancel", response_model=JobCancelResponse)
def cancel_job_endpoint(
    job_id: str,
    request: Request,
    session: Session = Depends(get_session),
    operator_user: User = Depends(require_operator),
) -> JobCancelResponse:
    """Cancels a currently executing video analysis job (Operator or Admin only)."""
    job = job_manager.get_job(job_id)
    if not job:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Job with ID '{job_id}' not found.",
        )

    success = job_manager.cancel_job(job_id)
    if success:
        client_ip = extract_client_ip(request)
        log_audit_event(
            session=session,
            username=operator_user.username,
            user_role=operator_user.role,
            action="job_cancelled",
            target_type="job",
            target_id=job_id,
            ip_address=client_ip,
            user_id=operator_user.id,
        )
        return JobCancelResponse(
            job_id=job_id,
            status="cancelled",
            message=f"Job {job_id} cancelled successfully.",
        )
    return JobCancelResponse(
        job_id=job_id,
        status=job.status,
        message=f"Job {job_id} is already in state '{job.status}'.",
    )


@router.get("/{job_id}", response_model=JobStatusResponse)
def get_job_status_endpoint(
    job_id: str,
    viewer: User = Depends(require_viewer),
) -> JobStatusResponse:
    """Retrieves current execution status, incidents, and summary for a job."""
    job = job_manager.get_job(job_id)
    if not job:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Job with ID '{job_id}' not found.",
        )

    return JobStatusResponse(
        job_id=job.job_id,
        status=job.status,
        created_at=job.created_at,
        video_source=job.video_source,
        model_name=job.model_name,
        confidence=job.confidence,
        incidents=job.incidents,
        summary=job.summary,
        error=job.error,
    )

