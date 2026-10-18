"""
backend.app.services.job_manager
Thread-safe, non-blocking background video analysis job manager for ACS.

Features:
- Single active job concurrency enforcement (409 Conflict if busy).
- Background OS thread execution using acs.detector (ByteTrack + AccidentEventManager).
- Non-blocking WebSocket message broadcasting via asyncio loop.call_soon_threadsafe.
- Max 15 FPS frame transmission rate limiting for WebSockets.
- Max 960px dimension preservation and JPEG quality 70 encoding.
- Graceful cancellation and client disconnect handling.
"""
import asyncio
import base64
import logging
import os
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set

import cv2

from acs.config import config
from acs.detector import AccidentDetector, AccidentEventManager

logger = logging.getLogger("acs.backend.job_manager")

# Maximum streaming frame rate to avoid overwhelming client websocket bandwidth
MAX_STREAM_FPS = 15.0
MIN_FRAME_INTERVAL = 1.0 / MAX_STREAM_FPS
MAX_FRAME_DIMENSION = 960
JPEG_QUALITY = 70


@dataclass
class Job:
    """Represents a video analysis job state."""

    job_id: str
    video_source: str
    video_path: str
    model_name: str
    confidence: float
    use_injury_model: bool
    created_at: float
    max_frames: Optional[int] = None
    status: str = "pending"  # "pending", "running", "completed", "cancelled", "failed"
    cancel_event: threading.Event = field(default_factory=threading.Event)
    subscribers: Set[asyncio.Queue] = field(default_factory=set)
    incidents: List[dict] = field(default_factory=list)
    summary: Optional[dict] = None
    error: Optional[str] = None
    thread: Optional[threading.Thread] = None


class JobManager:
    """Singleton coordinator managing background video analysis jobs."""

    def __init__(self) -> None:
        self.jobs: Dict[str, Job] = {}
        self.active_job_id: Optional[str] = None
        self.lock = threading.Lock()
        self.loop: Optional[asyncio.AbstractEventLoop] = None

    def set_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        """Captures the active asyncio event loop for threadsafe message dispatch."""
        self.loop = loop

    def has_active_job(self) -> bool:
        """Returns True if a job is currently pending or actively running."""
        with self.lock:
            if not self.active_job_id:
                return False
            job = self.jobs.get(self.active_job_id)
            if not job:
                self.active_job_id = None
                return False
            if job.status in ("pending", "running") and job.thread and job.thread.is_alive():
                return True
            # Clean up stale active job ID
            self.active_job_id = None
            return False

    def create_job(
        self,
        video_path: str,
        video_source: str,
        model_name: Optional[str] = None,
        confidence: Optional[float] = None,
        use_injury_model: bool = False,
        max_frames: Optional[int] = None,
    ) -> Job:
        """Creates and launches a new background analysis job.

        Raises:
            RuntimeError: If another job is currently active.
            FileNotFoundError: If the specified video path does not exist.
        """
        if self.has_active_job():
            raise RuntimeError("Another video analysis job is currently running.")

        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found at: '{video_path}'")

        job_id = f"job-{uuid.uuid4().hex[:8]}"
        chosen_model = model_name or config.model_path
        chosen_conf = confidence if confidence is not None else config.confidence_threshold

        job = Job(
            job_id=job_id,
            video_source=video_source,
            video_path=video_path,
            model_name=chosen_model,
            confidence=chosen_conf,
            use_injury_model=use_injury_model,
            created_at=time.time(),
            max_frames=max_frames,
            status="pending",
        )

        with self.lock:
            self.jobs[job_id] = job
            self.active_job_id = job_id

        # Launch background OS worker thread
        worker_thread = threading.Thread(
            target=self._run_inference_worker,
            args=(job,),
            daemon=True,
            name=f"worker-{job_id}",
        )
        job.thread = worker_thread
        worker_thread.start()

        return job

    def get_job(self, job_id: str) -> Optional[Job]:
        """Retrieves job by ID."""
        return self.jobs.get(job_id)

    def cancel_job(self, job_id: str) -> bool:
        """Signals cancellation to an active job."""
        job = self.jobs.get(job_id)
        if not job:
            return False

        if job.status in ("pending", "running"):
            job.cancel_event.set()
            job.status = "cancelled"
            logger.info("Cancellation requested for job %s", job_id)
            return True
        return False

    def subscribe(self, job_id: str) -> Optional[asyncio.Queue]:
        """Registers a new WebSocket listener queue for the given job."""
        job = self.jobs.get(job_id)
        if not job:
            return None

        # Try to acquire running loop if not yet set
        if self.loop is None or not self.loop.is_running():
            try:
                self.loop = asyncio.get_running_loop()
            except RuntimeError:
                pass

        q: asyncio.Queue = asyncio.Queue()
        with self.lock:
            job.subscribers.add(q)
        return q

    def unsubscribe(self, job_id: str, q: asyncio.Queue) -> None:
        """Removes a WebSocket listener queue."""
        job = self.jobs.get(job_id)
        if job:
            with self.lock:
                job.subscribers.discard(q)

    def broadcast(self, job_id: str, message: dict) -> None:
        """Delivers message thread-safely to all active WebSocket subscriber queues."""
        job = self.jobs.get(job_id)
        if not job:
            return

        with self.lock:
            subscribers = list(job.subscribers)

        if not subscribers:
            return

        loop = self.loop
        if loop and loop.is_running():
            def _deliver():
                for queue in subscribers:
                    try:
                        queue.put_nowait(message)
                    except Exception:
                        pass

            try:
                loop.call_soon_threadsafe(_deliver)
            except RuntimeError:
                pass

    def _run_inference_worker(self, job: Job) -> None:
        """Background thread worker running YOLO tracking and AccidentEventManager."""
        job.status = "running"
        logger.info(
            "Worker started for job %s on video %s [Model: %s | Conf: %.2f]",
            job.job_id,
            job.video_path,
            job.model_name,
            job.confidence,
        )

        cap = cv2.VideoCapture(job.video_path)
        if not cap.isOpened():
            err_msg = f"Failed to open video file: {job.video_path}"
            logger.error(err_msg)
            job.status = "failed"
            job.error = err_msg
            self.broadcast(job.job_id, {"type": "error", "message": err_msg})
            with self.lock:
                if self.active_job_id == job.job_id:
                    self.active_job_id = None
            return

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
        if job.max_frames and job.max_frames < total_frames:
            total_frames = job.max_frames

        # Initialize detector and state machine
        try:
            injury_path = config.injury_model_path if job.use_injury_model else None
            detector = AccidentDetector(
                model_path=job.model_name,
                injury_model_path=injury_path,
                confidence_threshold=job.confidence,
                device=config.device,
            )
            event_manager = AccidentEventManager(
                confidence_threshold=job.confidence,
                output_dir=config.report_images_dir,
                reports_dir=config.reports_dir,
                auto_save=True,
            )
        except Exception as exc:
            logger.exception("Failed to initialize detector/event_manager for job %s: %s", job.job_id, exc)
            job.status = "failed"
            job.error = str(exc)
            self.broadcast(job.job_id, {"type": "error", "message": str(exc)})
            cap.release()
            with self.lock:
                if self.active_job_id == job.job_id:
                    self.active_job_id = None
            return

        frame_idx = 0
        start_wall_time = time.time()
        last_frame_sent_time = 0.0

        try:
            while cap.isOpened() and not job.cancel_event.is_set():
                ret, frame = cap.read()
                if not ret:
                    break

                frame_idx += 1
                timestamp = frame_idx / fps

                # Execute ByteTrack tracking inference
                result = detector.track_frame(frame)

                # Feed into incident lifecycle aggregator
                completed_incident = event_manager.update(result, frame=frame, timestamp=timestamp)
                if completed_incident:
                    # Run automated verification stage (physics + VLM)
                    try:
                        from acs.verifier import verify_incident
                        decision = verify_incident(
                            completed_incident,
                            detection_history=completed_incident.detection_history,
                            frames=completed_incident.verification_frames,
                        )
                        completed_incident.verification_status = decision.status
                        completed_incident.vlm_confidence = decision.confidence
                        completed_incident.vlm_reason = decision.reason
                        completed_incident.physics_passed = decision.physics_passed
                    except Exception as ver_err:
                        logger.warning("Verification error for incident %s: %s", completed_incident.incident_id, ver_err)
                        completed_incident.verification_status = "unverified"

                    keyframe_fname = os.path.basename(completed_incident.image_path)
                    duration_val = max(0.1, completed_incident.end_time - completed_incident.start_time)
                    clip_url = f"/api/incidents/{completed_incident.incident_id}/clip" if completed_incident.clip_path else None
                    clip_ann_url = f"/api/incidents/{completed_incident.incident_id}/clip?annotated=true" if completed_incident.clip_annotated_path else None
                    incident_msg = {
                        "type": "incident",
                        "incident_id": completed_incident.incident_id,
                        "class_name": completed_incident.class_name,
                        "max_confidence": float(completed_incident.max_confidence),
                        "start_timestamp": completed_incident.start_timestamp,
                        "end_timestamp": completed_incident.end_timestamp,
                        "start_time": completed_incident.start_time,
                        "end_time": completed_incident.end_time,
                        "duration": duration_val,
                        "keyframe_url": f"/media/keyframes/{keyframe_fname}",
                        "verification_status": completed_incident.verification_status,
                        "severity": completed_incident.severity or "medium",
                        "severity_score": completed_incident.severity_score,
                        "recommended_units": completed_incident.recommended_units,
                        "clip_url": clip_url,
                        "clip_annotated_url": clip_ann_url,
                    }
                    job.incidents.append(incident_msg)
                    self.broadcast(job.job_id, incident_msg)

                    # Persist to SQLite DB
                    try:
                        from backend.app.services.incident_service import save_incident_to_db
                        db_status = (
                            "approved" if completed_incident.verification_status == "auto_confirmed"
                            else ("rejected" if completed_incident.verification_status == "suppressed" else "pending")
                        )
                        save_incident_to_db(
                            incident_id=completed_incident.incident_id,
                            start_timestamp=completed_incident.start_timestamp,
                            end_timestamp=completed_incident.end_timestamp,
                            class_name=completed_incident.class_name,
                            max_confidence=float(completed_incident.max_confidence),
                            image_path=completed_incident.image_path,
                            duration=duration_val,
                            status=db_status,
                            verification_status=completed_incident.verification_status,
                            vlm_confidence=completed_incident.vlm_confidence,
                            vlm_reason=completed_incident.vlm_reason,
                            physics_passed=completed_incident.physics_passed,
                            clip_path=completed_incident.clip_path,
                            clip_annotated_path=completed_incident.clip_annotated_path,
                            impact_offset_seconds=completed_incident.impact_offset_seconds,
                            severity=completed_incident.severity,
                            severity_score=completed_incident.severity_score,
                            severity_breakdown=completed_incident.severity_breakdown,
                            recommended_units=completed_incident.recommended_units,
                        )
                    except Exception as db_err:
                        logger.warning("Failed to persist incident to DB: %s", db_err)

                # Throttle frame broadcasts to MAX_STREAM_FPS (15 fps)
                now = time.time()
                is_final_frame = frame_idx == total_frames
                if now - last_frame_sent_time >= MIN_FRAME_INTERVAL or is_final_frame:
                    last_frame_sent_time = now

                    # Resize frame to max 960px preserving aspect ratio
                    h, w = result.annotated_frame.shape[:2]
                    max_dim = max(h, w)
                    if max_dim > MAX_FRAME_DIMENSION:
                        scale = MAX_FRAME_DIMENSION / float(max_dim)
                        disp_w, disp_h = int(w * scale), int(h * scale)
                        disp_frame = cv2.resize(result.annotated_frame, (disp_w, disp_h))
                    else:
                        disp_frame = result.annotated_frame

                    # Encode to JPEG with quality 70
                    _, buffer = cv2.imencode(
                        ".jpg", disp_frame, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY]
                    )
                    b64_str = base64.b64encode(buffer).decode("utf-8")
                    elapsed = max(0.001, now - start_wall_time)
                    current_fps = round(frame_idx / elapsed, 1)

                    frame_msg = {
                        "type": "frame",
                        "index": frame_idx,
                        "total": total_frames,
                        "fps": current_fps,
                        "jpeg_base64": b64_str,
                    }
                    self.broadcast(job.job_id, frame_msg)

                if job.max_frames and frame_idx >= job.max_frames:
                    logger.info("Job %s hit max_frames limit (%d)", job.job_id, job.max_frames)
                    break

            # Handle active incident closure on stream end if not cancelled
            if not job.cancel_event.is_set():
                final_incident = event_manager.end_active_incident(timestamp=frame_idx / fps)
                if final_incident:
                    # Run automated verification stage (physics + VLM)
                    try:
                        from acs.verifier import verify_incident
                        decision = verify_incident(
                            final_incident,
                            detection_history=final_incident.detection_history,
                            frames=final_incident.verification_frames,
                        )
                        final_incident.verification_status = decision.status
                        final_incident.vlm_confidence = decision.confidence
                        final_incident.vlm_reason = decision.reason
                        final_incident.physics_passed = decision.physics_passed
                    except Exception as ver_err:
                        logger.warning("Verification error for final incident %s: %s", final_incident.incident_id, ver_err)
                        final_incident.verification_status = "unverified"

                    keyframe_fname = os.path.basename(final_incident.image_path)
                    duration_val = max(0.1, final_incident.end_time - final_incident.start_time)
                    incident_msg = {
                        "type": "incident",
                        "incident_id": final_incident.incident_id,
                        "class_name": final_incident.class_name,
                        "max_confidence": float(final_incident.max_confidence),
                        "start_timestamp": final_incident.start_timestamp,
                        "end_timestamp": final_incident.end_timestamp,
                        "start_time": final_incident.start_time,
                        "end_time": final_incident.end_time,
                        "duration": duration_val,
                        "keyframe_url": f"/media/keyframes/{keyframe_fname}",
                        "verification_status": final_incident.verification_status,
                        "severity": final_incident.severity or "medium",
                        "severity_score": final_incident.severity_score,
                        "recommended_units": final_incident.recommended_units,
                    }
                    job.incidents.append(incident_msg)
                    self.broadcast(job.job_id, incident_msg)

                    try:
                        from backend.app.services.incident_service import save_incident_to_db
                        db_status = (
                            "approved" if final_incident.verification_status == "auto_confirmed"
                            else ("rejected" if final_incident.verification_status == "suppressed" else "pending")
                        )
                        save_incident_to_db(
                            incident_id=final_incident.incident_id,
                            start_timestamp=final_incident.start_timestamp,
                            end_timestamp=final_incident.end_timestamp,
                            class_name=final_incident.class_name,
                            max_confidence=float(final_incident.max_confidence),
                            image_path=final_incident.image_path,
                            duration=duration_val,
                            status=db_status,
                            verification_status=final_incident.verification_status,
                            vlm_confidence=final_incident.vlm_confidence,
                            vlm_reason=final_incident.vlm_reason,
                            physics_passed=final_incident.physics_passed,
                            clip_path=final_incident.clip_path,
                            clip_annotated_path=final_incident.clip_annotated_path,
                            impact_offset_seconds=final_incident.impact_offset_seconds,
                            severity=final_incident.severity,
                            severity_score=final_incident.severity_score,
                            severity_breakdown=final_incident.severity_breakdown,
                            recommended_units=final_incident.recommended_units,
                        )
                    except Exception as db_err:
                        logger.warning("Failed to persist final incident to DB: %s", db_err)

        except Exception as exc:
            logger.exception("Inference loop failure on job %s: %s", job.job_id, exc)
            job.status = "failed"
            job.error = str(exc)
            self.broadcast(job.job_id, {"type": "error", "message": str(exc)})
            return
        finally:
            cap.release()
            total_elapsed = max(0.001, time.time() - start_wall_time)
            calc_fps = round(frame_idx / total_elapsed, 2)

            final_status = "cancelled" if job.cancel_event.is_set() else "completed"
            job.status = final_status

            summary = {
                "job_id": job.job_id,
                "status": final_status,
                "frames_processed": frame_idx,
                "total_frames": total_frames,
                "fps": calc_fps,
                "incidents_count": len(job.incidents),
                "elapsed_seconds": round(total_elapsed, 2),
            }
            job.summary = summary
            logger.info("Job %s ended with status '%s' [Summary: %s]", job.job_id, final_status, summary)
            self.broadcast(job.job_id, {"type": "done", "summary": summary})

            with self.lock:
                if self.active_job_id == job.job_id:
                    self.active_job_id = None


# Global job manager singleton instance
job_manager = JobManager()
