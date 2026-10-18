"""
backend.app.services.camera_manager
High-throughput, multi-camera manager and isolated worker orchestration for ACS.

Key Architecture Requirements:
1. One dedicated worker thread per camera.
2. Complete state isolation: each camera worker has its own dedicated AccidentDetector
   and AccidentEventManager, guaranteeing that ByteTrack track IDs, debounce counters,
   cooldown timers, and incident lifecycles never cross-contaminate.
3. Auto-reconnect with exponential backoff on stream drops.
4. Frame-skip policy under load: drops stale frames and prevents client queue bloat.
5. Per-camera health telemetry: connected, fps, last frame age, dropped frames, incident status.
6. Safe secret handling: RTSP/HTTP credentials are encrypted in DB and masked in responses/logs.
"""
import asyncio
import base64
import logging
import os
import threading
import time
from collections import deque
from typing import Dict, List, Optional, Set, Tuple

import cv2
import numpy as np
from sqlmodel import Session, select

from acs.config import config
from acs.detector import AccidentDetector, AccidentEventManager, AccidentIncident, DetectionResult
from backend.app.db.database import engine
from backend.app.db.models import Camera, Incident
from backend.app.schemas import CameraHealth
from backend.app.services.camera_source import (
    CameraSource,
    create_camera_source,
    decrypt_credentials,
    encrypt_credentials,
    mask_credentials,
)

logger = logging.getLogger("acs.backend.camera_manager")

# Maximum frame broadcast rate to avoid saturating WebSocket connections
MAX_BROADCAST_FPS = 15.0
MIN_BROADCAST_INTERVAL = 1.0 / MAX_BROADCAST_FPS
FRAME_JPEG_QUALITY = 70
MAX_FRAME_DIMENSION = 960


class CameraWorker:
    """Dedicated worker thread managing capture, inference, tracking, and incident detection

    for a single camera.
    """

    def __init__(
        self,
        camera_id: str,
        name: str,
        decrypted_source: str,
        confidence_override: Optional[float] = None,
        loop: Optional[asyncio.AbstractEventLoop] = None,
    ):
        self.camera_id = camera_id
        self.name = name
        self.raw_source = decrypted_source
        self.masked_source = mask_credentials(decrypted_source)
        self.confidence_override = confidence_override
        self.loop = loop

        # Threading controls
        self.thread: Optional[threading.Thread] = None
        self.stop_event = threading.Event()
        self.subscribers: Set[asyncio.Queue] = set()
        self.subscribers_lock = threading.Lock()

        # Dedicated Isolated Detector & Event Manager
        # Custom confidence if provided; else system default
        acc_conf = (
            confidence_override
            if confidence_override is not None
            else getattr(config, "accident_confidence_threshold", config.confidence_threshold)
        )
        worker_device = os.getenv("CAMERA_WORKER_DEVICE", "cpu")
        self.detector = AccidentDetector(
            confidence_threshold=acc_conf,
            accident_confidence_threshold=acc_conf,
            normal_confidence_threshold=getattr(config, "normal_confidence_threshold", 0.35),
            device=worker_device,
        )
        self.event_manager = AccidentEventManager(
            accident_confidence_threshold=acc_conf,
            output_dir=config.report_images_dir,
            reports_dir=config.reports_dir,
            csv_filename="incidents.csv",
            auto_save=True,
        )

        # Health & Telemetry State
        self.connected: bool = False
        self.last_frame_time: float = 0.0
        self.dropped_frames: int = 0
        self.reconnect_count: int = 0
        self.current_fps: float = 0.0
        self._fps_timestamps: deque = deque(maxlen=30)
        self.active_incident_id: Optional[str] = None
        self.last_accident_class: Optional[str] = None
        self.last_accident_confidence: Optional[float] = None

    def start(self) -> None:
        """Launches the dedicated worker thread."""
        if self.thread and self.thread.is_alive():
            return
        self.stop_event.clear()
        self.thread = threading.Thread(
            target=self._run_loop,
            name=f"CameraWorker-{self.camera_id}",
            daemon=True,
        )
        self.thread.start()
        logger.info("Started CameraWorker for camera '%s' (%s)", self.name, self.camera_id)

    def stop(self) -> None:
        """Signals the worker thread to terminate and waits for join."""
        self.stop_event.set()
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=3.0)
        self.connected = False
        logger.info("Stopped CameraWorker for camera '%s' (%s)", self.name, self.camera_id)

    def get_health(self) -> CameraHealth:
        """Returns snapshot of current camera health metrics."""
        now = time.time()
        age = (now - self.last_frame_time) if self.last_frame_time > 0 else 999.0
        return CameraHealth(
            connected=self.connected,
            fps=round(self.current_fps, 1),
            last_frame_age=round(age, 2),
            dropped_frames=self.dropped_frames,
            reconnect_count=self.reconnect_count,
            is_incident_active=self.event_manager.is_incident_active,
            active_incident_id=self.event_manager.current_incident_id,
            last_accident_class=self.last_accident_class,
            last_accident_confidence=self.last_accident_confidence,
        )

    def subscribe(self, queue: asyncio.Queue) -> None:
        """Registers a WebSocket subscriber queue."""
        with self.subscribers_lock:
            self.subscribers.add(queue)

    def unsubscribe(self, queue: asyncio.Queue) -> None:
        """Unregisters a WebSocket subscriber queue."""
        with self.subscribers_lock:
            self.subscribers.discard(queue)

    def _broadcast_message(self, message: dict) -> None:
        """Threadsafe broadcast of a message to all connected WebSocket subscribers.

        Implements frame-skip policy: drops frames for slow subscribers rather than
        queueing.
        """
        if not self.loop or self.loop.is_closed():
            return

        with self.subscribers_lock:
            subs = list(self.subscribers)

        if not subs:
            return

        def _dispatch():
            for q in subs:
                try:
                    if q.full():
                        # Frame-skip policy under load: drop oldest or drop current
                        try:
                            q.get_nowait()
                        except Exception:
                            pass
                        self.dropped_frames += 1
                    q.put_nowait(message)
                except Exception:
                    self.dropped_frames += 1

        try:
            self.loop.call_soon_threadsafe(_dispatch)
        except RuntimeError:
            pass

    def _persist_incident_to_db(self, incident: AccidentIncident) -> None:
        """Stores a finalized accident incident in SQLite database."""
        try:
            with Session(engine) as session:
                existing = session.get(Incident, incident.incident_id)
                if not existing:
                    keyframe_url = f"/media/keyframes/{os.path.basename(incident.image_path)}"
                    db_status = (
                        "approved" if incident.verification_status == "auto_confirmed"
                        else ("rejected" if incident.verification_status == "suppressed" else "pending")
                    )
                    db_inc = Incident(
                        incident_id=incident.incident_id,
                        start_timestamp=incident.start_timestamp,
                        end_timestamp=incident.end_timestamp,
                        class_name=incident.class_name,
                        max_confidence=incident.max_confidence,
                        image_path=incident.image_path,
                        keyframe_url=keyframe_url,
                        severity=incident.vlm_severity or ("high" if incident.max_confidence >= 0.70 else "medium"),
                        status=db_status,
                        verification_status=incident.verification_status,
                        vlm_confidence=incident.vlm_confidence,
                        vlm_reason=incident.vlm_reason,
                        physics_passed=incident.physics_passed,
                        clip_path=incident.clip_path,
                        clip_annotated_path=incident.clip_annotated_path,
                        impact_offset_seconds=incident.impact_offset_seconds,
                        created_at=incident.start_time,
                        duration=max(0.0, incident.end_time - incident.start_time),
                    )
                    session.add(db_inc)
                    session.commit()
                    logger.info("Persisted incident %s from camera '%s' to DB (status: %s, clip: %s)", incident.incident_id, self.name, incident.verification_status, bool(incident.clip_path))
        except Exception as exc:
            logger.error("Failed to persist camera incident to DB: %s", exc)

    def _run_loop(self) -> None:
        """Main worker loop with auto-reconnect backoff and isolated tracking inference."""
        backoff_delay = 1.0
        max_backoff = 30.0
        last_broadcast_time = 0.0

        while not self.stop_event.is_set():
            source = create_camera_source(self.raw_source, name=self.name, loop=True)
            logger.info("Attempting connection to camera '%s' source: %s", self.name, self.masked_source)

            opened = source.open()
            if not opened:
                self.connected = False
                self.reconnect_count += 1
                logger.warning(
                    "Camera '%s' failed to connect. Auto-reconnecting in %.1fs (attempt #%d)...",
                    self.name,
                    backoff_delay,
                    self.reconnect_count,
                )
                # Exponential backoff sleep with responsive stop_event checks
                sleep_slices = int(backoff_delay / 0.2)
                for _ in range(max(1, sleep_slices)):
                    if self.stop_event.is_set():
                        break
                    time.sleep(0.2)
                backoff_delay = min(max_backoff, backoff_delay * 2.0)
                continue

            # Successfully connected
            self.connected = True
            backoff_delay = 1.0
            logger.info("Camera '%s' connected successfully.", self.name)

            frame_idx = 0
            while not self.stop_event.is_set():
                read_ok, frame = source.read()
                if not read_ok or frame is None:
                    logger.warning("Stream drop detected on camera '%s'. Initiating auto-reconnect...", self.name)
                    self.connected = False
                    self.reconnect_count += 1
                    break

                now = time.time()
                self.last_frame_time = now
                frame_idx += 1

                # Calculate rolling FPS
                self._fps_timestamps.append(now)
                if len(self._fps_timestamps) > 1:
                    time_span = self._fps_timestamps[-1] - self._fps_timestamps[0]
                    if time_span > 0:
                        self.current_fps = (len(self._fps_timestamps) - 1) / time_span

                # Execute ByteTrack tracking inference with camera's isolated detector
                try:
                    detection_result: DetectionResult = self.detector.track_frame(
                        frame=frame,
                        resize_width=min(MAX_FRAME_DIMENSION, frame.shape[1]),
                    )
                except Exception as infer_err:
                    logger.error("Inference error on camera '%s': %s", self.name, infer_err)
                    continue

                # Update camera's isolated AccidentEventManager
                completed_incident = self.event_manager.update(
                    detection_result=detection_result,
                    frame=detection_result.annotated_frame,
                    timestamp=now,
                )

                # Update health incident properties
                if self.event_manager.is_incident_active:
                    self.active_incident_id = self.event_manager.current_incident_id
                    self.last_accident_class = self.event_manager.current_class_name
                    self.last_accident_confidence = self.event_manager.current_max_confidence
                elif completed_incident:
                    self.active_incident_id = None
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
                        completed_incident.vlm_severity = decision.severity
                        completed_incident.physics_passed = decision.physics_passed
                    except Exception as ver_err:
                        logger.warning("Verification error for camera incident %s: %s", completed_incident.incident_id, ver_err)
                        completed_incident.verification_status = "unverified"

                    self._persist_incident_to_db(completed_incident)
                    # Broadcast incident event
                    self._broadcast_message(
                        {
                            "type": "incident",
                            "camera_id": self.camera_id,
                            "camera_name": self.name,
                            "incident_id": completed_incident.incident_id,
                            "class_name": completed_incident.class_name,
                            "confidence": completed_incident.max_confidence,
                            "start_timestamp": completed_incident.start_timestamp,
                            "end_timestamp": completed_incident.end_timestamp,
                            "keyframe_url": f"/media/keyframes/{os.path.basename(completed_incident.image_path)}",
                            "verification_status": completed_incident.verification_status,
                        }
                    )

                # WebSocket frame rate limiter & transmission
                if (now - last_broadcast_time) >= MIN_BROADCAST_INTERVAL:
                    last_broadcast_time = now

                    # Downscale and compress to JPEG
                    disp_frame = detection_result.annotated_frame
                    h, w = disp_frame.shape[:2]
                    if max(h, w) > MAX_FRAME_DIMENSION:
                        scale = MAX_FRAME_DIMENSION / max(h, w)
                        disp_frame = cv2.resize(disp_frame, (int(w * scale), int(h * scale)))

                    encode_params = [cv2.IMWRITE_JPEG_QUALITY, FRAME_JPEG_QUALITY]
                    ret_enc, buffer = cv2.imencode(".jpg", disp_frame, encode_params)
                    if ret_enc:
                        b64_frame = base64.b64encode(buffer).decode("utf-8")
                        self._broadcast_message(
                            {
                                "type": "frame",
                                "camera_id": self.camera_id,
                                "camera_name": self.name,
                                "frame_number": frame_idx,
                                "timestamp": now,
                                "fps": round(self.current_fps, 1),
                                "incident_active": self.event_manager.is_incident_active,
                                "active_incident_id": self.event_manager.current_incident_id,
                                "frame": f"data:image/jpeg;base64,{b64_frame}",
                                "detections_count": len(detection_result.detections),
                            }
                        )

            # Clean release of source before potential reconnect
            source.release()

        self.connected = False


class CameraManager:
    """Central singleton managing all CCTV camera configurations and their dedicated workers."""

    def __init__(self) -> None:
        self.workers: Dict[str, CameraWorker] = {}
        self.lock = threading.Lock()
        self.loop: Optional[asyncio.AbstractEventLoop] = None

    def set_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        """Captures the active asyncio event loop for threadsafe WebSocket broadcasting."""
        self.loop = loop
        with self.lock:
            for worker in self.workers.values():
                worker.loop = loop

    def initialize_cameras(self) -> None:
        """Loads cameras from SQLite database and initializes active workers."""
        with Session(engine) as session:
            cameras = session.exec(select(Camera)).all()

            # Seed default demo cameras if database is empty
            if not cameras:
                logger.info("Seeding default demo CCTV cameras in database...")
                demo_sample_path = os.path.join(
                    os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))),
                    "acc_ex.mp4",
                )
                if not os.path.exists(demo_sample_path):
                    demo_sample_path = os.path.join(
                        os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))),
                        "ml",
                        "data",
                        "video_eval",
                        "test",
                        "test_holdout_acc_ex.mp4",
                    )

                seed_cameras = [
                    Camera(
                        id="cam-highway-north",
                        name="Highway 101 Northbound",
                        source_type="file",
                        source_url_encrypted=encrypt_credentials(demo_sample_path),
                        location_lat=37.7749,
                        location_lng=-122.4194,
                        enabled=True,
                        confidence_override=0.55,
                        created_at=time.time(),
                        updated_at=time.time(),
                    ),
                    Camera(
                        id="cam-junction-4",
                        name="Arterial Junction 4",
                        source_type="file",
                        source_url_encrypted=encrypt_credentials(demo_sample_path),
                        location_lat=37.7833,
                        location_lng=-122.4167,
                        enabled=True,
                        confidence_override=0.55,
                        created_at=time.time(),
                        updated_at=time.time(),
                    ),
                ]
                for c in seed_cameras:
                    session.add(c)
                session.commit()
                cameras = seed_cameras

            # Start workers for enabled cameras
            for cam in cameras:
                if cam.enabled:
                    self.start_worker_for_camera(cam)

    def start_worker_for_camera(self, camera: Camera) -> None:
        """Starts or restarts a dedicated worker for a given Camera DB record."""
        with self.lock:
            existing = self.workers.get(camera.id)
            if existing:
                existing.stop()

            decrypted_source = decrypt_credentials(camera.source_url_encrypted)
            worker = CameraWorker(
                camera_id=camera.id,
                name=camera.name,
                decrypted_source=decrypted_source,
                confidence_override=camera.confidence_override,
                loop=self.loop,
            )
            self.workers[camera.id] = worker
            worker.start()

    def stop_worker(self, camera_id: str) -> None:
        """Stops and cleans up worker for a camera."""
        with self.lock:
            worker = self.workers.pop(camera_id, None)
            if worker:
                worker.stop()

    def get_health(self, camera_id: str) -> CameraHealth:
        """Returns health telemetry for a camera, or default offline status if stopped."""
        with self.lock:
            worker = self.workers.get(camera_id)
            if worker:
                return worker.get_health()

        return CameraHealth(
            connected=False,
            fps=0.0,
            last_frame_age=999.0,
            dropped_frames=0,
            reconnect_count=0,
            is_incident_active=False,
        )

    def subscribe(self, camera_id: str, queue: asyncio.Queue) -> bool:
        """Subscribes an asyncio Queue to receive broadcast messages from a camera."""
        with self.lock:
            worker = self.workers.get(camera_id)
            if worker:
                worker.subscribe(queue)
                return True
            return False

    def unsubscribe(self, camera_id: str, queue: asyncio.Queue) -> None:
        """Unsubscribes a queue from a camera."""
        with self.lock:
            worker = self.workers.get(camera_id)
            if worker:
                worker.unsubscribe(queue)

    def shutdown(self) -> None:
        """Stops all active camera workers during application shutdown."""
        with self.lock:
            workers_list = list(self.workers.values())
            self.workers.clear()

        for w in workers_list:
            w.stop()
        logger.info("All camera workers shut down.")


# Global CameraManager singleton
camera_manager = CameraManager()
