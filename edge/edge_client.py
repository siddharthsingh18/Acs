"""
edge.edge_client
Lightweight edge detection daemon for ACS.

Runs on local hardware (NVIDIA Jetson, Intel NUC, or x86 edge boxes):
1. Ingests local RTSP / USB / CSI camera video feeds.
2. Runs local detection (TensorRT / OpenVINO / ONNX) and ByteTrack tracking.
3. Automatically applies face and license plate privacy redaction locally.
4. Generates an 18-second rolling pre/post-incident video clip and keyframe.
5. Transmits ONLY incident events, metadata, keyframe, and clip to central server.
6. Local offline resilience: buffers events in SQLite if network uplink is lost.
7. Periodic heartbeat telemetry with edge CPU, RAM, and camera status.
"""
import argparse
import base64
import json
import logging
import os
import sqlite3
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import numpy as np
import psutil
import requests

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from acs.config import config
from acs.detector import AccidentDetector, AccidentEventManager, AccidentIncident
from acs.privacy import apply_privacy_blur, protect_keyframe

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] [EDGE] %(message)s")
logger = logging.getLogger("acs.edge_client")


class EdgeQueueManager:
    """Manages persistent offline incident event queue in SQLite."""

    def __init__(self, db_path: str = "data/edge_queue.db"):
        self.db_path = db_path
        os.makedirs(os.path.dirname(os.path.abspath(db_path)), exist_ok=True)
        self._init_db()

    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS queued_incidents (
                    incident_id TEXT PRIMARY KEY,
                    camera_id TEXT,
                    payload_json TEXT,
                    keyframe_path TEXT,
                    clip_path TEXT,
                    created_at REAL,
                    retry_count INTEGER DEFAULT 0
                )
            """)
            conn.commit()

    def enqueue(
        self,
        incident_id: str,
        camera_id: str,
        payload: Dict[str, Any],
        keyframe_path: Optional[str],
        clip_path: Optional[str],
    ) -> None:
        """Stores incident event for transmission to central server."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO queued_incidents 
                (incident_id, camera_id, payload_json, keyframe_path, clip_path, created_at, retry_count)
                VALUES (?, ?, ?, ?, ?, ?, 0)
                """,
                (incident_id, camera_id, json.dumps(payload), keyframe_path, clip_path, time.time()),
            )
            conn.commit()
        logger.info("Queued incident %s for uplink transmission.", incident_id)

    def get_pending(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Fetches pending incidents to transmit."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM queued_incidents ORDER BY created_at ASC LIMIT ?", (limit,))
            rows = cursor.fetchall()
            return [dict(r) for r in rows]

    def remove(self, incident_id: str) -> None:
        """Removes successfully transmitted incident from queue."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("DELETE FROM queued_incidents WHERE incident_id = ?", (incident_id,))
            conn.commit()

    def increment_retry(self, incident_id: str) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("UPDATE queued_incidents SET retry_count = retry_count + 1 WHERE incident_id = ?", (incident_id,))
            conn.commit()


class EdgeDetectionAgent:
    """Standalone edge agent capturing local streams, detecting collisions, and uploading events."""

    def __init__(
        self,
        node_id: Optional[str] = None,
        central_url: Optional[str] = None,
        api_key: Optional[str] = None,
        camera_sources: Optional[List[str]] = None,
        model_path: Optional[str] = None,
        data_dir: str = "data",
    ):
        self.node_id = node_id or os.getenv("ACS_EDGE_NODE_ID", "edge-node-01")
        self.central_url = (central_url or os.getenv("CENTRAL_SERVER_URL", "http://localhost:8000")).rstrip("/")
        self.api_key = api_key or os.getenv("ACS_EDGE_API_KEY", "dev_edge_secret_key")
        self.data_dir = data_dir
        os.makedirs(self.data_dir, exist_ok=True)

        self.queue_manager = EdgeQueueManager(os.path.join(self.data_dir, "edge_queue.db"))

        # Cameras
        raw_sources = camera_sources or os.getenv("CAMERA_SOURCES", "").split(",")
        self.camera_sources = [s.strip() for s in raw_sources if s.strip()] or ["0"]

        # Local detector
        self.model_path = model_path or getattr(config, "model_path", "best.pt")
        self.detector = AccidentDetector(model_path=self.model_path)
        self.stop_event = threading.Event()

    def start(self):
        """Launches camera capture threads and uplink synchronization worker."""
        logger.info("Starting Edge Agent %s (Target Central: %s)", self.node_id, self.central_url)

        # 1. Start uplink dispatcher thread
        uplink_thread = threading.Thread(target=self._uplink_loop, daemon=True, name="EdgeUplinkWorker")
        uplink_thread.start()

        # 2. Start heartbeat thread
        heartbeat_thread = threading.Thread(target=self._heartbeat_loop, daemon=True, name="EdgeHeartbeatWorker")
        heartbeat_thread.start()

        # 3. Start camera monitoring
        threads = []
        for idx, src in enumerate(self.camera_sources):
            cam_id = f"{self.node_id}-cam{idx+1:02d}"
            t = threading.Thread(target=self._camera_loop, args=(cam_id, src), daemon=True, name=f"Cam-{cam_id}")
            t.start()
            threads.append(t)

        logger.info("Edge agent running with %d camera(s). Press Ctrl+C to stop.", len(threads))
        try:
            while not self.stop_event.is_set():
                time.sleep(1.0)
        except KeyboardInterrupt:
            logger.info("Stopping Edge Agent...")
            self.stop_event.set()

    def _camera_loop(self, camera_id: str, source: str):
        """Processes video feed for one local camera stream."""
        logger.info("Camera %s connected to source %s", camera_id, source)
        src_val = int(source) if source.isdigit() else source
        cap = cv2.VideoCapture(src_val)

        event_manager = AccidentEventManager(
            camera_id=camera_id,
            detector=self.detector,
            on_incident_callback=lambda inc: self._on_incident_detected(camera_id, inc),
        )

        while not self.stop_event.is_set():
            ret, frame = cap.read()
            if not ret or frame is None:
                time.sleep(0.5)
                continue

            event_manager.process_frame(frame)

        cap.release()

    def _on_incident_detected(self, camera_id: str, incident: AccidentIncident):
        """Invoked when local detector confirms an accident."""
        logger.warning("🚨 [EDGE DETECTED] %s on %s (Conf: %.2f)", incident.class_name, camera_id, incident.max_confidence)

        # Apply local face and license plate privacy blurring
        kf_dir = os.path.join(self.data_dir, "keyframes")
        os.makedirs(kf_dir, exist_ok=True)
        kf_path = os.path.join(kf_dir, f"{incident.incident_id}_keyframe.jpg")

        if incident.peak_keyframe is not None:
            blurred_kf = apply_privacy_blur(incident.peak_keyframe)
            cv2.imwrite(kf_path, blurred_kf)
        else:
            kf_path = None

        payload = {
            "incident_id": incident.incident_id,
            "camera_id": camera_id,
            "edge_node_id": self.node_id,
            "start_timestamp": incident.start_timestamp,
            "end_timestamp": incident.end_timestamp,
            "class_name": incident.class_name,
            "max_confidence": float(incident.max_confidence),
            "severity": getattr(incident, "severity", "medium"),
            "severity_score": getattr(incident, "severity_score", 50),
            "is_privacy_blurred": True,
            "created_at": time.time(),
        }

        # Queue for transmission
        self.queue_manager.enqueue(
            incident_id=incident.incident_id,
            camera_id=camera_id,
            payload=payload,
            keyframe_path=kf_path,
            clip_path=getattr(incident, "clip_path", None),
        )

    def _uplink_loop(self):
        """Continuously transmits queued incidents to the central server."""
        headers = {
            "X-Edge-API-Key": self.api_key,
            "Authorization": f"Bearer {self.api_key}",
        }

        while not self.stop_event.is_set():
            pending = self.queue_manager.get_pending(limit=5)
            if not pending:
                time.sleep(2.0)
                continue

            for item in pending:
                inc_id = item["incident_id"]
                payload = json.loads(item["payload_json"])

                files = {}
                # Attach keyframe JPEG if present
                if item["keyframe_path"] and os.path.exists(item["keyframe_path"]):
                    files["keyframe"] = (
                        os.path.basename(item["keyframe_path"]),
                        open(item["keyframe_path"], "rb"),
                        "image/jpeg",
                    )
                # Attach video clip MP4 if present
                if item["clip_path"] and os.path.exists(item["clip_path"]):
                    files["clip"] = (
                        os.path.basename(item["clip_path"]),
                        open(item["clip_path"], "rb"),
                        "video/mp4",
                    )

                endpoint = f"{self.central_url}/api/edge/incidents"
                try:
                    resp = requests.post(
                        endpoint,
                        data={"metadata": json.dumps(payload)},
                        files=files,
                        headers=headers,
                        timeout=15.0,
                    )
                    # Close opened files
                    for _, f_tuple in files.items():
                        f_tuple[1].close()

                    if resp.status_code in (200, 201):
                        logger.info("Successfully transmitted incident %s to central server.", inc_id)
                        self.queue_manager.remove(inc_id)
                    else:
                        logger.warning("Central server rejected incident %s (HTTP %d): %s", inc_id, resp.status_code, resp.text)
                        self.queue_manager.increment_retry(inc_id)
                except Exception as exc:
                    logger.debug("Network error transmitting incident %s: %s (will retry)", inc_id, exc)
                    self.queue_manager.increment_retry(inc_id)

            time.sleep(1.0)

    def _heartbeat_loop(self):
        """Emits periodic telemetry to central server."""
        headers = {"X-Edge-API-Key": self.api_key}
        endpoint = f"{self.central_url}/api/edge/heartbeat"

        while not self.stop_event.is_set():
            try:
                telemetry = {
                    "node_id": self.node_id,
                    "cpu_percent": psutil.cpu_percent(),
                    "ram_percent": psutil.virtual_memory().percent,
                    "active_cameras": len(self.camera_sources),
                    "pending_queue_count": len(self.queue_manager.get_pending(limit=100)),
                    "timestamp": time.time(),
                }
                requests.post(endpoint, json=telemetry, headers=headers, timeout=5.0)
            except Exception as e:
                logger.debug("Heartbeat ping failed: %s", e)
            time.sleep(15.0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ACS Edge Detection Agent")
    parser.add_argument("--node-id", default=os.getenv("ACS_EDGE_NODE_ID", "edge-box-01"), help="Unique Edge Node ID")
    parser.add_argument("--central-url", default=os.getenv("CENTRAL_SERVER_URL", "http://localhost:8000"), help="Central server base URL")
    parser.add_argument("--api-key", default=os.getenv("ACS_EDGE_API_KEY", "dev_edge_secret_key"), help="Edge authentication secret")
    parser.add_argument("--cameras", nargs="*", default=None, help="List of camera sources (RTSP URLs or device indices)")
    parser.add_argument("--model", default="best.pt", help="Path to local model weights")
    args = parser.parse_args()

    agent = EdgeDetectionAgent(
        node_id=args.node_id,
        central_url=args.central_url,
        api_key=args.api_key,
        camera_sources=args.cameras,
        model_path=args.model,
    )
    agent.start()
