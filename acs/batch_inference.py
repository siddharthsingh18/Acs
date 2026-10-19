"""
acs.batch_inference
Multi-camera batched inference pipeline for GPU and hardware accelerators.

Batches incoming video frames across independent camera streams into a single
model forward pass (batch size B in [1..32]), amortizing GPU kernel launch overhead,
maximizing Tensor Core utilization, and sustaining multiple 720p 15 FPS camera streams.
"""
import logging
import queue
import threading
import time
from dataclasses import dataclass
from typing import Any, Callable, List, Optional

import numpy as np
import torch

from acs.config import config

logger = logging.getLogger("acs.batch_inference")


@dataclass
class FramePacket:
    """Represents a frame ingested from an individual camera stream."""
    camera_id: str
    frame: np.ndarray
    timestamp: float
    callback: Optional[Callable[[Any], None]] = None


class BatchedInferenceEngine:
    """Manages multi-stream frame collation and batched forward passes on hardware accelerators."""

    def __init__(
        self,
        model_path: Optional[str] = None,
        max_batch_size: int = 8,
        batch_timeout_ms: float = 12.0,
        device: Optional[str] = None,
        imgsz: int = 640,
    ):
        self.model_path = model_path or getattr(config, "model_path", "best.pt")
        self.max_batch_size = max_batch_size
        self.batch_timeout_ms = batch_timeout_ms
        self.imgsz = imgsz

        # Hardware accelerator selection
        if device:
            self.device = device
        elif torch.cuda.is_available():
            self.device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            self.device = "mps"
        else:
            self.device = "cpu"

        self._queue: queue.Queue = queue.Queue(maxsize=128)
        self._stop_event = threading.Event()
        self._worker_thread: Optional[threading.Thread] = None

        # Lazy model initialization
        self._model = None
        self._lock = threading.Lock()

        # Telemetry stats
        self.total_batches_processed = 0
        self.total_frames_processed = 0
        self.total_inference_time_ms = 0.0

    def _load_model(self):
        if self._model is None:
            from ultralytics import YOLO
            logger.info("Initializing batched inference engine on %s with model %s", self.device, self.model_path)
            self._model = YOLO(self.model_path)
            # Warm up on device
            dummy = np.zeros((self.imgsz, self.imgsz, 3), dtype=np.uint8)
            _ = self._model(dummy, verbose=False, device=self.device)

    def start(self):
        """Starts the background batching worker thread."""
        if self._worker_thread and self._worker_thread.is_alive():
            return
        self._stop_event.clear()
        self._worker_thread = threading.Thread(target=self._batch_loop, daemon=True, name="BatchInferenceWorker")
        self._worker_thread.start()
        logger.info("Batched inference worker started (max_batch_size=%d, timeout=%.1fms)", self.max_batch_size, self.batch_timeout_ms)

    def stop(self):
        """Signals and stops the background batching thread."""
        self._stop_event.set()
        if self._worker_thread:
            self._worker_thread.join(timeout=2.0)
            self._worker_thread = None

    def submit_frame(
        self,
        camera_id: str,
        frame: np.ndarray,
        callback: Optional[Callable[[Any], None]] = None,
    ) -> bool:
        """Enqueues a frame for batched GPU forward pass.

        Returns False if the ingestion queue is saturated (frame drop).
        """
        packet = FramePacket(
            camera_id=camera_id,
            frame=frame,
            timestamp=time.time(),
            callback=callback,
        )
        try:
            self._queue.put_nowait(packet)
            return True
        except queue.Full:
            return False

    def predict_batch_sync(self, frames: List[np.ndarray]) -> List[Any]:
        """Direct synchronous batch inference forward pass on list of BGR frames."""
        self._load_model()
        if not frames:
            return []
        # Ultralytics natively handles list of images as a batch tensor
        results = self._model(frames, imgsz=self.imgsz, device=self.device, verbose=False)
        return results

    def _batch_loop(self):
        """Continuous background thread gathering frames into dynamic batches."""
        self._load_model()
        timeout_sec = self.batch_timeout_ms / 1000.0

        while not self._stop_event.is_set():
            batch_packets: List[FramePacket] = []
            deadline = time.time() + timeout_sec

            # Gather up to max_batch_size packets before deadline
            while len(batch_packets) < self.max_batch_size and time.time() < deadline:
                remaining = max(0.001, deadline - time.time())
                try:
                    packet = self._queue.get(timeout=remaining)
                    batch_packets.append(packet)
                except queue.Empty:
                    break

            if not batch_packets:
                continue

            # Execute forward pass
            t0 = time.perf_counter()
            frames_to_infer = [p.frame for p in batch_packets]
            try:
                results = self._model(frames_to_infer, imgsz=self.imgsz, device=self.device, verbose=False)
                t1 = time.perf_counter()
                elapsed_ms = (t1 - t0) * 1000.0

                self.total_batches_processed += 1
                self.total_frames_processed += len(batch_packets)
                self.total_inference_time_ms += elapsed_ms

                # Dispatch results to respective callbacks
                for packet, res in zip(batch_packets, results):
                    if packet.callback:
                        try:
                            packet.callback(res)
                        except Exception as cb_err:
                            logger.debug("Error in batch result callback for %s: %s", packet.camera_id, cb_err)
            except Exception as inf_err:
                logger.error("Batched forward pass failed: %s", inf_err)
