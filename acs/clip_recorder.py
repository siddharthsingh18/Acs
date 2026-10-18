"""
acs.clip_recorder
Rolling frame buffer, incident clip extraction, H.264 MP4 encoding (ffmpeg),
and disk retention management.

Features:
- Bounded in-memory rolling ring buffer per camera/stream (default 10 seconds).
- Pre-incident (8s) and post-incident (10s) clip collection.
- H.264 MP4 export using ffmpeg libx264 with faststart for streaming & HTTP range requests.
- Dual version rendering: raw footage and annotated footage with bounding boxes burned in.
- Clip retention policy enforcement: max age in days and max disk usage in MB.
"""
from collections import deque
import dataclasses
import glob
import logging
import os
import shutil
import subprocess
import threading
import time
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from acs.config import config

logger = logging.getLogger("acs.clip_recorder")


@dataclasses.dataclass
class BufferedFrame:
    """A timestamped video frame held in the ring buffer."""

    timestamp: float
    raw_frame: np.ndarray
    annotated_frame: np.ndarray


class CameraRollingBuffer:
    """Thread-safe, strictly bounded in-memory ring buffer holding the last N seconds of frames.

    Memory stays completely flat over indefinite continuous runs because old frames
    are automatically evicted by deque(maxlen=...) and garbage collected.
    """

    def __init__(
        self,
        max_seconds: float = 10.0,
        target_fps: float = 30.0,
        max_frames: Optional[int] = None,
    ):
        """Initializes the rolling ring buffer.

        Args:
            max_seconds: Rolling window duration in seconds (default: 10.0s).
            target_fps: Estimated stream frames per second (default: 30.0).
            max_frames: Explicit maxlen override. If None, computed as int(max_seconds * target_fps * 1.2).
        """
        self.max_seconds = max_seconds
        self.target_fps = target_fps
        computed_len = int(max_seconds * target_fps * 1.2) + 15
        self.max_frames = max_frames if max_frames is not None else max(60, computed_len)

        self._buffer: deque = deque(maxlen=self.max_frames)
        self._lock = threading.Lock()

    def append(
        self,
        timestamp: float,
        raw_frame: np.ndarray,
        annotated_frame: Optional[np.ndarray] = None,
    ) -> None:
        """Appends a timestamped frame to the rolling ring buffer.

        Args:
            timestamp: Frame epoch time or video timestamp in seconds.
            raw_frame: Raw BGR uint8 frame.
            annotated_frame: Optional frame with bounding boxes burned in.
        """
        ann = annotated_frame if annotated_frame is not None else raw_frame
        # Ensure numpy array copy/contiguous
        raw_copy = raw_frame if raw_frame is not None else np.zeros((240, 320, 3), dtype=np.uint8)
        ann_copy = ann if ann is not None else raw_copy

        with self._lock:
            self._buffer.append(
                BufferedFrame(
                    timestamp=float(timestamp),
                    raw_frame=raw_copy,
                    annotated_frame=ann_copy,
                )
            )

    def get_window(
        self,
        start_time: float,
        end_time: float,
    ) -> Tuple[List[float], List[np.ndarray], List[np.ndarray]]:
        """Extracts all frames between start_time and end_time.

        Args:
            start_time: Start timestamp (inclusive).
            end_time: End timestamp (inclusive).

        Returns:
            Tuple of (timestamps, raw_frames, annotated_frames).
        """
        with self._lock:
            ts_list: List[float] = []
            raw_list: List[np.ndarray] = []
            ann_list: List[np.ndarray] = []
            for item in self._buffer:
                if start_time <= item.timestamp <= end_time:
                    ts_list.append(item.timestamp)
                    raw_list.append(item.raw_frame)
                    ann_list.append(item.annotated_frame)
            return ts_list, raw_list, ann_list

    def get_recent(
        self,
        seconds: float = 8.0,
        before_time: Optional[float] = None,
    ) -> Tuple[List[float], List[np.ndarray], List[np.ndarray]]:
        """Extracts frames from the preceding N seconds before a given timestamp."""
        with self._lock:
            if not self._buffer:
                return [], [], []
            ref_time = before_time if before_time is not None else self._buffer[-1].timestamp
            cutoff = ref_time - seconds
            ts_list: List[float] = []
            raw_list: List[np.ndarray] = []
            ann_list: List[np.ndarray] = []
            for item in self._buffer:
                if cutoff <= item.timestamp <= ref_time:
                    ts_list.append(item.timestamp)
                    raw_list.append(item.raw_frame)
                    ann_list.append(item.annotated_frame)
            return ts_list, raw_list, ann_list

    def clear(self) -> None:
        """Empties the rolling buffer."""
        with self._lock:
            self._buffer.clear()

    def __len__(self) -> int:
        with self._lock:
            return len(self._buffer)


class ActiveClipSession:
    """State tracking for an in-progress incident clip recording session."""

    def __init__(
        self,
        incident_id: str,
        impact_time: float,
        pre_timestamps: List[float],
        pre_raw_frames: List[np.ndarray],
        pre_annotated_frames: List[np.ndarray],
        pre_seconds: float = 8.0,
        post_seconds: float = 10.0,
        fps: float = 30.0,
    ):
        self.incident_id = incident_id
        self.impact_time = impact_time
        self.pre_seconds = pre_seconds
        self.post_seconds = post_seconds
        self.fps = fps
        self.target_end_time = impact_time + post_seconds

        # Pre-frames collected from rolling buffer
        self.timestamps: List[float] = list(pre_timestamps)
        self.raw_frames: List[np.ndarray] = list(pre_raw_frames)
        self.annotated_frames: List[np.ndarray] = list(pre_annotated_frames)

        self.clip_start_time = pre_timestamps[0] if pre_timestamps else impact_time
        self.is_finalized = False
        self.clip_raw_path: Optional[str] = None
        self.clip_annotated_path: Optional[str] = None

    def add_frame(
        self,
        timestamp: float,
        raw_frame: np.ndarray,
        annotated_frame: np.ndarray,
    ) -> None:
        """Appends a new frame during active incident recording."""
        if self.is_finalized:
            return
        self.timestamps.append(timestamp)
        self.raw_frames.append(raw_frame)
        self.annotated_frames.append(annotated_frame)

    @property
    def impact_offset_seconds(self) -> float:
        """Relative time (in seconds) from clip beginning to impact onset."""
        if not self.timestamps:
            return self.pre_seconds
        return max(0.0, self.impact_time - self.timestamps[0])


def save_h264_clip(
    frames: List[np.ndarray],
    output_path: str,
    fps: float = 30.0,
) -> bool:
    """Encodes a sequence of BGR frames as an H.264 MP4 using ffmpeg with faststart.

    Falls back to cv2.VideoWriter if ffmpeg is unavailable.

    Args:
        frames: Sequence of BGR uint8 numpy arrays.
        output_path: Target .mp4 file path.
        fps: Playback framerate.

    Returns:
        bool: True if clip was created and has non-zero size, False otherwise.
    """
    if not frames:
        logger.warning("save_h264_clip called with empty frame list.")
        return False

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    height, width = frames[0].shape[:2]
    fps_val = max(1.0, float(fps))

    # 1. Primary Encoder: ffmpeg with libx264 and +faststart
    try:
        cmd = [
            "ffmpeg",
            "-y",
            "-f", "rawvideo",
            "-vcodec", "rawvideo",
            "-s", f"{width}x{height}",
            "-pix_fmt", "bgr24",
            "-r", f"{fps_val:.2f}",
            "-i", "-",
            "-c:v", "libx264",
            "-preset", "fast",
            "-crf", "23",
            "-pix_fmt", "yuv420p",
            "-movflags", "+faststart",
            output_path,
        ]
        proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        for f in frames:
            if f.shape[:2] != (height, width):
                f = cv2.resize(f, (width, height))
            proc.stdin.write(f.astype(np.uint8).tobytes())
        proc.stdin.close()
        proc.wait(timeout=30)
        if proc.returncode == 0 and os.path.exists(output_path) and os.path.getsize(output_path) > 0:
            return True
        else:
            stderr_out = proc.stderr.read().decode("utf-8", errors="ignore")
            logger.warning("ffmpeg returned non-zero code %d: %s", proc.returncode, stderr_out[:200])
    except Exception as exc:
        logger.warning("ffmpeg encoding failed for '%s': %s. Falling back to OpenCV.", output_path, exc)

    # 2. Secondary Fallback: cv2.VideoWriter
    try:
        for fourcc_code in ["avc1", "mp4v"]:
            fourcc = cv2.VideoWriter_fourcc(*fourcc_code)
            writer = cv2.VideoWriter(output_path, fourcc, fps_val, (width, height))
            if writer.isOpened():
                for f in frames:
                    if f.shape[:2] != (height, width):
                        f = cv2.resize(f, (width, height))
                    writer.write(f.astype(np.uint8))
                writer.release()
                if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
                    return True
    except Exception as cv_exc:
        logger.error("OpenCV video writing failed for '%s': %s", output_path, cv_exc)

    return False


def cleanup_expired_clips(
    clips_dir: Optional[str] = None,
    retention_days: Optional[int] = None,
    max_disk_usage_mb: Optional[int] = None,
) -> dict:
    """Enforces retention policy: deletes clips older than N days and caps total disk usage.

    Args:
        clips_dir: Directory containing .mp4 files (defaults to config.clips_dir).
        retention_days: Maximum age in days before clip is purged (defaults to config.clip_retention_days).
        max_disk_usage_mb: Maximum total size in MB for all clips in directory (defaults to config.clip_max_disk_usage_mb).

    Returns:
        dict: Summary of purged files, freed bytes, and remaining storage stats.
    """
    target_dir = clips_dir or getattr(config, "clips_dir", "report_clips")
    max_age_days = retention_days if retention_days is not None else getattr(config, "clip_retention_days", 7)
    max_mb = max_disk_usage_mb if max_disk_usage_mb is not None else getattr(config, "clip_max_disk_usage_mb", 2048)
    max_bytes = max_mb * 1024 * 1024

    if not os.path.exists(target_dir):
        return {
            "deleted_count": 0,
            "freed_bytes": 0,
            "total_files": 0,
            "total_bytes": 0,
            "total_mb": 0.0,
        }

    now = time.time()
    cutoff_time = now - (max_age_days * 86400)
    deleted_count = 0
    freed_bytes = 0

    all_files = glob.glob(os.path.join(target_dir, "*.mp4"))

    # Phase 1: Delete files older than retention_days
    remaining_files = []
    for f in all_files:
        try:
            mtime = os.path.getmtime(f)
            size = os.path.getsize(f)
            if mtime < cutoff_time:
                os.remove(f)
                deleted_count += 1
                freed_bytes += size
            else:
                remaining_files.append((f, mtime, size))
        except OSError:
            pass

    # Phase 2: Check total size against max_disk_usage_mb
    total_size = sum(item[2] for item in remaining_files)
    if total_size > max_bytes:
        # Sort oldest first
        remaining_files.sort(key=lambda item: item[1])
        surplus = total_size - max_bytes
        for f, mtime, size in remaining_files:
            if surplus <= 0:
                break
            try:
                os.remove(f)
                deleted_count += 1
                freed_bytes += size
                surplus -= size
                total_size -= size
            except OSError:
                pass

    total_mb = round(total_size / (1024 * 1024), 2)
    current_files = glob.glob(os.path.join(target_dir, "*.mp4"))

    return {
        "deleted_count": deleted_count,
        "freed_bytes": freed_bytes,
        "total_files": len(current_files),
        "total_bytes": total_size,
        "total_mb": total_mb,
    }


def get_clip_storage_stats(clips_dir: Optional[str] = None) -> dict:
    """Returns storage utilization metrics for the clips directory."""
    target_dir = clips_dir or getattr(config, "clips_dir", "report_clips")
    if not os.path.exists(target_dir):
        return {"total_files": 0, "total_bytes": 0, "total_mb": 0.0, "clips_dir": target_dir}

    all_files = glob.glob(os.path.join(target_dir, "*.mp4"))
    total_bytes = sum(os.path.getsize(f) for f in all_files if os.path.isfile(f))
    return {
        "total_files": len(all_files),
        "total_bytes": total_bytes,
        "total_mb": round(total_bytes / (1024 * 1024), 2),
        "clips_dir": target_dir,
    }


class IncidentClipRecorder:
    """Coordinates rolling buffer ingestion, active clip window recording, and H.264 export."""

    def __init__(
        self,
        rolling_buffer_seconds: float = 10.0,
        pre_incident_seconds: float = 8.0,
        post_incident_seconds: float = 10.0,
        output_dir: Optional[str] = None,
        target_fps: float = 30.0,
    ):
        self.rolling_buffer = CameraRollingBuffer(
            max_seconds=rolling_buffer_seconds,
            target_fps=target_fps,
        )
        self.pre_incident_seconds = pre_incident_seconds
        self.post_incident_seconds = post_incident_seconds
        self.output_dir = output_dir or getattr(config, "clips_dir", "report_clips")
        self.target_fps = target_fps
        self.active_sessions: Dict[str, ActiveClipSession] = {}
        self._lock = threading.Lock()

    def feed_frame(
        self,
        timestamp: float,
        raw_frame: np.ndarray,
        annotated_frame: Optional[np.ndarray] = None,
    ) -> None:
        """Pushes a new frame to the rolling ring buffer and any active incident sessions."""
        ann = annotated_frame if annotated_frame is not None else raw_frame
        self.rolling_buffer.append(timestamp, raw_frame, ann)

        with self._lock:
            # Feed active recording sessions
            for session in list(self.active_sessions.values()):
                if not session.is_finalized:
                    session.add_frame(timestamp, raw_frame, ann)

    def start_incident(
        self,
        incident_id: str,
        impact_time: float,
    ) -> ActiveClipSession:
        """Starts an active clip session for an incident, snapshotting 8s pre-impact frames."""
        with self._lock:
            if incident_id in self.active_sessions:
                return self.active_sessions[incident_id]

            # Collect 8 seconds prior to impact
            pre_ts, pre_raw, pre_ann = self.rolling_buffer.get_recent(
                seconds=self.pre_incident_seconds,
                before_time=impact_time,
            )

            session = ActiveClipSession(
                incident_id=incident_id,
                impact_time=impact_time,
                pre_timestamps=pre_ts,
                pre_raw_frames=pre_raw,
                pre_annotated_frames=pre_ann,
                pre_seconds=self.pre_incident_seconds,
                post_seconds=self.post_incident_seconds,
                fps=self.target_fps,
            )
            self.active_sessions[incident_id] = session
            return session

    def finalize_incident(
        self,
        incident_id: str,
        output_dir: Optional[str] = None,
    ) -> Tuple[Optional[str], Optional[str], float]:
        """Finalizes and encodes both raw and annotated H.264 MP4 clips for the incident.

        Returns:
            Tuple of (raw_clip_path, annotated_clip_path, impact_offset_seconds).
        """
        session: Optional[ActiveClipSession] = None
        with self._lock:
            session = self.active_sessions.get(incident_id)

        if not session:
            logger.warning("No clip session found for incident %s", incident_id)
            return None, None, 0.0

        if session.is_finalized and session.clip_raw_path and session.clip_annotated_path:
            return session.clip_raw_path, session.clip_annotated_path, session.impact_offset_seconds

        target_dir = output_dir or self.output_dir
        os.makedirs(target_dir, exist_ok=True)

        raw_path = os.path.join(target_dir, f"{incident_id}_raw.mp4")
        ann_path = os.path.join(target_dir, f"{incident_id}_annotated.mp4")

        # Encode raw clip
        raw_ok = save_h264_clip(session.raw_frames, raw_path, fps=session.fps)
        # Encode annotated clip with bounding boxes burned in
        ann_ok = save_h264_clip(session.annotated_frames, ann_path, fps=session.fps)

        session.is_finalized = True
        session.clip_raw_path = raw_path if raw_ok else None
        session.clip_annotated_path = ann_path if ann_ok else None

        # Run retention check after saving new clips
        try:
            cleanup_expired_clips(clips_dir=target_dir)
        except Exception as cl_err:
            logger.warning("Error running retention cleanup: %s", cl_err)

        return session.clip_raw_path, session.clip_annotated_path, session.impact_offset_seconds
