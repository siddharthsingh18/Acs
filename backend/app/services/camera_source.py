"""
backend.app.services.camera_source
CameraSource abstraction and concrete implementations for:
1. FileCameraSource (local video file, uploaded file, with optional looping)
2. WebcamCameraSource (local USB/integrated camera index)
3. StreamCameraSource (RTSP, RTSPS, HTTP MJPEG, RTMP)

Includes secure credential encryption and masking utilities to ensure
RTSP/HTTP credentials are NEVER exposed in API responses or logs.
"""
from abc import ABC, abstractmethod
import base64
import hashlib
import logging
import os
import re
from typing import Optional, Tuple
from urllib.parse import urlparse, urlunparse

import cv2
import numpy as np

logger = logging.getLogger("acs.backend.camera_source")

# -----------------------------------------------------------------------------
# Security & Credential Encryption Utilities
# -----------------------------------------------------------------------------
_ENCRYPTION_KEY_ENV = "FERNET_KEY"
_DEFAULT_SALT = b"acs_camera_secret_salt_2026"


def _get_fernet_cipher():
    """Return the configured Fernet cipher; deterministic production fallbacks are forbidden."""
    from cryptography.fernet import Fernet

    env_key = os.getenv(_ENCRYPTION_KEY_ENV) or os.getenv("CAMERA_ENCRYPTION_KEY")
    if env_key:
        return Fernet(env_key.encode() if isinstance(env_key, str) else env_key)

    if os.getenv("ENVIRONMENT", "").lower() in {"production", "prod"}:
        raise RuntimeError("FERNET_KEY is required for camera credential encryption in production.")

    secret = os.getenv("SECRET_KEY", "acs-development-only-camera-key-not-for-production")
    logger.warning("Using a development-only camera encryption key; configure FERNET_KEY outside development.")
    derived = hashlib.pbkdf2_hmac("sha256", secret.encode(), _DEFAULT_SALT, 100_000)
    key_b64 = base64.urlsafe_b64encode(derived)
    return Fernet(key_b64)


def encrypt_credentials(raw_str: str) -> str:
    """Encrypts a plaintext connection string or URL for safe database persistence."""
    if not raw_str:
        return ""
    cipher = _get_fernet_cipher()
    return cipher.encrypt(raw_str.encode("utf-8")).decode("utf-8")


def decrypt_credentials(encrypted_str: str) -> str:
    """Decrypts a previously encrypted connection string or URL for runtime connection."""
    if not encrypted_str:
        return ""
    cipher = _get_fernet_cipher()
    return cipher.decrypt(encrypted_str.encode("utf-8")).decode("utf-8")


def mask_credentials(url_or_path: str) -> str:
    """Masks username and password in RTSP, HTTP, or custom connection URLs.

    Examples:
        rtsp://admin:pass123@192.168.1.100:554/live -> rtsp://***:***@192.168.1.100:554/live
        http://user:secret@camera.local/mjpeg -> http://***:***@camera.local/mjpeg
        0 -> 0 (webcam)
        /path/to/video.mp4 -> /path/to/video.mp4
    """
    if not url_or_path:
        return ""
    if not any(url_or_path.startswith(p) for p in ("rtsp://", "rtsps://", "http://", "https://", "rtmp://")):
        return url_or_path

    try:
        parsed = urlparse(url_or_path)
        if parsed.username or parsed.password:
            netloc = parsed.netloc
            # Replace user:pass@ with ***:***@
            userinfo, _, hostport = netloc.rpartition("@")
            masked_netloc = f"***:***@{hostport}" if userinfo else netloc
            return urlunparse(parsed._replace(netloc=masked_netloc))
        return url_or_path
    except Exception:
        # Regex fallback
        return re.sub(r"://([^:@]+):([^@]+)@", r"://***:***@", url_or_path)


# -----------------------------------------------------------------------------
# CameraSource Abstraction
# -----------------------------------------------------------------------------
class CameraSource(ABC):
    """Abstract Base Class for video and camera capture sources."""

    def __init__(self, raw_source: str, name: str = "camera"):
        self.raw_source = raw_source
        self.name = name
        self._is_opened = False

    @property
    @abstractmethod
    def source_type(self) -> str:
        """Type identifier: 'file', 'webcam', 'rtsp', or 'http'."""
        pass

    @property
    def masked_source(self) -> str:
        """Returns the connection string with all credentials safely masked."""
        return mask_credentials(self.raw_source)

    @abstractmethod
    def open(self) -> bool:
        """Opens the video source. Returns True if successfully opened."""
        pass

    @abstractmethod
    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        """Reads the next frame.

        Returns:
            Tuple[bool, Optional[np.ndarray]]: (success, bgr_frame)
        """
        pass

    @abstractmethod
    def release(self) -> None:
        """Releases the underlying capture resources."""
        pass

    def is_opened(self) -> bool:
        """Returns True if the source is currently opened and available for reading."""
        return self._is_opened


# -----------------------------------------------------------------------------
# Concrete Implementations
# -----------------------------------------------------------------------------
class FileCameraSource(CameraSource):
    """Reads from a local video file (or uploaded file) with optional auto-looping."""

    def __init__(self, file_path: str, loop: bool = True, name: str = "file_camera"):
        super().__init__(file_path, name=name)
        self.file_path = file_path
        self.loop = loop
        self.cap: Optional[cv2.VideoCapture] = None

    @property
    def source_type(self) -> str:
        return "file"

    def open(self) -> bool:
        if not os.path.exists(self.file_path):
            logger.error("FileCameraSource file not found: %s", self.file_path)
            self._is_opened = False
            return False

        if self.cap is not None:
            self.cap.release()

        self.cap = cv2.VideoCapture(self.file_path)
        self._is_opened = self.cap.isOpened()
        if not self._is_opened:
            logger.error("Failed to open video file: %s", self.file_path)
        return self._is_opened

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        if not self._is_opened or self.cap is None:
            return False, None

        ret, frame = self.cap.read()
        if not ret or frame is None:
            if self.loop:
                # Rewind to start of video for continuous stream emulation
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, frame = self.cap.read()
                if ret and frame is not None:
                    return True, frame
            return False, None

        return True, frame

    def release(self) -> None:
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        self._is_opened = False


class WebcamCameraSource(CameraSource):
    """Captures from a local webcam device index."""

    def __init__(self, device_index: int = 0, name: str = "webcam"):
        super().__init__(str(device_index), name=name)
        self.device_index = int(device_index)
        self.cap: Optional[cv2.VideoCapture] = None

    @property
    def source_type(self) -> str:
        return "webcam"

    def open(self) -> bool:
        if self.cap is not None:
            self.cap.release()

        # On macOS, cv2.CAP_AVFOUNDATION is preferred; fallback to default
        try:
            self.cap = cv2.VideoCapture(self.device_index, cv2.CAP_AVFOUNDATION)
            if not self.cap.isOpened():
                self.cap = cv2.VideoCapture(self.device_index)
        except Exception:
            self.cap = cv2.VideoCapture(self.device_index)

        self._is_opened = self.cap.isOpened()
        if self._is_opened:
            # Set single frame buffer size to avoid frame queuing delay
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        else:
            logger.warning("Could not open webcam device index %d", self.device_index)
        return self._is_opened

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        if not self._is_opened or self.cap is None:
            return False, None
        ret, frame = self.cap.read()
        return ret, frame

    def release(self) -> None:
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        self._is_opened = False


class StreamCameraSource(CameraSource):
    """Connects to an RTSP, RTSPS, RTMP, or HTTP MJPEG/HLS network video stream."""

    def __init__(self, stream_url: str, name: str = "network_camera"):
        super().__init__(stream_url, name=name)
        self.stream_url = stream_url
        self.cap: Optional[cv2.VideoCapture] = None

    @property
    def source_type(self) -> str:
        if self.stream_url.startswith(("rtsp://", "rtsps://")):
            return "rtsp"
        return "http"

    def open(self) -> bool:
        if self.cap is not None:
            self.cap.release()

        # Configure low-latency TCP transport for RTSP streams
        os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp|analyzeduration;1000000|probesize;1000000"

        logger.info("Opening network camera stream: %s", self.masked_source)
        try:
            self.cap = cv2.VideoCapture(self.stream_url)
        except Exception:
            self.cap = cv2.VideoCapture(self.stream_url, cv2.CAP_ANY)
        self._is_opened = self.cap.isOpened()
        if self._is_opened:
            # Drop OpenCV internal buffer to 1 frame to ensure real-time low-latency telemetry
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        else:
            logger.warning("Failed to open network camera stream: %s", self.masked_source)
        return self._is_opened

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        if not self._is_opened or self.cap is None:
            return False, None
        ret, frame = self.cap.read()
        return ret, frame

    def release(self) -> None:
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        self._is_opened = False


# -----------------------------------------------------------------------------
# Factory Function
# -----------------------------------------------------------------------------
def create_camera_source(source_url: str, name: str = "camera", loop: bool = True) -> CameraSource:
    """Factory creating the appropriate CameraSource implementation from a source string.

    Rules:
    - Pure integer or "webcam:N": WebcamCameraSource
    - Starts with "rtsp://", "rtsps://", "http://", "https://", "rtmp://": StreamCameraSource
    - Everything else: FileCameraSource
    """
    clean_source = source_url.strip()
    if clean_source.isdigit():
        return WebcamCameraSource(device_index=int(clean_source), name=name)
    if clean_source.lower().startswith("webcam:"):
        idx_str = clean_source.split(":", 1)[1]
        idx = int(idx_str) if idx_str.isdigit() else 0
        return WebcamCameraSource(device_index=idx, name=name)
    if any(clean_source.lower().startswith(p) for p in ("rtsp://", "rtsps://", "http://", "https://", "rtmp://")):
        return StreamCameraSource(stream_url=clean_source, name=name)
    return FileCameraSource(file_path=clean_source, loop=loop, name=name)
