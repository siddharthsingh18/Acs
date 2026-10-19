"""
backend.app.core.metrics
Central Prometheus metrics registry and instrumentations for ACS.

Tracks:
1. fps per camera (Gauge: acs_camera_fps)
2. inference latency (Histogram: acs_inference_duration_seconds)
3. queue depth (Gauge: acs_queue_depth for review_queue, jobs, subscribers)
4. incidents (Counter: acs_incidents_total)
5. VLM calls and failures (Counter: acs_vlm_calls_total, Histogram: acs_vlm_duration_seconds)
6. alert delivery (Counter: acs_alert_delivery_total, Histogram: acs_alert_duration_seconds)
7. camera connectivity and stream drops (Gauge: acs_camera_connected, Counter: acs_camera_reconnects_total)
8. model drift status (Gauge: acs_model_drift_detected)
"""
import logging
from typing import Optional

try:
    from prometheus_client import (
        CONTENT_TYPE_LATEST,
        Counter,
        Gauge,
        Histogram,
        generate_latest,
    )
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False
    CONTENT_TYPE_LATEST = "text/plain; version=0.0.4; charset=utf-8"

logger = logging.getLogger("acs.metrics")

if PROMETHEUS_AVAILABLE:
    # 1. Camera Performance & FPS
    CAMERA_FPS = Gauge(
        "acs_camera_fps",
        "Current frame processing and inference frames per second per camera",
        ["camera_id", "camera_name"],
    )

    CAMERA_CONNECTED = Gauge(
        "acs_camera_connected",
        "Camera connection status (1 = connected, 0 = offline)",
        ["camera_id", "camera_name"],
    )

    CAMERA_DROPPED_FRAMES = Counter(
        "acs_camera_dropped_frames_total",
        "Total dropped frames per camera due to slow subscribers or frame-skip policy",
        ["camera_id"],
    )

    CAMERA_RECONNECTS = Counter(
        "acs_camera_reconnects_total",
        "Total reconnection attempts per camera due to stream interruption",
        ["camera_id"],
    )

    # 2. YOLO Inference Latency
    INFERENCE_LATENCY = Histogram(
        "acs_inference_duration_seconds",
        "YOLO inference latency in seconds per frame",
        ["camera_id", "model"],
        buckets=(0.005, 0.01, 0.02, 0.03, 0.05, 0.075, 0.1, 0.15, 0.25, 0.5, 1.0),
    )

    # 3. Queue Depth
    QUEUE_DEPTH = Gauge(
        "acs_queue_depth",
        "Current backlog depth of queues (review_queue, active_jobs, subscribers)",
        ["queue_name"],
    )

    # 4. Incident Detections
    INCIDENTS_TOTAL = Counter(
        "acs_incidents_total",
        "Cumulative count of accident incidents detected by event manager",
        ["camera_id", "severity", "verification_status"],
    )

    # 5. VLM Verification Calls & Failures
    VLM_CALLS_TOTAL = Counter(
        "acs_vlm_calls_total",
        "Total Google Gemini VLM multimodal verification calls and outcomes",
        ["status", "model"],  # status: success, failure, timeout, cached, budget_exhausted
    )

    VLM_DURATION_SECONDS = Histogram(
        "acs_vlm_duration_seconds",
        "Latency of Google Gemini VLM verification API calls in seconds",
        ["model"],
        buckets=(0.25, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0, 8.0, 12.0, 20.0),
    )

    # 6. Emergency Alert Delivery
    ALERT_DELIVERY_TOTAL = Counter(
        "acs_alert_delivery_total",
        "Cumulative emergency alert dispatches broken down by channel and delivery status",
        ["channel", "status"],  # channel: telegram, smtp, webpush, console; status: success, failure, rate_limited, suppressed_demo
    )

    ALERT_DURATION_SECONDS = Histogram(
        "acs_alert_duration_seconds",
        "Latency in seconds for external notification delivery dispatch",
        ["channel"],
        buckets=(0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0),
    )

    # 7. Model Drift Signals
    MODEL_DRIFT_DETECTED = Gauge(
        "acs_model_drift_detected",
        "Model drift indicator per camera (1 = drift alert triggered, 0 = normal)",
        ["camera_id", "signal"],  # signal: confidence, count, composite
    )

    # 8. HTTP API Requests & Latency
    HTTP_REQUESTS_TOTAL = Counter(
        "acs_http_requests_total",
        "Total HTTP requests handled by ACS FastAPI backend",
        ["method", "endpoint", "status_code"],
    )

    HTTP_REQUEST_DURATION_SECONDS = Histogram(
        "acs_http_request_duration_seconds",
        "HTTP request latency in seconds",
        ["method", "endpoint"],
        buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0),
    )


def record_camera_fps(camera_id: str, camera_name: str, fps: float) -> None:
    """Records current FPS for a camera."""
    if PROMETHEUS_AVAILABLE:
        CAMERA_FPS.labels(camera_id=camera_id, camera_name=camera_name).set(max(0.0, float(fps)))


def record_camera_connection(camera_id: str, camera_name: str, connected: bool) -> None:
    """Records connected state for a camera."""
    if PROMETHEUS_AVAILABLE:
        CAMERA_CONNECTED.labels(camera_id=camera_id, camera_name=camera_name).set(1.0 if connected else 0.0)


def record_camera_dropped_frame(camera_id: str, count: int = 1) -> None:
    """Increments dropped frame counter for a camera."""
    if PROMETHEUS_AVAILABLE and count > 0:
        CAMERA_DROPPED_FRAMES.labels(camera_id=camera_id).inc(count)


def record_camera_reconnect(camera_id: str) -> None:
    """Increments camera reconnect attempt counter."""
    if PROMETHEUS_AVAILABLE:
        CAMERA_RECONNECTS.labels(camera_id=camera_id).inc()


def record_inference_latency(camera_id: str, latency_seconds: float, model: str = "yolo") -> None:
    """Records duration of a single model inference frame."""
    if PROMETHEUS_AVAILABLE:
        INFERENCE_LATENCY.labels(camera_id=camera_id, model=model).observe(max(0.0, float(latency_seconds)))


def record_queue_depth(queue_name: str, depth: int) -> None:
    """Sets current depth for a named queue."""
    if PROMETHEUS_AVAILABLE:
        QUEUE_DEPTH.labels(queue_name=queue_name).set(max(0, int(depth)))


def record_incident(camera_id: str, severity: str, verification_status: str) -> None:
    """Increments the detected incident counter."""
    if PROMETHEUS_AVAILABLE:
        INCIDENTS_TOTAL.labels(
            camera_id=camera_id or "unknown",
            severity=severity or "medium",
            verification_status=verification_status or "unverified",
        ).inc()


def record_vlm_call(status: str, duration_seconds: Optional[float] = None, model: str = "gemini") -> None:
    """Records a VLM verification call outcome and duration."""
    if PROMETHEUS_AVAILABLE:
        VLM_CALLS_TOTAL.labels(status=status, model=model).inc()
        if duration_seconds is not None and duration_seconds >= 0:
            VLM_DURATION_SECONDS.labels(model=model).observe(float(duration_seconds))


def record_alert_delivery(channel: str, status: str, duration_seconds: Optional[float] = None) -> None:
    """Records external alert delivery attempt outcome and dispatch duration."""
    if PROMETHEUS_AVAILABLE:
        ALERT_DELIVERY_TOTAL.labels(channel=channel, status=status).inc()
        if duration_seconds is not None and duration_seconds >= 0:
            ALERT_DURATION_SECONDS.labels(channel=channel).observe(float(duration_seconds))


def record_model_drift(camera_id: str, signal: str, is_drifted: bool) -> None:
    """Sets model drift indicator for a camera and signal."""
    if PROMETHEUS_AVAILABLE:
        MODEL_DRIFT_DETECTED.labels(camera_id=camera_id, signal=signal).set(1.0 if is_drifted else 0.0)


def record_http_request(method: str, endpoint: str, status_code: int, duration_seconds: float) -> None:
    """Records HTTP request telemetry."""
    if PROMETHEUS_AVAILABLE:
        HTTP_REQUESTS_TOTAL.labels(method=method, endpoint=endpoint, status_code=str(status_code)).inc()
        HTTP_REQUEST_DURATION_SECONDS.labels(method=method, endpoint=endpoint).observe(max(0.0, float(duration_seconds)))


def get_prometheus_metrics() -> bytes:
    """Generates the latest Prometheus exposition output."""
    if not PROMETHEUS_AVAILABLE:
        return b"# prometheus_client not available\n"
    return generate_latest()
