"""
backend.app.schemas
Pydantic v2 schemas for API requests and responses.
"""
from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    """Health check status response."""

    status: str = Field(default="ok", description="Health status string")
    service: str = Field(default="acs-backend", description="Backend service name")
    version: str = Field(default="1.0.0", description="API version")
    timestamp: str = Field(description="ISO 8601 UTC timestamp")


class DeviceInfo(BaseModel):
    """Compute device information."""

    device: str = Field(description="Active compute device identifier (e.g. mps, cuda:0, cpu)")
    cuda_available: bool = Field(description="Whether CUDA acceleration is available")
    mps_available: bool = Field(description="Whether Apple Silicon MPS is available")
    device_name: Optional[str] = Field(default=None, description="Human-readable device name")


class ClassItem(BaseModel):
    """Model class item with localized translation and collision designation."""

    name: str = Field(description="Raw class label name in YOLO model")
    name_ar: str = Field(description="Localized Arabic class name")
    is_accident: bool = Field(
        description="True if designated as an accident class, False for normal traffic"
    )


class LanguageItem(BaseModel):
    """Supported language descriptor."""

    code: str = Field(description="Language code ('en', 'ar')")
    name: str = Field(description="Language display name")
    dir: str = Field(description="Text direction ('ltr' or 'rtl')")


class ConfigDefaults(BaseModel):
    """Current application default configuration values."""

    model_path: str = Field(description="Default primary YOLO weights file")
    injury_model_path: str = Field(description="Configured secondary injury model file")
    injury_model_available: bool = Field(description="True if injury model file exists on disk")
    confidence_threshold: float = Field(description="Detection confidence threshold")
    injury_confidence_threshold: float = Field(description="Injury detection threshold")
    cooldown_seconds: float = Field(description="Cooldown delay between incidents in seconds")
    gemini_model_name: str = Field(description="Gemini multimodal model name")
    alerts_enabled: bool = Field(description="Whether emergency alert dispatch is active")
    alerts_dry_run: bool = Field(description="Whether alert dispatch is in simulation mode")
    reports_dir: str = Field(description="Directory storing incident CSV reports")
    report_images_dir: str = Field(description="Directory storing incident keyframes")
    clips_dir: str = Field(description="Directory storing incident video clips")
    clip_retention_days: int = Field(description="Days before stored video clips are deleted")
    clip_max_disk_usage_mb: int = Field(description="Maximum MB disk usage for stored clips")
    severity_low_max: int = Field(default=35, description="Upper threshold score for low severity tier")
    severity_medium_max: int = Field(default=70, description="Upper threshold score for medium severity tier")
    severity_scorer_type: str = Field(default="rule_based", description="Active severity scorer implementation")
    temp_dir: str = Field(description="Directory storing temporary uploads")


class SystemConfigResponse(BaseModel):
    """Complete system configuration response for GET /api/config."""

    available_models: List[str] = Field(
        description="List of available .pt YOLO checkpoints in repository"
    )
    device_info: DeviceInfo = Field(description="Compute hardware status")
    defaults: ConfigDefaults = Field(description="Current system configuration defaults")
    supported_languages: List[LanguageItem] = Field(description="Supported localization languages")
    classes: List[ClassItem] = Field(
        description="List of all supported classes and collision designations"
    )


class JobCreateRequest(BaseModel):
    """JSON payload for initiating a video analysis job."""

    source: Optional[str] = Field(
        default=None, description="Set to 'sample' to use the bundled demo video (acc_ex.mp4)"
    )
    model_name: Optional[str] = Field(
        default=None, description="YOLO weights file to use (e.g. 'best.pt')"
    )
    confidence: Optional[float] = Field(
        default=None, ge=0.05, le=1.0, description="Detection confidence threshold"
    )
    use_injury_model: Optional[bool] = Field(
        default=False, description="Whether to run the secondary injury model"
    )
    max_frames: Optional[int] = Field(
        default=None, description="Optional cap on number of frames to analyze (useful for tests)"
    )


class JobCreatedResponse(BaseModel):
    """Response returned upon successfully queuing a video analysis job."""

    job_id: str = Field(description="Unique identifier for the analysis job")
    status: str = Field(default="pending", description="Initial job status ('pending')")
    video_source: str = Field(description="Source identifier or video filename")
    model_name: str = Field(description="Selected YOLO model weights")
    confidence: float = Field(description="Detection confidence threshold used")


class JobSummary(BaseModel):
    """Post-analysis execution summary statistics."""

    job_id: str
    status: str
    frames_processed: int
    total_frames: int
    fps: float
    incidents_count: int
    elapsed_seconds: float


class JobCancelResponse(BaseModel):
    """Response returned upon cancelling an analysis job."""

    job_id: str
    status: str = Field(description="Updated job status ('cancelled')")
    message: str


class IncidentItem(BaseModel):
    """Structured incident record emitted during streaming."""

    incident_id: str
    class_name: str
    max_confidence: float
    start_timestamp: str
    end_timestamp: str
    start_time: float
    end_time: float
    duration: float
    keyframe_url: str
    verification_status: Optional[str] = None
    severity: Optional[str] = None
    severity_score: Optional[int] = None
    recommended_units: Optional[List[str]] = None
    clip_url: Optional[str] = None
    clip_annotated_url: Optional[str] = None


class JobStatusResponse(BaseModel):
    """Detailed job status and summary report."""

    job_id: str
    status: str
    created_at: float
    video_source: str
    model_name: str
    confidence: float
    incidents: List[IncidentItem] = Field(default_factory=list)
    summary: Optional[JobSummary] = None
    error: Optional[str] = None


class IncidentResponse(BaseModel):
    """Full incident record for detail and list APIs."""

    incident_id: str
    camera_id: Optional[str] = None
    start_timestamp: str
    end_timestamp: str
    class_name: str
    max_confidence: float
    image_path: str
    keyframe_url: str
    severity: str
    severity_score: Optional[int] = 50
    severity_breakdown: Optional[dict] = None
    recommended_units: Optional[List[str]] = None
    status: str
    reviewer_name: Optional[str] = None
    reviewed_at: Optional[str] = None
    dispatch_report: Optional[dict] = None
    dispatch_text: Optional[str] = None
    verification_status: Optional[str] = None
    vlm_confidence: Optional[float] = None
    vlm_reason: Optional[str] = None
    physics_passed: Optional[bool] = None
    clip_path: Optional[str] = None
    clip_url: Optional[str] = None
    clip_annotated_path: Optional[str] = None
    clip_annotated_url: Optional[str] = None
    impact_offset_seconds: Optional[float] = None
    reviewed_label: Optional[str] = None
    corrected_class: Optional[str] = None
    corrected_boxes: Optional[str] = None
    created_at: float
    duration: float
    is_privacy_blurred: Optional[bool] = True
    has_unredacted_original: Optional[bool] = False
    is_acknowledged: Optional[bool] = False
    acknowledged_at: Optional[str] = None
    acknowledged_by: Optional[str] = None
    ack_token: Optional[str] = None
    escalation_tier: Optional[int] = 1
    is_demo: Optional[bool] = False


class IncidentListResponse(BaseModel):
    """Paginated list of incidents."""

    items: List[IncidentResponse]
    total: int
    page: int
    page_size: int
    total_pages: int


class IncidentStatusUpdateRequest(BaseModel):
    """Human-in-the-loop review status update."""

    status: str = Field(
        pattern="^(approved|rejected|dispatched)$",
        description="Updated status ('approved', 'rejected', or 'dispatched')",
    )
    reviewer_name: Optional[str] = Field(
        default="Reviewing Officer",
        description="Name or badge number of reviewer",
    )
    notes: Optional[str] = None
    corrected_class: Optional[str] = None
    corrected_boxes: Optional[Union[str, List[Any]]] = None


class ReviewDecisionRequest(BaseModel):
    """Review decision submitted from the fast swipe-style review queue."""

    decision: str = Field(
        pattern="^(accident|not_accident|wrong_class)$",
        description="Decision: 'accident' (approve), 'not_accident' (hard negative), or 'wrong_class'",
    )
    corrected_class: Optional[str] = Field(
        default=None,
        description="Corrected class label if wrong_class or specified",
    )
    corrected_boxes: Optional[Union[str, List[Any]]] = Field(
        default=None,
        description="Optional corrected bounding boxes (YOLO format or normalized JSON)",
    )
    reviewer_name: Optional[str] = Field(
        default="Operator",
        description="Name or badge number of operator",
    )
    notes: Optional[str] = Field(
        default=None,
        description="Optional reviewer notes or context",
    )


class ReviewFeedbackResponse(BaseModel):
    """Record of an operator review decision stored for ML dataset export."""

    id: Optional[int] = None
    incident_id: str
    decision: str
    original_class: str
    corrected_class: Optional[str] = None
    confidence: float
    image_path: str
    clip_path: Optional[str] = None
    clip_annotated_path: Optional[str] = None
    reviewer_name: str
    reviewed_at: str
    notes: Optional[str] = None
    corrected_boxes: Optional[str] = None
    is_exported: bool
    exported_at: Optional[str] = None
    created_at: float


class ReviewQueueItem(BaseModel):
    """Candidate detection item presented in the fast review queue."""

    incident_id: str
    class_name: str
    max_confidence: float
    image_path: str
    keyframe_url: str
    clip_path: Optional[str] = None
    clip_url: Optional[str] = None
    clip_annotated_path: Optional[str] = None
    clip_annotated_url: Optional[str] = None
    verification_status: Optional[str] = None
    vlm_confidence: Optional[float] = None
    vlm_reason: Optional[str] = None
    physics_passed: Optional[bool] = None
    uncertainty_reason: str
    start_timestamp: str
    severity: Optional[str] = None
    severity_score: Optional[int] = None
    duration: float = 0.0


class ReviewQueueResponse(BaseModel):
    """List of uncertain detections queued for review."""

    items: List[ReviewQueueItem]
    total: int


class ReviewStatsResponse(BaseModel):
    """Summary of review decisions and dataset staging readiness."""

    total_reviewed: int
    accidents_approved: int
    hard_negatives: int
    wrong_class_count: int
    staging_pending_export: int


class AlertTestRequest(BaseModel):
    """Payload to trigger test alert."""

    dry_run: Optional[bool] = False
    lang: Optional[str] = "en"
    channels: Optional[List[str]] = None


class AlertTestResponse(BaseModel):
    """Delivery status results for test alert."""

    results: dict
    message: str


class ChannelStatusInfo(BaseModel):
    """Configuration status for an alert channel without disclosing credentials."""

    configured: bool = Field(description="True if all required credentials are set in environment")
    description: str = Field(description="Human-readable configuration state")


class AlertSettingsResponse(BaseModel):
    """Current emergency alert system configuration."""

    enabled: bool
    dry_run: bool
    channels: List[str]
    channel_status: Optional[dict[str, ChannelStatusInfo]] = Field(
        default_factory=dict,
        description="Configuration readiness per channel without exposing secrets",
    )
    vapid_public_key: Optional[str] = Field(default=None, description="VAPID public key for Web Push subscription")
    escalation_enabled: Optional[bool] = Field(default=True, description="Whether automated multi-tier escalation is enabled")
    escalation_delay_minutes: Optional[float] = Field(default=5.0, description="Minutes before unacknowledged incidents escalate to Tier 2")


class AlertSettingsUpdateRequest(BaseModel):
    """Payload to update emergency alert configuration."""

    enabled: Optional[bool] = None
    dry_run: Optional[bool] = None
    channels: Optional[List[str]] = None
    model_path: Optional[str] = None
    confidence_threshold: Optional[float] = None
    cooldown_seconds: Optional[float] = None
    escalation_enabled: Optional[bool] = None
    escalation_delay_minutes: Optional[float] = None


class AlertDeliveryResponse(BaseModel):
    """Delivery status record for a notification channel."""

    id: Optional[int] = None
    incident_id: str
    channel: str
    tier: int = 1
    status: str  # sent, failed, acknowledged, dry_run, suppressed
    details: Optional[str] = None
    created_at: float
    acknowledged_at: Optional[float] = None
    acknowledged_by: Optional[str] = None


class WebPushSubscribeRequest(BaseModel):
    """Payload to register a Web Push browser subscription."""

    endpoint: str
    keys: dict  # {"p256dh": "...", "auth": "..."}
    user_agent: Optional[str] = None


class WebPushUnsubscribeRequest(BaseModel):
    """Payload to unsubscribe Web Push browser endpoint."""

    endpoint: str


class IncidentAcknowledgeRequest(BaseModel):
    """Payload for operator acknowledgment."""

    notes: Optional[str] = None


class StatsResponse(BaseModel):
    """Aggregated analytical KPIs across incidents."""

    total_incidents: int
    incidents_today: int
    by_class: dict
    by_status: dict
    average_confidence: float
    average_review_time_seconds: Optional[float] = None


# -----------------------------------------------------------------------------
# Live Camera Feed Schemas
# -----------------------------------------------------------------------------
class CameraHealth(BaseModel):
    """Live telemetry and health indicators for an active camera worker."""

    connected: bool = Field(description="Whether the camera stream is currently open and delivering frames")
    fps: float = Field(description="Current processing throughput frames-per-second")
    last_frame_age: float = Field(description="Elapsed seconds since the most recent frame was received")
    dropped_frames: int = Field(description="Cumulative count of skipped or dropped frames under load")
    reconnect_count: int = Field(description="Number of stream disconnect & auto-reconnect attempts")
    is_incident_active: bool = Field(description="True if an accident incident is actively ongoing on this camera")
    active_incident_id: Optional[str] = Field(default=None, description="Active incident tracking ID if ongoing")
    last_accident_class: Optional[str] = Field(default=None, description="Class label of the most recent accident")
    last_accident_confidence: Optional[float] = Field(default=None, description="Peak confidence of active accident")


class CameraCreateRequest(BaseModel):
    """Payload for registering a new live CCTV camera feed."""

    name: str = Field(description="Human-readable camera title or intersection name")
    source_url: str = Field(
        description="Connection URL or device index (e.g. '0', 'rtsp://...', 'http://...', or file path)"
    )
    location_lat: Optional[float] = Field(default=None, description="Camera GPS latitude coordinate")
    location_lng: Optional[float] = Field(default=None, description="Camera GPS longitude coordinate")
    enabled: Optional[bool] = Field(default=True, description="Whether camera worker should be active")
    confidence_override: Optional[float] = Field(
        default=None, ge=0.05, le=1.0, description="Optional per-camera accident confidence threshold override"
    )


class CameraUpdateRequest(BaseModel):
    """Payload for updating camera configuration attributes."""

    name: Optional[str] = Field(default=None, description="Updated camera title")
    source_url: Optional[str] = Field(default=None, description="Updated connection string/URL")
    location_lat: Optional[float] = Field(default=None, description="Updated latitude")
    location_lng: Optional[float] = Field(default=None, description="Updated longitude")
    enabled: Optional[bool] = Field(default=None, description="Enable or disable camera worker")
    confidence_override: Optional[float] = Field(
        default=None, ge=0.05, le=1.0, description="Updated confidence override"
    )


class CameraResponse(BaseModel):
    """Camera descriptor with credentials safely masked and live health metrics attached."""

    id: str = Field(description="Unique camera identifier")
    name: str = Field(description="Camera title")
    source_type: str = Field(description="Source classification: 'file', 'webcam', 'rtsp', or 'http'")
    source_url: str = Field(
        description="Masked connection string (passwords & usernames are strictly redacted)"
    )
    location_lat: Optional[float] = Field(default=None, description="GPS latitude")
    location_lng: Optional[float] = Field(default=None, description="GPS longitude")
    enabled: bool = Field(description="Whether camera is currently active")
    confidence_override: Optional[float] = Field(default=None, description="Custom confidence override")
    created_at: float = Field(description="Creation epoch timestamp")
    updated_at: float = Field(description="Last update epoch timestamp")
    health: Optional[CameraHealth] = Field(default=None, description="Real-time stream health telemetry")


class RetentionSettingsResponse(BaseModel):
    """Current clip retention rules and disk storage metrics."""

    retention_days: int = Field(description="Days before stored video clips are deleted")
    max_disk_usage_mb: int = Field(description="Maximum disk storage in MB allocated for clips")
    current_disk_usage_mb: float = Field(description="Current disk storage in MB used by clips")
    total_clips_count: int = Field(description="Total number of stored video clip files")
    clips_dir: str = Field(description="Filesystem directory where video clips are stored")
    privacy_blur_enabled: Optional[bool] = Field(default=True, description="Whether face and plate privacy blur is active")
    privacy_preserve_unblurred_original: Optional[bool] = Field(default=True, description="Whether unredacted originals are vaulted")
    incident_retention_days: Optional[int] = Field(default=30, description="Days before incident database records and keyframes are purged")


class RetentionSettingsUpdateRequest(BaseModel):
    """Payload to update video clip retention policies and trigger manual cleanup."""

    retention_days: Optional[int] = Field(default=None, ge=1, le=365, description="New max clip age in days")
    max_disk_usage_mb: Optional[int] = Field(default=None, ge=50, le=1000000, description="New max clip disk usage cap in MB")
    trigger_cleanup: Optional[bool] = Field(default=True, description="Whether to run cleanup immediately upon save")
    privacy_blur_enabled: Optional[bool] = Field(default=None, description="Toggle privacy blurring")
    privacy_preserve_unblurred_original: Optional[bool] = Field(default=None, description="Toggle unredacted vaulting")
    incident_retention_days: Optional[int] = Field(default=None, ge=1, le=3650, description="New max incident database age in days")


class UserResponse(BaseModel):
    """Public user descriptor without password hash."""

    id: str = Field(description="Unique user ID")
    username: str = Field(description="User login username")
    full_name: str = Field(description="User display full name")
    role: str = Field(description="Security role: viewer, operator, admin")
    is_active: bool = Field(description="Whether account is active")
    created_at: float = Field(description="Account creation epoch")
    last_login_at: Optional[float] = Field(default=None, description="Last successful login epoch")


class LoginRequest(BaseModel):
    """Credentials payload for JWT authentication."""

    username: str = Field(description="Account username")
    password: str = Field(description="Account password")


class TokenResponse(BaseModel):
    """OAuth2 compatible JWT token response with user profile."""

    access_token: str = Field(description="Signed JWT access token")
    refresh_token: str = Field(description="Signed JWT refresh token")
    token_type: str = Field(default="bearer", description="Token type descriptor")
    user: UserResponse = Field(description="Authenticated user details")


class RefreshTokenRequest(BaseModel):
    """Payload to exchange a refresh token for a fresh access token."""

    refresh_token: Optional[str] = Field(default=None, description="Refresh token string (if not in cookie)")


class UserCreateRequest(BaseModel):
    """Payload to register a new user (Admin only)."""

    username: str = Field(min_length=3, max_length=50, description="Unique username")
    password: str = Field(min_length=8, description="Initial account password")
    full_name: str = Field(min_length=2, max_length=100, description="User full name")
    role: str = Field(default="viewer", description="Role: viewer, operator, or admin")


class UserUpdateRequest(BaseModel):
    """Payload to modify existing user details or credentials (Admin only)."""

    full_name: Optional[str] = Field(default=None, min_length=2, max_length=100)
    role: Optional[str] = Field(default=None, description="viewer, operator, or admin")
    is_active: Optional[bool] = Field(default=None)
    password: Optional[str] = Field(default=None, min_length=8, description="Optional new password")


class AuditLogResponse(BaseModel):
    """Single audit event log entry."""

    id: int = Field(description="Unique audit event sequence ID")
    timestamp: str = Field(description="ISO 8601 UTC timestamp")
    epoch: float = Field(description="Event timestamp epoch")
    user_id: Optional[str] = Field(default=None, description="ID of acting user")
    username: str = Field(description="Username of actor")
    user_role: str = Field(description="Role of actor at time of action")
    action: str = Field(description="Action name identifier")
    target_type: str = Field(description="Category of affected resource")
    target_id: Optional[str] = Field(default=None, description="Resource identifier")
    ip_address: Optional[str] = Field(default=None, description="Client IPv4 or IPv6 address")
    user_agent: Optional[str] = Field(default=None, description="Client HTTP user-agent")
    details: Optional[str] = Field(default=None, description="JSON string with additional contextual parameters")


class AuditLogListResponse(BaseModel):
    """Paginated collection of audit log events."""

    items: List[AuditLogResponse] = Field(description="List of audit log records")
    total: int = Field(description="Total matching audit records")
    skip: int = Field(description="Pagination offset")
    limit: int = Field(description="Pagination limit")





