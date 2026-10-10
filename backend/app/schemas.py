"""
backend.app.schemas
Pydantic v2 schemas for API requests and responses.
"""
from typing import List, Optional
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
    start_timestamp: str
    end_timestamp: str
    class_name: str
    max_confidence: float
    image_path: str
    keyframe_url: str
    severity: str
    status: str
    reviewer_name: Optional[str] = None
    reviewed_at: Optional[str] = None
    dispatch_report: Optional[dict] = None
    dispatch_text: Optional[str] = None
    created_at: float
    duration: float


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


class AlertSettingsUpdateRequest(BaseModel):
    """Payload to update emergency alert configuration."""

    enabled: Optional[bool] = None
    dry_run: Optional[bool] = None
    channels: Optional[List[str]] = None
    model_path: Optional[str] = None
    confidence_threshold: Optional[float] = None
    cooldown_seconds: Optional[float] = None


class StatsResponse(BaseModel):
    """Aggregated analytical KPIs across incidents."""

    total_incidents: int
    incidents_today: int
    by_class: dict
    by_status: dict
    average_confidence: float
    average_review_time_seconds: Optional[float] = None


