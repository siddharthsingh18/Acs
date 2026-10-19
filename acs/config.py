"""
acs.config
Configuration module using a dataclass with environment variable overrides.
Defines model paths, thresholds, devices, output directories, cooldown timings,
and designated dataset class definitions.
"""
from dataclasses import dataclass, field
import os
from typing import List, Optional, Set
import torch
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# The 9 designated dataset class names
CLASS_NAMES: List[str] = [
    "bike_bike_accident",
    "bike_object_accident",
    "bike_person_accident",
    "car",
    "car_bike_accident",
    "car_car_accident",
    "car_object_accident",
    "car_person_accident",
    "person",
]

# Set containing only accident classes; normal traffic ('car', 'person', 'bike') excluded
ACCIDENT_CLASSES: Set[str] = {
    "bike_bike_accident",
    "bike_object_accident",
    "bike_person_accident",
    "car_bike_accident",
    "car_car_accident",
    "car_object_accident",
    "car_person_accident",
}


def auto_detect_device() -> str:
    """Automatically select compute device: cuda:0 if available, else mps, else cpu.

    Returns:
        str: Device identifier string ('cuda:0', 'mps', or 'cpu').
    """
    if torch.cuda.is_available():
        return "cuda:0"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def scan_available_models(directory: str = ".") -> List[str]:
    """Scans the given directory for available YOLO .pt weights files.

    Args:
        directory: Directory path to scan (defaults to current working directory).

    Returns:
        List[str]: Sorted list of available .pt filenames.
    """
    models: List[str] = []
    if os.path.exists(directory):
        for item in sorted(os.listdir(directory)):
            full_path = os.path.join(directory, item)
            if item.endswith(".pt") and os.path.isfile(full_path) and not item.startswith("."):
                models.append(item)
    return models or ["best.pt"]


@dataclass
class AppConfig:
    """Application configuration container supporting environment variable overrides."""

    # Model file paths
    model_path: str = field(
        default_factory=lambda: os.getenv("MODEL_PATH", "best.pt")
    )
    injury_model_path: str = field(
        default_factory=lambda: os.getenv("INJURY_MODEL_PATH", "injury.pt")
    )

    # Inference and Threshold Tuning Parameters (Environment-Overridable)
    confidence_threshold: float = field(
        default_factory=lambda: float(os.getenv("CONFIDENCE_THRESHOLD", "0.55"))
    )
    accident_confidence_threshold: float = field(
        default_factory=lambda: float(os.getenv("ACCIDENT_CONFIDENCE_THRESHOLD", os.getenv("CONFIDENCE_THRESHOLD", "0.55")))
    )
    normal_confidence_threshold: float = field(
        default_factory=lambda: float(os.getenv("NORMAL_CONFIDENCE_THRESHOLD", "0.35"))
    )
    injury_confidence_threshold: float = field(
        default_factory=lambda: float(os.getenv("INJURY_CONFIDENCE_THRESHOLD", "0.72"))
    )
    consecutive_frames_threshold: int = field(
        default_factory=lambda: int(os.getenv("CONSECUTIVE_FRAMES_THRESHOLD", "5"))
    )
    timeout_seconds: float = field(
        default_factory=lambda: float(os.getenv("TIMEOUT_SECONDS", "3.0"))
    )
    require_overlapping_entities: bool = field(
        default_factory=lambda: os.getenv("REQUIRE_OVERLAPPING_ENTITIES", "false").lower() in ("true", "1", "yes")
    )
    overlap_iou_threshold: float = field(
        default_factory=lambda: float(os.getenv("OVERLAP_IOU_THRESHOLD", "0.00"))
    )
    nms_iou: float = field(
        default_factory=lambda: float(os.getenv("NMS_IOU", "0.50"))
    )
    tracker_track_buffer: int = field(
        default_factory=lambda: int(os.getenv("TRACKER_TRACK_BUFFER", "30"))
    )
    tracker_match_thresh: float = field(
        default_factory=lambda: float(os.getenv("TRACKER_MATCH_THRESH", "0.80"))
    )

    # Hardware compute device
    device: str = field(
        default_factory=lambda: os.getenv("DEVICE", auto_detect_device())
    )

    # Gemini multimodal LLM model name and execution parameters
    gemini_model_name: str = field(
        default_factory=lambda: os.getenv("GEMINI_MODEL_NAME", "gemini-2.5-flash")
    )
    gemini_timeout_seconds: float = field(
        default_factory=lambda: float(os.getenv("GEMINI_TIMEOUT_SECONDS", "15.0"))
    )
    gemini_max_retries: int = field(
        default_factory=lambda: int(os.getenv("GEMINI_MAX_RETRIES", "3"))
    )

    # Second-Stage Temporal Model Gate Configuration
    temporal_model_enabled: bool = field(
        default_factory=lambda: os.getenv("TEMPORAL_MODEL_ENABLED", "true").lower() in ("true", "1", "yes")
    )
    temporal_model_path: str = field(
        default_factory=lambda: os.getenv("TEMPORAL_MODEL_PATH", "models/temporal_gate.pt")
    )
    temporal_threshold: float = field(
        default_factory=lambda: float(os.getenv("TEMPORAL_THRESHOLD", "0.50"))
    )

    # VLM Verification Stage Configuration
    vlm_verification_enabled: bool = field(
        default_factory=lambda: os.getenv("VLM_VERIFICATION_ENABLED", "true").lower() in ("true", "1", "yes")
    )
    physics_check_enabled: bool = field(
        default_factory=lambda: os.getenv("PHYSICS_CHECK_ENABLED", "true").lower() in ("true", "1", "yes")
    )
    vlm_daily_budget: int = field(
        default_factory=lambda: int(os.getenv("VLM_DAILY_BUDGET", "200"))
    )
    vlm_timeout_seconds: float = field(
        default_factory=lambda: float(os.getenv("VLM_TIMEOUT_SECONDS", "10.0"))
    )
    vlm_max_retries: int = field(
        default_factory=lambda: int(os.getenv("VLM_MAX_RETRIES", "2"))
    )
    vlm_confidence_threshold: float = field(
        default_factory=lambda: float(os.getenv("VLM_CONFIDENCE_THRESHOLD", "0.50"))
    )
    vlm_budget_file: str = field(
        default_factory=lambda: os.getenv("VLM_BUDGET_FILE", "reports/vlm_budget.json")
    )

    # Storage output directories
    reports_dir: str = field(
        default_factory=lambda: os.getenv("REPORTS_DIR", "reports")
    )
    report_images_dir: str = field(
        default_factory=lambda: os.getenv("REPORT_IMAGES_DIR", "report_images")
    )
    clips_dir: str = field(
        default_factory=lambda: os.getenv("CLIPS_DIR", "report_clips")
    )
    temp_dir: str = field(
        default_factory=lambda: os.getenv("TEMP_DIR", "temp")
    )

    # Video Clip Recording & Buffer Configuration
    clip_buffer_seconds: float = field(
        default_factory=lambda: float(os.getenv("CLIP_BUFFER_SECONDS", "10.0"))
    )
    clip_pre_incident_seconds: float = field(
        default_factory=lambda: float(os.getenv("CLIP_PRE_INCIDENT_SECONDS", "8.0"))
    )
    clip_post_incident_seconds: float = field(
        default_factory=lambda: float(os.getenv("CLIP_POST_INCIDENT_SECONDS", "10.0"))
    )
    clip_retention_days: int = field(
        default_factory=lambda: int(os.getenv("CLIP_RETENTION_DAYS", "7"))
    )
    clip_max_disk_usage_mb: int = field(
        default_factory=lambda: int(os.getenv("CLIP_MAX_DISK_USAGE_MB", "2048"))
    )

    # Multi-Factor Severity Scoring & Threshold Settings (0 to 100)
    severity_low_max: int = field(
        default_factory=lambda: int(os.getenv("SEVERITY_LOW_MAX", "35"))
    )
    severity_medium_max: int = field(
        default_factory=lambda: int(os.getenv("SEVERITY_MEDIUM_MAX", "70"))
    )
    severity_scorer_type: str = field(
        default_factory=lambda: os.getenv("SEVERITY_SCORER_TYPE", "rule_based")
    )
    camera_pixels_per_meter: Optional[float] = field(
        default_factory=lambda: (
            float(os.getenv("CAMERA_PIXELS_PER_METER"))
            if os.getenv("CAMERA_PIXELS_PER_METER") is not None
            else None
        )
    )

    # Cooldown seconds between accident captures to prevent duplicate alerts
    cooldown_seconds: float = field(
        default_factory=lambda: float(os.getenv("COOLDOWN_SECONDS", "5.0"))
    )

    # Public Demo Mode & Ephemeral Upload Policy
    demo_mode: bool = field(
        default_factory=lambda: os.getenv("DEMO_MODE", "false").lower() in ("true", "1", "yes")
    )
    public_demo: bool = field(
        default_factory=lambda: os.getenv("PUBLIC_DEMO", os.getenv("DEMO_MODE", "false")).lower() in ("true", "1", "yes")
    )
    store_uploads: bool = field(
        default_factory=lambda: os.getenv("STORE_UPLOADS", "false" if os.getenv("PUBLIC_DEMO", "false").lower() in ("true", "1", "yes") else "true").lower() in ("true", "1", "yes")
    )

    # Emergency Alert Configuration
    alerts_enabled: bool = field(
        default_factory=lambda: os.getenv("ALERTS_ENABLED", "true").lower() in ("true", "1", "yes")
    )
    alerts_dry_run: bool = field(
        default_factory=lambda: True if os.getenv("PUBLIC_DEMO", "false").lower() in ("true", "1", "yes") or os.getenv("DEMO_MODE", "false").lower() in ("true", "1", "yes") else os.getenv("ALERTS_DRY_RUN", "false").lower() in ("true", "1", "yes")
    )
    telegram_bot_token: Optional[str] = field(
        default_factory=lambda: os.getenv("TELEGRAM_BOT_TOKEN")
    )
    telegram_chat_id: Optional[str] = field(
        default_factory=lambda: os.getenv("TELEGRAM_CHAT_ID")
    )
    smtp_host: Optional[str] = field(
        default_factory=lambda: os.getenv("SMTP_HOST")
    )
    smtp_port: int = field(
        default_factory=lambda: int(os.getenv("SMTP_PORT", "587"))
    )
    smtp_user: Optional[str] = field(
        default_factory=lambda: os.getenv("SMTP_USER")
    )
    smtp_password: Optional[str] = field(
        default_factory=lambda: os.getenv("SMTP_PASSWORD")
    )
    smtp_from: Optional[str] = field(
        default_factory=lambda: os.getenv("SMTP_FROM")
    )
    smtp_to: Optional[str] = field(
        default_factory=lambda: os.getenv("SMTP_TO")
    )
    smtp_use_tls: bool = field(
        default_factory=lambda: os.getenv("SMTP_USE_TLS", "true").lower() in ("true", "1", "yes")
    )

    # Web Push (VAPID) Configuration
    vapid_public_key: Optional[str] = field(
        default_factory=lambda: os.getenv("VAPID_PUBLIC_KEY")
    )
    vapid_private_key: Optional[str] = field(
        default_factory=lambda: os.getenv("VAPID_PRIVATE_KEY")
    )
    vapid_claims_sub: str = field(
        default_factory=lambda: os.getenv("VAPID_CLAIMS_SUB", "mailto:alerts@acs-surveillance.local")
    )
    vapid_keys_file: str = field(
        default_factory=lambda: os.getenv("VAPID_KEYS_FILE", "reports/vapid_keys.json")
    )

    # Alert Escalation & Storm Rate Limiting
    escalation_enabled: bool = field(
        default_factory=lambda: os.getenv("ALERT_ESCALATION_ENABLED", "true").lower() in ("true", "1", "yes")
    )
    escalation_delay_minutes: float = field(
        default_factory=lambda: float(os.getenv("ALERT_ESCALATION_DELAY_MINUTES", "5.0"))
    )
    storm_rate_limit_seconds: float = field(
        default_factory=lambda: float(os.getenv("STORM_RATE_LIMIT_SECONDS", "60.0"))
    )
    storm_max_alerts_per_window: int = field(
        default_factory=lambda: int(os.getenv("STORM_MAX_ALERTS_PER_WINDOW", "2"))
    )
    app_base_url: str = field(
        default_factory=lambda: os.getenv("APP_BASE_URL", "http://localhost:8000")
    )

    # Privacy Protection & Retention Configuration
    privacy_blur_enabled: bool = field(
        default_factory=lambda: os.getenv("PRIVACY_BLUR_ENABLED", "true").lower() in ("true", "1", "yes")
    )
    privacy_blur_faces: bool = field(
        default_factory=lambda: os.getenv("PRIVACY_BLUR_FACES", "true").lower() in ("true", "1", "yes")
    )
    privacy_blur_plates: bool = field(
        default_factory=lambda: os.getenv("PRIVACY_BLUR_PLATES", "true").lower() in ("true", "1", "yes")
    )
    privacy_preserve_unblurred_original: bool = field(
        default_factory=lambda: os.getenv("PRIVACY_PRESERVE_UNBLURRED_ORIGINAL", "true").lower() in ("true", "1", "yes")
    )
    privacy_unredacted_dir: str = field(
        default_factory=lambda: os.getenv("PRIVACY_UNREDACTED_DIR", "media/unredacted")
    )
    incident_retention_days: int = field(
        default_factory=lambda: int(os.getenv("INCIDENT_RETENTION_DAYS", "30"))
    )

    # Class definitions
    class_names: List[str] = field(default_factory=lambda: list(CLASS_NAMES))
    accident_classes: Set[str] = field(default_factory=lambda: set(ACCIDENT_CLASSES))

    def ensure_directories(self) -> None:
        """Create required output directories if they do not exist."""
        for directory in [
            self.reports_dir,
            self.report_images_dir,
            self.clips_dir,
            self.temp_dir,
            self.privacy_unredacted_dir,
            os.path.join(self.privacy_unredacted_dir, "keyframes"),
            os.path.join(self.privacy_unredacted_dir, "clips"),
        ]:
            os.makedirs(directory, exist_ok=True)


# Default global configuration instance
config = AppConfig()
config.ensure_directories()
