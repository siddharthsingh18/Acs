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

    # Inference parameters
    confidence_threshold: float = field(
        default_factory=lambda: float(os.getenv("CONFIDENCE_THRESHOLD", "0.72"))
    )
    injury_confidence_threshold: float = field(
        default_factory=lambda: float(os.getenv("INJURY_CONFIDENCE_THRESHOLD", "0.72"))
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

    # Storage output directories
    reports_dir: str = field(
        default_factory=lambda: os.getenv("REPORTS_DIR", "reports")
    )
    report_images_dir: str = field(
        default_factory=lambda: os.getenv("REPORT_IMAGES_DIR", "report_images")
    )
    temp_dir: str = field(
        default_factory=lambda: os.getenv("TEMP_DIR", "temp")
    )

    # Cooldown seconds between accident captures to prevent duplicate alerts
    cooldown_seconds: float = field(
        default_factory=lambda: float(os.getenv("COOLDOWN_SECONDS", "5.0"))
    )

    # Emergency Alert Configuration
    alerts_enabled: bool = field(
        default_factory=lambda: os.getenv("ALERTS_ENABLED", "true").lower() in ("true", "1", "yes")
    )
    alerts_dry_run: bool = field(
        default_factory=lambda: os.getenv("ALERTS_DRY_RUN", "false").lower() in ("true", "1", "yes")
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

    # Class definitions
    class_names: List[str] = field(default_factory=lambda: list(CLASS_NAMES))
    accident_classes: Set[str] = field(default_factory=lambda: set(ACCIDENT_CLASSES))

    def ensure_directories(self) -> None:
        """Create required output directories if they do not exist."""
        for directory in [self.reports_dir, self.report_images_dir, self.temp_dir]:
            os.makedirs(directory, exist_ok=True)


# Default global configuration instance
config = AppConfig()
config.ensure_directories()
