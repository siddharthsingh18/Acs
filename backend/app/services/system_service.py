"""
backend.app.services.system_service
Provides system inspection, model scanning, device detection, and class listing.
Imports acs modules directly to preserve business logic single-source-of-truth.
"""
import os
from typing import List
import torch

from acs.config import (
    ACCIDENT_CLASSES,
    CLASS_NAMES,
    AppConfig,
    scan_available_models,
)
from acs.i18n import translate_class_name
from backend.app.schemas import (
    ClassItem,
    ConfigDefaults,
    DeviceInfo,
    LanguageItem,
    SystemConfigResponse,
)


def get_device_info(cfg: AppConfig) -> DeviceInfo:
    """Inspects compute device and PyTorch acceleration support."""
    cuda_available = torch.cuda.is_available()
    mps_available = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()

    device_name = None
    if cuda_available:
        try:
            device_name = torch.cuda.get_device_name(0)
        except Exception:
            device_name = "NVIDIA CUDA Device"
    elif mps_available:
        device_name = "Apple Silicon GPU (MPS)"
    else:
        device_name = "Host CPU"

    return DeviceInfo(
        device=cfg.device,
        cuda_available=cuda_available,
        mps_available=mps_available,
        device_name=device_name,
    )


def get_class_list() -> List[ClassItem]:
    """Builds class item list with collision flags and bilingual translations."""
    items: List[ClassItem] = []
    for c in CLASS_NAMES:
        is_acc = c in ACCIDENT_CLASSES
        items.append(
            ClassItem(
                name=c,
                name_ar=translate_class_name(c, lang="ar"),
                is_accident=is_acc,
            )
        )
    return items


def get_supported_languages() -> List[LanguageItem]:
    """Returns supported languages for internationalization."""
    return [
        LanguageItem(code="en", name="English", dir="ltr"),
        LanguageItem(code="ar", name="العربية", dir="rtl"),
    ]


def get_system_config(cfg: AppConfig) -> SystemConfigResponse:
    """Aggregates all system configuration metadata into SystemConfigResponse schema."""
    available_models = scan_available_models(".")
    device_info = get_device_info(cfg)

    injury_exists = os.path.exists(cfg.injury_model_path)
    defaults = ConfigDefaults(
        model_path=cfg.model_path,
        injury_model_path=cfg.injury_model_path,
        injury_model_available=injury_exists,
        confidence_threshold=cfg.confidence_threshold,
        injury_confidence_threshold=cfg.injury_confidence_threshold,
        cooldown_seconds=cfg.cooldown_seconds,
        gemini_model_name=cfg.gemini_model_name,
        alerts_enabled=cfg.alerts_enabled,
        alerts_dry_run=cfg.alerts_dry_run,
        reports_dir=cfg.reports_dir,
        report_images_dir=cfg.report_images_dir,
        clips_dir=getattr(cfg, "clips_dir", "report_clips"),
        clip_retention_days=getattr(cfg, "clip_retention_days", 7),
        clip_max_disk_usage_mb=getattr(cfg, "clip_max_disk_usage_mb", 2048),
        severity_low_max=getattr(cfg, "severity_low_max", 35),
        severity_medium_max=getattr(cfg, "severity_medium_max", 70),
        severity_scorer_type=getattr(cfg, "severity_scorer_type", "rule_based"),
        temp_dir=cfg.temp_dir,
        demo_mode=getattr(cfg, "demo_mode", False),
        public_demo=getattr(cfg, "public_demo", False),
        store_uploads=getattr(cfg, "store_uploads", True),
    )

    return SystemConfigResponse(
        available_models=available_models,
        device_info=device_info,
        defaults=defaults,
        supported_languages=get_supported_languages(),
        classes=get_class_list(),
    )
