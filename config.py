"""
config.py (Compatibility wrapper)
Re-exports configuration classes and helpers from acs.config.
"""
from acs.config import (
    config,
    AppConfig,
    CLASS_NAMES,
    ACCIDENT_CLASSES,
    auto_detect_device,
    scan_available_models,
)

__all__ = [
    "config",
    "AppConfig",
    "CLASS_NAMES",
    "ACCIDENT_CLASSES",
    "auto_detect_device",
    "scan_available_models",
]
