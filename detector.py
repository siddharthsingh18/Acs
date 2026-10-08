"""
detector.py (Compatibility wrapper)
Re-exports detector classes and functions from acs.detector.
"""
from acs.detector import (
    AccidentDetector,
    AccidentEventManager,
    AccidentIncident,
    DetectionBox,
    DetectionResult,
    load_model,
)

__all__ = [
    "AccidentDetector",
    "AccidentEventManager",
    "AccidentIncident",
    "DetectionBox",
    "DetectionResult",
    "load_model",
]
