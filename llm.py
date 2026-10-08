"""
llm.py (Compatibility wrapper)
Re-exports LLM dispatch reporting functions from acs.llm.
"""
from acs.llm import (
    DispatchReportModel,
    build_dispatch_prompt,
    build_fallback_report,
    find_surrounding_frames,
    generate_accident_report,
    generate_dispatch_report,
    render_dispatch_text,
)

__all__ = [
    "DispatchReportModel",
    "build_dispatch_prompt",
    "build_fallback_report",
    "find_surrounding_frames",
    "generate_accident_report",
    "generate_dispatch_report",
    "render_dispatch_text",
]
