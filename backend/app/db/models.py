"""
backend.app.db.models
SQLModel database tables for ACS.
"""
from typing import Optional
from sqlmodel import Field, SQLModel


class Incident(SQLModel, table=True):
    """Represents a recorded accident incident stored in SQLite."""

    incident_id: str = Field(primary_key=True, index=True)
    start_timestamp: str = Field(index=True)
    end_timestamp: str
    class_name: str = Field(index=True)
    max_confidence: float
    image_path: str
    keyframe_url: str
    severity: str = Field(default="medium", index=True)  # "low", "medium", "high"
    status: str = Field(default="pending", index=True)  # "pending", "approved", "rejected", "dispatched"
    reviewer_name: Optional[str] = None
    reviewed_at: Optional[str] = None
    dispatch_report: Optional[str] = None  # JSON string of structured dispatch report
    dispatch_text: Optional[str] = None  # Rendered markdown text summary
    created_at: float = Field(index=True)
    duration: float = Field(default=0.0)
