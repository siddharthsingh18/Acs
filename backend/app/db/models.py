"""
backend.app.db.models
SQLModel database tables for ACS.
"""
from typing import Optional
from sqlmodel import Field, SQLModel


class Incident(SQLModel, table=True):
    """Represents a recorded accident incident stored in SQLite."""

    incident_id: str = Field(primary_key=True, index=True)
    camera_id: Optional[str] = Field(default=None, index=True)
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
    verification_status: Optional[str] = Field(default=None, index=True)  # "auto_confirmed", "needs_review", "suppressed", "unverified"
    vlm_confidence: Optional[float] = None
    vlm_reason: Optional[str] = None
    physics_passed: Optional[bool] = None
    clip_path: Optional[str] = None
    clip_annotated_path: Optional[str] = None
    impact_offset_seconds: Optional[float] = None
    severity_score: Optional[int] = Field(default=50, index=True)  # 0 to 100
    severity_breakdown: Optional[str] = None  # JSON string of breakdown components and factors
    recommended_units: Optional[str] = None  # JSON string list of recommended responder units
    reviewed_label: Optional[str] = Field(default=None, index=True)  # "accident", "not_accident", "wrong_class"
    corrected_class: Optional[str] = None
    corrected_boxes: Optional[str] = None  # JSON string of bounding boxes
    created_at: float = Field(index=True)
    duration: float = Field(default=0.0)
    is_privacy_blurred: bool = Field(default=True, index=True)
    unredacted_keyframe_path: Optional[str] = None
    unredacted_clip_path: Optional[str] = None


class ReviewFeedback(SQLModel, table=True):
    """Stores operator review decisions and ground-truth corrections for model retraining."""

    id: Optional[int] = Field(default=None, primary_key=True)
    incident_id: str = Field(index=True)
    decision: str = Field(index=True)  # "accident" (approved), "not_accident" (rejected / hard negative), "wrong_class"
    original_class: str
    corrected_class: Optional[str] = None
    confidence: float = Field(default=0.0)
    image_path: str
    clip_path: Optional[str] = None
    clip_annotated_path: Optional[str] = None
    reviewer_name: str = Field(default="Operator")
    reviewed_at: str  # ISO timestamp
    notes: Optional[str] = None
    corrected_boxes: Optional[str] = None  # JSON string of bounding boxes
    is_exported: bool = Field(default=False, index=True)
    exported_at: Optional[str] = None
    created_at: float = Field(index=True)


class Camera(SQLModel, table=True):
    """Represents a configured CCTV camera or video feed."""

    id: str = Field(primary_key=True, index=True)
    name: str = Field(index=True)
    source_type: str = Field(default="file")  # "file", "webcam", "rtsp", "http"
    source_url_encrypted: str = Field(description="Fernet-encrypted connection string/URL")
    location_lat: Optional[float] = None
    location_lng: Optional[float] = None
    enabled: bool = Field(default=True, index=True)
    confidence_override: Optional[float] = None
    created_at: float = Field(index=True)
    updated_at: float


class User(SQLModel, table=True):
    """User account model with Argon2 password hash and role-based permissions."""

    id: str = Field(primary_key=True, index=True)
    username: str = Field(unique=True, index=True)
    hashed_password: str
    full_name: str
    role: str = Field(default="viewer", index=True)  # "viewer", "operator", "admin"
    is_active: bool = Field(default=True, index=True)
    created_at: float = Field(index=True)
    last_login_at: Optional[float] = None


class AuditLog(SQLModel, table=True):
    """Append-only audit trail recording sensitive system and operator actions."""

    id: Optional[int] = Field(default=None, primary_key=True)
    timestamp: str = Field(index=True)  # ISO 8601 string
    epoch: float = Field(index=True)
    user_id: Optional[str] = Field(default=None, index=True)
    username: str = Field(index=True)
    user_role: str = Field(index=True)
    action: str = Field(index=True)  # "incident_approved", "incident_rejected", "settings_updated", etc.
    target_type: str = Field(index=True)  # "incident", "settings", "alert", "camera", "user", "auth"
    target_id: Optional[str] = Field(default=None, index=True)
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    details: Optional[str] = None  # JSON string


