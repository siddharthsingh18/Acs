"""
backend.app.db.models
SQLModel database tables for ACS.
"""
from typing import Optional
from sqlalchemy import Column, Index
from sqlmodel import Field, SQLModel
from backend.app.db.types import UTCEpoch, UTCISO


class Incident(SQLModel, table=True):
    """Represents a recorded accident incident."""

    __table_args__ = (
        Index("ix_incident_camera_start", "camera_id", "start_timestamp"),
        Index("ix_incident_status_created", "status", "created_at"),
        Index("ix_incident_severity_created", "severity", "created_at"),
    )

    incident_id: str = Field(primary_key=True, index=True)
    camera_id: Optional[str] = Field(default=None, index=True)
    start_timestamp: str = Field(sa_column=Column(UTCISO(), nullable=False, index=True))
    end_timestamp: str = Field(sa_column=Column(UTCISO(), nullable=False))
    class_name: str = Field(index=True)
    incident_type: Optional[str] = Field(default=None, index=True)
    max_confidence: float
    image_path: str
    keyframe_url: str
    severity: str = Field(default="medium", index=True)  # "low", "medium", "high"
    status: str = Field(default="pending", index=True)  # "pending", "approved", "rejected", "dispatched"
    reviewer_name: Optional[str] = None
    reviewed_at: Optional[str] = Field(default=None, sa_column=Column(UTCISO(), nullable=True))
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
    created_at: float = Field(sa_column=Column(UTCEpoch(), nullable=False, index=True))
    duration: float = Field(default=0.0)
    is_privacy_blurred: bool = Field(default=True, index=True)
    unredacted_keyframe_path: Optional[str] = None
    unredacted_clip_path: Optional[str] = None
    is_acknowledged: bool = Field(default=False, index=True)
    acknowledged_at: Optional[str] = Field(default=None, sa_column=Column(UTCISO(), nullable=True))
    acknowledged_by: Optional[str] = None
    ack_token: Optional[str] = Field(default=None, index=True)
    escalation_tier: int = Field(default=1, index=True)
    last_escalated_at: Optional[float] = Field(default=None, sa_column=Column(UTCEpoch(), nullable=True))
    is_demo: bool = Field(default=False, index=True)


class WebPushSubscription(SQLModel, table=True):
    """Stores Web Push API browser and device endpoints for PWA push alerts."""

    id: str = Field(primary_key=True, index=True)
    endpoint: str = Field(unique=True, index=True)
    p256dh: str
    auth: str
    user_id: Optional[str] = Field(default=None, index=True)
    user_agent: Optional[str] = None
    created_at: float = Field(sa_column=Column(UTCEpoch(), nullable=False, index=True))
    is_active: bool = Field(default=True, index=True)


class AlertDelivery(SQLModel, table=True):
    """Tracks per-channel delivery and acknowledgment state for each incident alert."""

    __table_args__ = (
        Index("ix_alert_delivery_incident_channel_tier", "incident_id", "channel", "tier"),
    )

    id: Optional[int] = Field(default=None, primary_key=True)
    incident_id: str = Field(index=True)
    channel: str = Field(index=True)  # "webpush", "telegram", "email", "console"
    tier: int = Field(default=1, index=True)  # 1 for initial tier, 2 for escalated tier
    status: str = Field(index=True)  # "sent", "failed", "acknowledged", "dry_run", "suppressed"
    details: Optional[str] = None  # JSON string or explanation
    created_at: float = Field(sa_column=Column(UTCEpoch(), nullable=False, index=True))
    acknowledged_at: Optional[float] = Field(default=None, sa_column=Column(UTCEpoch(), nullable=True))
    acknowledged_by: Optional[str] = None


class ReviewFeedback(SQLModel, table=True):
    """Stores operator review decisions and ground-truth corrections for model retraining."""

    __table_args__ = (
        Index("ix_review_feedback_exported_created", "is_exported", "created_at"),
    )

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
    reviewed_at: str = Field(sa_column=Column(UTCISO(), nullable=False))
    notes: Optional[str] = None
    corrected_boxes: Optional[str] = None  # JSON string of bounding boxes
    is_exported: bool = Field(default=False, index=True)
    exported_at: Optional[str] = Field(default=None, sa_column=Column(UTCISO(), nullable=True))
    created_at: float = Field(sa_column=Column(UTCEpoch(), nullable=False, index=True))


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
    created_at: float = Field(sa_column=Column(UTCEpoch(), nullable=False, index=True))
    updated_at: float = Field(sa_column=Column(UTCEpoch(), nullable=False))


class User(SQLModel, table=True):
    """User account model with Argon2 password hash and role-based permissions."""

    id: str = Field(primary_key=True, index=True)
    username: str = Field(unique=True, index=True)
    hashed_password: str
    full_name: str
    role: str = Field(default="viewer", index=True)  # "viewer", "operator", "admin"
    is_active: bool = Field(default=True, index=True)
    created_at: float = Field(sa_column=Column(UTCEpoch(), nullable=False, index=True))
    last_login_at: Optional[float] = Field(default=None, sa_column=Column(UTCEpoch(), nullable=True))


class AuditLog(SQLModel, table=True):
    """Append-only audit trail recording sensitive system and operator actions."""

    id: Optional[int] = Field(default=None, primary_key=True)
    timestamp: str = Field(sa_column=Column(UTCISO(), nullable=False, index=True))
    epoch: float = Field(sa_column=Column(UTCEpoch(), nullable=False, index=True))
    user_id: Optional[str] = Field(default=None, index=True)
    username: str = Field(index=True)
    user_role: str = Field(index=True)
    action: str = Field(index=True)  # "incident_approved", "incident_rejected", "settings_updated", etc.
    target_type: str = Field(index=True)  # "incident", "settings", "alert", "camera", "user", "auth"
    target_id: Optional[str] = Field(default=None, index=True)
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    details: Optional[str] = None  # JSON string
