"""Initial schema generated from the current SQLModel entities.

Revision ID: 001_initial_schema
Revises:
"""
from alembic import op
import sqlalchemy as sa


revision = "001_initial_schema"
down_revision = None
branch_labels = None
depends_on = None


def _utc_datetime():
    return sa.DateTime(timezone=True).with_variant(sa.String(40), "sqlite")


def upgrade() -> None:
    op.create_table(
        "incident",
        sa.Column("incident_id", sa.String(), primary_key=True),
        sa.Column("camera_id", sa.String(), nullable=True),
        sa.Column("start_timestamp", _utc_datetime(), nullable=False),
        sa.Column("end_timestamp", _utc_datetime(), nullable=False),
        sa.Column("class_name", sa.String(), nullable=False),
        sa.Column("incident_type", sa.String(), nullable=True),
        sa.Column("max_confidence", sa.Float(), nullable=False),
        sa.Column("image_path", sa.String(), nullable=False),
        sa.Column("keyframe_url", sa.String(), nullable=False),
        sa.Column("severity", sa.String(), server_default="medium", nullable=False),
        sa.Column("status", sa.String(), server_default="pending", nullable=False),
        sa.Column("reviewer_name", sa.String(), nullable=True),
        sa.Column("reviewed_at", _utc_datetime(), nullable=True),
        sa.Column("dispatch_report", sa.String(), nullable=True),
        sa.Column("dispatch_text", sa.String(), nullable=True),
        sa.Column("verification_status", sa.String(), nullable=True),
        sa.Column("vlm_confidence", sa.Float(), nullable=True),
        sa.Column("vlm_reason", sa.String(), nullable=True),
        sa.Column("physics_passed", sa.Boolean(), nullable=True),
        sa.Column("clip_path", sa.String(), nullable=True),
        sa.Column("clip_annotated_path", sa.String(), nullable=True),
        sa.Column("impact_offset_seconds", sa.Float(), nullable=True),
        sa.Column("severity_score", sa.Integer(), server_default="50", nullable=False),
        sa.Column("severity_breakdown", sa.String(), nullable=True),
        sa.Column("recommended_units", sa.String(), nullable=True),
        sa.Column("reviewed_label", sa.String(), nullable=True),
        sa.Column("corrected_class", sa.String(), nullable=True),
        sa.Column("corrected_boxes", sa.String(), nullable=True),
        sa.Column("created_at", _utc_datetime(), nullable=False),
        sa.Column("duration", sa.Float(), server_default="0", nullable=False),
        sa.Column("is_privacy_blurred", sa.Boolean(), server_default=sa.true(), nullable=False),
        sa.Column("unredacted_keyframe_path", sa.String(), nullable=True),
        sa.Column("unredacted_clip_path", sa.String(), nullable=True),
        sa.Column("is_acknowledged", sa.Boolean(), server_default=sa.false(), nullable=False),
        sa.Column("acknowledged_at", _utc_datetime(), nullable=True),
        sa.Column("acknowledged_by", sa.String(), nullable=True),
        sa.Column("ack_token", sa.String(), nullable=True),
        sa.Column("escalation_tier", sa.Integer(), server_default="1", nullable=False),
        sa.Column("last_escalated_at", _utc_datetime(), nullable=True),
        sa.Column("is_demo", sa.Boolean(), server_default=sa.false(), nullable=False),
    )
    op.create_table(
        "webpushsubscription",
        sa.Column("id", sa.String(), primary_key=True),
        sa.Column("endpoint", sa.String(), nullable=False, unique=True),
        sa.Column("p256dh", sa.String(), nullable=False),
        sa.Column("auth", sa.String(), nullable=False),
        sa.Column("user_id", sa.String(), nullable=True),
        sa.Column("user_agent", sa.String(), nullable=True),
        sa.Column("created_at", _utc_datetime(), nullable=False),
        sa.Column("is_active", sa.Boolean(), server_default=sa.true(), nullable=False),
    )
    op.create_table(
        "alertdelivery",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("incident_id", sa.String(), nullable=False),
        sa.Column("channel", sa.String(), nullable=False),
        sa.Column("tier", sa.Integer(), server_default="1", nullable=False),
        sa.Column("status", sa.String(), nullable=False),
        sa.Column("details", sa.String(), nullable=True),
        sa.Column("created_at", _utc_datetime(), nullable=False),
        sa.Column("acknowledged_at", _utc_datetime(), nullable=True),
        sa.Column("acknowledged_by", sa.String(), nullable=True),
    )
    op.create_table(
        "reviewfeedback",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("incident_id", sa.String(), nullable=False),
        sa.Column("decision", sa.String(), nullable=False),
        sa.Column("original_class", sa.String(), nullable=False),
        sa.Column("corrected_class", sa.String(), nullable=True),
        sa.Column("confidence", sa.Float(), server_default="0", nullable=False),
        sa.Column("image_path", sa.String(), nullable=False),
        sa.Column("clip_path", sa.String(), nullable=True),
        sa.Column("clip_annotated_path", sa.String(), nullable=True),
        sa.Column("reviewer_name", sa.String(), server_default="Operator", nullable=False),
        sa.Column("reviewed_at", _utc_datetime(), nullable=False),
        sa.Column("notes", sa.String(), nullable=True),
        sa.Column("corrected_boxes", sa.String(), nullable=True),
        sa.Column("is_exported", sa.Boolean(), server_default=sa.false(), nullable=False),
        sa.Column("exported_at", _utc_datetime(), nullable=True),
        sa.Column("created_at", _utc_datetime(), nullable=False),
    )
    op.create_table(
        "camera",
        sa.Column("id", sa.String(), primary_key=True),
        sa.Column("name", sa.String(), nullable=False),
        sa.Column("source_type", sa.String(), server_default="file", nullable=False),
        sa.Column("source_url_encrypted", sa.String(), nullable=False),
        sa.Column("location_lat", sa.Float(), nullable=True),
        sa.Column("location_lng", sa.Float(), nullable=True),
        sa.Column("enabled", sa.Boolean(), server_default=sa.true(), nullable=False),
        sa.Column("confidence_override", sa.Float(), nullable=True),
        sa.Column("created_at", _utc_datetime(), nullable=False),
        sa.Column("updated_at", _utc_datetime(), nullable=False),
    )
    op.create_table(
        "user",
        sa.Column("id", sa.String(), primary_key=True),
        sa.Column("username", sa.String(), nullable=False, unique=True),
        sa.Column("hashed_password", sa.String(), nullable=False),
        sa.Column("full_name", sa.String(), nullable=False),
        sa.Column("role", sa.String(), server_default="viewer", nullable=False),
        sa.Column("is_active", sa.Boolean(), server_default=sa.true(), nullable=False),
        sa.Column("created_at", _utc_datetime(), nullable=False),
        sa.Column("last_login_at", _utc_datetime(), nullable=True),
    )
    op.create_table(
        "auditlog",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("timestamp", _utc_datetime(), nullable=False),
        sa.Column("epoch", _utc_datetime(), nullable=False),
        sa.Column("user_id", sa.String(), nullable=True),
        sa.Column("username", sa.String(), nullable=False),
        sa.Column("user_role", sa.String(), nullable=False),
        sa.Column("action", sa.String(), nullable=False),
        sa.Column("target_type", sa.String(), nullable=False),
        sa.Column("target_id", sa.String(), nullable=True),
        sa.Column("ip_address", sa.String(), nullable=True),
        sa.Column("user_agent", sa.String(), nullable=True),
        sa.Column("details", sa.String(), nullable=True),
    )
    for table, column in (
        ("incident", "camera_id"),
        ("incident", "start_timestamp"),
        ("incident", "class_name"),
        ("incident", "incident_type"),
        ("incident", "severity"),
        ("incident", "status"),
        ("incident", "verification_status"),
        ("incident", "severity_score"),
        ("incident", "reviewed_label"),
        ("incident", "created_at"),
        ("incident", "is_privacy_blurred"),
        ("incident", "is_acknowledged"),
        ("incident", "ack_token"),
        ("incident", "escalation_tier"),
        ("incident", "is_demo"),
        ("webpushsubscription", "endpoint"),
        ("webpushsubscription", "user_id"),
        ("webpushsubscription", "created_at"),
        ("webpushsubscription", "is_active"),
        ("alertdelivery", "incident_id"),
        ("alertdelivery", "channel"),
        ("alertdelivery", "tier"),
        ("alertdelivery", "status"),
        ("alertdelivery", "created_at"),
        ("reviewfeedback", "incident_id"),
        ("reviewfeedback", "decision"),
        ("reviewfeedback", "is_exported"),
        ("reviewfeedback", "created_at"),
        ("camera", "name"),
        ("camera", "enabled"),
        ("camera", "created_at"),
        ("user", "username"),
        ("user", "role"),
        ("user", "is_active"),
        ("user", "created_at"),
        ("auditlog", "timestamp"),
        ("auditlog", "epoch"),
        ("auditlog", "user_id"),
        ("auditlog", "username"),
        ("auditlog", "user_role"),
        ("auditlog", "action"),
        ("auditlog", "target_type"),
        ("auditlog", "target_id"),
    ):
        op.create_index(f"ix_{table}_{column}", table, [column], unique=False)
    op.create_index("ix_incident_camera_start", "incident", ["camera_id", "start_timestamp"])
    op.create_index("ix_incident_status_created", "incident", ["status", "created_at"])
    op.create_index("ix_incident_severity_created", "incident", ["severity", "created_at"])
    op.create_index(
        "ix_alert_delivery_incident_channel_tier",
        "alertdelivery",
        ["incident_id", "channel", "tier"],
    )
    op.create_index(
        "ix_review_feedback_exported_created",
        "reviewfeedback",
        ["is_exported", "created_at"],
    )


def downgrade() -> None:
    for table in ("auditlog", "user", "camera", "reviewfeedback", "alertdelivery", "webpushsubscription", "incident"):
        op.drop_table(table)
