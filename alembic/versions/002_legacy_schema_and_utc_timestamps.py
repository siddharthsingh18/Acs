"""Fill legacy columns and normalize timestamp storage.

Revision ID: 002_legacy_utc
Revises: 001_initial_schema
"""
from alembic import op
import sqlalchemy as sa


revision = "002_legacy_utc"
down_revision = "001_initial_schema"
branch_labels = None
depends_on = None


def _add_legacy_incident_columns() -> None:
    inspector = sa.inspect(op.get_bind())
    columns = {column["name"] for column in inspector.get_columns("incident")}
    additions = {
        "camera_id": sa.String(),
        "incident_type": sa.String(),
        "verification_status": sa.String(),
        "vlm_confidence": sa.Float(),
        "vlm_reason": sa.String(),
        "physics_passed": sa.Boolean(),
        "clip_path": sa.String(),
        "clip_annotated_path": sa.String(),
        "impact_offset_seconds": sa.Float(),
        "severity_score": sa.Integer(),
        "severity_breakdown": sa.String(),
        "recommended_units": sa.String(),
        "reviewed_label": sa.String(),
        "corrected_class": sa.String(),
        "corrected_boxes": sa.String(),
        "is_privacy_blurred": sa.Boolean(),
        "unredacted_keyframe_path": sa.String(),
        "unredacted_clip_path": sa.String(),
        "is_acknowledged": sa.Boolean(),
        "acknowledged_at": sa.String(40),
        "acknowledged_by": sa.String(),
        "ack_token": sa.String(),
        "escalation_tier": sa.Integer(),
        "last_escalated_at": sa.String(40),
        "is_demo": sa.Boolean(),
    }
    for name, column_type in additions.items():
        if name not in columns:
            op.add_column("incident", sa.Column(name, column_type, nullable=True))
    inspector = sa.inspect(op.get_bind())
    existing_indexes = {index["name"] for index in inspector.get_indexes("incident")}
    for index_name, column_names in (
        ("ix_incident_camera_id", ("camera_id",)),
        ("ix_incident_camera_start", ("camera_id", "start_timestamp")),
        ("ix_incident_status_created", ("status", "created_at")),
        ("ix_incident_severity_created", ("severity", "created_at")),
    ):
        if index_name not in existing_indexes:
            op.create_index(index_name, "incident", list(column_names), unique=False)


def _normalize_sqlite_timestamp_storage() -> None:
    bind = op.get_bind()
    if bind.dialect.name != "sqlite":
        return
    fields = {
        "incident": ("start_timestamp", "end_timestamp", "reviewed_at", "created_at", "acknowledged_at", "last_escalated_at"),
        "webpushsubscription": ("created_at",),
        "alertdelivery": ("created_at", "acknowledged_at"),
        "reviewfeedback": ("reviewed_at", "exported_at", "created_at"),
        "camera": ("created_at", "updated_at"),
        "user": ("created_at", "last_login_at"),
        "auditlog": ("timestamp", "epoch"),
    }
    for table, columns in fields.items():
        existing_columns = {
            column["name"]: column["type"] for column in sa.inspect(bind).get_columns(table)
        }
        convertible = [name for name in columns if name in existing_columns]
        if not convertible:
            continue
        with op.batch_alter_table(table) as batch:
            for name in convertible:
                batch.alter_column(
                    name,
                    existing_type=existing_columns[name],
                    type_=sa.String(40),
                )
        for name in convertible:
            op.execute(
                sa.text(
                    f"UPDATE {table} SET {name} = "
                    f"strftime('%Y-%m-%dT%H:%M:%fZ', CAST({name} AS REAL), 'unixepoch') "
                    f"WHERE {name} IS NOT NULL AND CAST({name} AS REAL) > 100000000 "
                    f"AND {name} NOT LIKE '%T%'"
                )
            )


def upgrade() -> None:
    _add_legacy_incident_columns()
    _normalize_sqlite_timestamp_storage()


def downgrade() -> None:
    raise RuntimeError("Legacy timestamp data cannot be safely downgraded to naive/epoch storage.")
