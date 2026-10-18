"""
backend.app.db.database
SQLite engine configuration, session management, and table initialization.
"""
import os
from typing import Generator
from sqlmodel import Session, SQLModel, create_engine

from acs.config import config

DB_FILE = os.getenv("SQLITE_DB_PATH", os.path.join(config.reports_dir, "acs.db"))
os.makedirs(os.path.dirname(os.path.abspath(DB_FILE)), exist_ok=True)

sqlite_url = f"sqlite:///{DB_FILE}"
connect_args = {"check_same_thread": False}
engine = create_engine(sqlite_url, echo=False, connect_args=connect_args)


def init_db() -> None:
    """Creates database tables if they do not already exist and migrates missing columns."""
    import backend.app.db.models  # noqa: F401
    from sqlalchemy import text

    SQLModel.metadata.create_all(engine)

    # Lightweight schema migration for SQLite: add any missing columns to incident table
    try:
        with engine.connect() as conn:
            cursor = conn.execute(text("PRAGMA table_info(incident);"))
            existing_cols = {row[1] for row in cursor.fetchall()}
            if existing_cols:
                cols_to_add = {
                    "verification_status": "TEXT",
                    "vlm_confidence": "REAL",
                    "vlm_reason": "TEXT",
                    "physics_passed": "BOOLEAN",
                    "clip_path": "TEXT",
                    "clip_annotated_path": "TEXT",
                    "impact_offset_seconds": "REAL",
                    "severity_score": "INTEGER",
                    "severity_breakdown": "TEXT",
                    "recommended_units": "TEXT",
                    "reviewed_label": "TEXT",
                    "corrected_class": "TEXT",
                    "corrected_boxes": "TEXT",
                }
                for col_name, col_type in cols_to_add.items():
                    if col_name not in existing_cols:
                        conn.execute(text(f"ALTER TABLE incident ADD COLUMN {col_name} {col_type};"))
                conn.commit()
    except Exception:
        pass


def get_session() -> Generator[Session, None, None]:
    """Dependency provider yielding a database session."""
    with Session(engine) as session:
        yield session
