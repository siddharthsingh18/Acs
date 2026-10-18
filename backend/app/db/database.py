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
    """Creates database tables if they do not already exist, migrates missing columns, and seeds default users."""
    import backend.app.db.models  # noqa: F401
    import time
    from sqlalchemy import text
    from sqlmodel import select
    from backend.app.db.models import User
    from backend.app.core.security import hash_password

    SQLModel.metadata.create_all(engine)

    # Lightweight schema migration for SQLite: add any missing columns to incident table
    try:
        with engine.connect() as conn:
            cursor = conn.execute(text("PRAGMA table_info(incident);"))
            existing_cols = {row[1] for row in cursor.fetchall()}
            if existing_cols:
                cols_to_add = {
                    "camera_id": "TEXT",
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

    # Seed default user accounts if User table is empty
    try:
        with Session(engine) as session:
            existing_user = session.exec(select(User)).first()
            if not existing_user:
                now = time.time()
                admin_pw = os.getenv("INITIAL_ADMIN_PASSWORD", "AdminSecure2026!#")
                operator_pw = os.getenv("INITIAL_OPERATOR_PASSWORD", "OperatorSecure2026!#")
                viewer_pw = os.getenv("INITIAL_VIEWER_PASSWORD", "ViewerSecure2026!#")

                users = [
                    User(
                        id="usr_admin",
                        username="admin",
                        hashed_password=hash_password(admin_pw),
                        full_name="System Administrator",
                        role="admin",
                        is_active=True,
                        created_at=now,
                    ),
                    User(
                        id="usr_operator",
                        username="operator",
                        hashed_password=hash_password(operator_pw),
                        full_name="Control Room Operator",
                        role="operator",
                        is_active=True,
                        created_at=now,
                    ),
                    User(
                        id="usr_viewer",
                        username="viewer",
                        hashed_password=hash_password(viewer_pw),
                        full_name="Safety Analyst",
                        role="viewer",
                        is_active=True,
                        created_at=now,
                    ),
                ]
                session.add_all(users)
                session.commit()
    except Exception as e:
        # Avoid crashing initialization if table locked or in test runner
        pass


def get_session() -> Generator[Session, None, None]:
    """Dependency provider yielding a database session."""
    with Session(engine) as session:
        yield session

