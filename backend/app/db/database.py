"""SQLAlchemy engine, migration bootstrap, and request-scoped SQLModel sessions."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Generator

from alembic import command
from alembic.config import Config
from sqlalchemy import inspect
from sqlmodel import Session, create_engine

from acs.config import config


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DB_FILE = os.getenv("SQLITE_DB_PATH", os.path.join(config.reports_dir, "acs.db"))
DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL:
    os.makedirs(os.path.dirname(os.path.abspath(DB_FILE)), exist_ok=True)
    DATABASE_URL = f"sqlite:///{DB_FILE}"
elif DATABASE_URL.startswith("postgresql://"):
    DATABASE_URL = DATABASE_URL.replace("postgresql://", "postgresql+psycopg://", 1)

connect_args = {"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {}
engine = create_engine(
    DATABASE_URL,
    echo=False,
    connect_args=connect_args,
    pool_pre_ping=not DATABASE_URL.startswith("sqlite"),
)


def _alembic_config() -> Config:
    config_path = PROJECT_ROOT / "alembic.ini"
    if not config_path.is_file():
        raise RuntimeError(f"Migration configuration is missing: {config_path}")
    alembic_config = Config(str(config_path))
    alembic_config.set_main_option("script_location", str(PROJECT_ROOT / "alembic"))
    alembic_config.set_main_option("sqlalchemy.url", DATABASE_URL.replace("%", "%%"))
    return alembic_config


def _stamp_legacy_schema(alembic_config: Config) -> None:
    """Stamp complete pre-Alembic installations so the compatibility revision can run."""
    inspector = inspect(engine)
    tables = set(inspector.get_table_names())
    model_tables = {
        "incident",
        "webpushsubscription",
        "alertdelivery",
        "reviewfeedback",
        "camera",
        "user",
        "auditlog",
    }
    if "alembic_version" in tables or not (tables & model_tables):
        return
    if not model_tables.issubset(tables):
        missing = sorted(model_tables - tables)
        raise RuntimeError(
            "Legacy database has a partial schema and cannot be safely baselined. "
            f"Back it up and migrate it explicitly; missing tables: {', '.join(missing)}."
        )
    command.stamp(alembic_config, "001_initial_schema")


def init_db() -> None:
    """Upgrade the database to the current Alembic revision or fail with context."""
    import backend.app.db.models  # noqa: F401

    try:
        alembic_config = _alembic_config()
        _stamp_legacy_schema(alembic_config)
        command.upgrade(alembic_config, "head")
    except Exception as exc:
        raise RuntimeError(
            f"Database migration failed for {engine.url.render_as_string(hide_password=True)}: {exc}"
        ) from exc


def get_session() -> Generator[Session, None, None]:
    """Dependency provider yielding a database session."""
    with Session(engine) as session:
        yield session
