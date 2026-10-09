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
    """Creates database tables if they do not already exist."""
    import backend.app.db.models  # noqa: F401

    SQLModel.metadata.create_all(engine)


def get_session() -> Generator[Session, None, None]:
    """Dependency provider yielding a database session."""
    with Session(engine) as session:
        yield session
