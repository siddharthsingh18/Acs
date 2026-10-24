from pathlib import Path

from sqlalchemy import inspect
from sqlalchemy.engine import make_url
from sqlmodel import Session, select

from backend.app.db import database
from backend.app.db.models import Incident


def test_init_db_runs_alembic_and_uses_utc_timestamps(monkeypatch, tmp_path: Path):
    db_path = tmp_path / "migration.db"
    url = f"sqlite:///{db_path}"
    engine = database.create_engine(url, connect_args={"check_same_thread": False})
    monkeypatch.setattr(database, "engine", engine)
    monkeypatch.setattr(database, "DATABASE_URL", url)

    database.init_db()

    tables = set(inspect(engine).get_table_names())
    assert {"incident", "user", "alembic_version"}.issubset(tables)
    with Session(engine) as session:
        session.add(
            Incident(
                incident_id="utc-test",
                start_timestamp="2026-10-24T00:00:00Z",
                end_timestamp="2026-10-24T00:00:01Z",
                class_name="accident",
                max_confidence=0.5,
                image_path="frame.jpg",
                keyframe_url="/frame.jpg",
                created_at=1792800000.0,
            )
        )
        session.commit()
        incident = session.exec(
            select(Incident).where(Incident.incident_id == "utc-test")
        ).one()
        assert incident.start_timestamp.endswith("Z")
        assert isinstance(incident.created_at, float)


def test_postgres_url_uses_psycopg_driver():
    parsed = make_url("postgresql+psycopg://acs:pw@db/acs")
    assert parsed.get_backend_name() == "postgresql"
    assert parsed.get_driver_name() == "psycopg"
