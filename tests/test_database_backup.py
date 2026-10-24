import sqlite3
from pathlib import Path

import pytest

from backend.app.db.backup import backup_sqlite, restore_sqlite


def test_sqlite_backup_and_restore_round_trip(tmp_path: Path):
    source = tmp_path / "source.db"
    backup = tmp_path / "backup.db"
    restored = tmp_path / "restored.db"
    with sqlite3.connect(source) as connection:
        connection.execute("CREATE TABLE sample (id INTEGER PRIMARY KEY, value TEXT NOT NULL)")
        connection.execute("INSERT INTO sample(value) VALUES ('restored')")

    backup_sqlite(source, backup)
    restore_sqlite(backup, restored)

    with sqlite3.connect(restored) as connection:
        assert connection.execute("SELECT value FROM sample").fetchone() == ("restored",)


def test_sqlite_restore_requires_explicit_overwrite(tmp_path: Path):
    backup = tmp_path / "backup.db"
    target = tmp_path / "target.db"
    sqlite3.connect(backup).close()
    sqlite3.connect(target).close()
    with pytest.raises(FileExistsError, match="overwrite=True"):
        restore_sqlite(backup, target)
