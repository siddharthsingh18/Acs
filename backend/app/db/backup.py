"""Explicit SQLite/PostgreSQL backup and restore utilities."""
from __future__ import annotations

import argparse
import os
import shutil
import sqlite3
import subprocess
from pathlib import Path

from sqlalchemy.engine import make_url


def _postgres_environment(url) -> dict[str, str]:
    env = os.environ.copy()
    if url.password:
        env["PGPASSWORD"] = url.password
    query = dict(url.query)
    if query.get("sslmode"):
        env["PGSSLMODE"] = str(query["sslmode"])
    return env


def backup_sqlite(source: Path, destination: Path) -> None:
    if not source.is_file():
        raise FileNotFoundError(f"SQLite database does not exist: {source}")
    if destination.exists():
        raise FileExistsError(f"Backup destination already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(f"file:{source.resolve()}?mode=ro", uri=True) as source_db:
        with sqlite3.connect(destination) as backup_db:
            source_db.backup(backup_db)


def restore_sqlite(backup: Path, destination: Path, *, overwrite: bool = False) -> None:
    if not backup.is_file():
        raise FileNotFoundError(f"SQLite backup does not exist: {backup}")
    if destination.exists() and not overwrite:
        raise FileExistsError(f"Restore target exists; pass overwrite=True explicitly: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(f"file:{backup.resolve()}?mode=ro", uri=True) as backup_db:
        with sqlite3.connect(destination) as destination_db:
            backup_db.backup(destination_db)


def backup_postgres(database_url: str, destination: Path) -> None:
    if destination.exists():
        raise FileExistsError(f"Backup destination already exists: {destination}")
    url = make_url(database_url)
    if url.get_backend_name() != "postgresql":
        raise ValueError("PostgreSQL backup requires a PostgreSQL URL.")
    executable = shutil.which("pg_dump")
    if not executable:
        raise FileNotFoundError("pg_dump is required for PostgreSQL backups.")
    destination.parent.mkdir(parents=True, exist_ok=True)
    command = [
        executable,
        "--format=custom",
        f"--file={destination}",
        f"--host={url.host or 'localhost'}",
        f"--port={url.port or 5432}",
        f"--username={url.username or ''}",
        f"--dbname={url.database or ''}",
    ]
    subprocess.run(command, env=_postgres_environment(url), check=True)


def restore_postgres(database_url: str, backup: Path, *, overwrite: bool = False) -> None:
    if not backup.is_file():
        raise FileNotFoundError(f"PostgreSQL backup does not exist: {backup}")
    if not overwrite:
        raise ValueError("PostgreSQL restore requires overwrite=True because existing objects are replaced.")
    url = make_url(database_url)
    if url.get_backend_name() != "postgresql":
        raise ValueError("PostgreSQL restore requires a PostgreSQL URL.")
    executable = shutil.which("pg_restore")
    if not executable:
        raise FileNotFoundError("pg_restore is required for PostgreSQL restore.")
    command = [
        executable,
        "--clean",
        "--if-exists",
        "--no-owner",
        f"--host={url.host or 'localhost'}",
        f"--port={url.port or 5432}",
        f"--username={url.username or ''}",
        f"--dbname={url.database or ''}",
        str(backup),
    ]
    subprocess.run(command, env=_postgres_environment(url), check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("backup", "restore"))
    parser.add_argument("source", type=Path, help="SQLite path or PostgreSQL custom backup")
    parser.add_argument("destination", type=Path)
    parser.add_argument("--database-url", help="PostgreSQL URL, required for Postgres operations")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.database_url:
        if args.action == "backup":
            backup_postgres(args.database_url, args.destination)
        else:
            restore_postgres(args.database_url, args.source, overwrite=args.overwrite)
    elif args.action == "backup":
        backup_sqlite(args.source, args.destination)
    else:
        restore_sqlite(args.source, args.destination, overwrite=args.overwrite)


if __name__ == "__main__":
    main()
