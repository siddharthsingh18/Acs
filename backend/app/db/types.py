"""UTC-aware database timestamp types with epoch/ISO-compatible Python values."""
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import DateTime, String
from sqlalchemy.types import TypeDecorator


def _as_utc(value: Any) -> datetime:
    if isinstance(value, datetime):
        timestamp = value
    elif isinstance(value, (int, float)):
        timestamp = datetime.fromtimestamp(value, tz=timezone.utc)
    elif isinstance(value, str):
        cleaned = value.strip()
        try:
            timestamp = datetime.fromtimestamp(float(cleaned), tz=timezone.utc)
        except ValueError:
            timestamp = None
        if timestamp is not None:
            return timestamp
        if cleaned.endswith(" UTC"):
            cleaned = cleaned[:-4] + "+00:00"
        timestamp = datetime.fromisoformat(cleaned.replace("Z", "+00:00"))
    else:
        raise TypeError(f"Unsupported timestamp value: {type(value).__name__}")
    if timestamp.tzinfo is None:
        raise ValueError("Naive datetimes are not accepted; provide a UTC-aware timestamp.")
    return timestamp.astimezone(timezone.utc)


class UTCEpoch(TypeDecorator[float]):
    """Store timezone-aware timestamps; expose epoch floats to existing API callers."""

    impl = DateTime
    cache_ok = True

    def load_dialect_impl(self, dialect):
        if dialect.name == "sqlite":
            return dialect.type_descriptor(String(40))
        return dialect.type_descriptor(DateTime(timezone=True))

    def process_bind_param(self, value: Any, dialect):
        if value is None:
            return None
        timestamp = _as_utc(value)
        if dialect.name == "sqlite":
            return timestamp.isoformat(timespec="microseconds")
        return timestamp

    def process_result_value(self, value: Any, dialect):
        if value is None:
            return None
        timestamp = _as_utc(value)
        return timestamp.timestamp()


class UTCISO(TypeDecorator[str]):
    """Store timezone-aware timestamps; expose ISO-8601 strings ending in Z."""

    impl = DateTime
    cache_ok = True

    def load_dialect_impl(self, dialect):
        if dialect.name == "sqlite":
            return dialect.type_descriptor(String(40))
        return dialect.type_descriptor(DateTime(timezone=True))

    def process_bind_param(self, value: Any, dialect):
        if value is None:
            return None
        timestamp = _as_utc(value)
        if dialect.name == "sqlite":
            return timestamp.isoformat(timespec="microseconds")
        return timestamp

    def process_result_value(self, value: Any, dialect):
        if value is None:
            return None
        return _as_utc(value).isoformat(timespec="milliseconds").replace("+00:00", "Z")
