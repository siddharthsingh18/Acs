"""Fail-closed production configuration validation."""
from __future__ import annotations

import base64
import binascii
import os
from typing import Iterable, Mapping

from backend.app.core.security import is_placeholder_secret, verify_password


DEFAULT_CREDENTIALS = {
    "admin": ("AdminSecure2026!#",),
    "operator": ("OperatorSecure2026!#",),
    "viewer": ("ViewerSecure2026!#",),
}


def _valid_secret(value: str | None) -> bool:
    if not value or len(value.strip()) < 32 or is_placeholder_secret(value):
        return False
    return True


def _valid_fernet_key(value: str | None) -> bool:
    if not _valid_secret(value):
        return False
    try:
        return len(base64.urlsafe_b64decode(value.encode("ascii"))) == 32
    except (ValueError, UnicodeEncodeError, binascii.Error):
        return False


def _contains_default_credentials(users: Iterable[tuple[str, str]]) -> list[str]:
    insecure = []
    for username, password_hash in users:
        candidates = DEFAULT_CREDENTIALS.get(username.lower())
        if candidates and any(verify_password(candidate, password_hash) for candidate in candidates):
            insecure.append(username)
    return insecure


def validate_production_environment(
    env: Mapping[str, str] | None = None,
    *,
    database_url: str | None = None,
    users: Iterable[tuple[str, str]] = (),
) -> None:
    values = os.environ if env is None else env
    is_production = values.get("ENVIRONMENT", "").lower() in {"production", "prod"}
    if not is_production:
        return

    errors = []
    signing_key = values.get("JWT_SECRET_KEY") or values.get("SECRET_KEY")
    if not _valid_secret(signing_key):
        errors.append("Set JWT_SECRET_KEY (or SECRET_KEY) to at least 32 non-placeholder characters.")

    encryption_key = values.get("FERNET_KEY")
    if not _valid_fernet_key(encryption_key):
        errors.append("Set FERNET_KEY to a valid 32-byte URL-safe base64 key.")

    initial_admin = values.get("INITIAL_ADMIN_PASSWORD")
    if initial_admin and not _valid_secret(initial_admin):
        errors.append("INITIAL_ADMIN_PASSWORD must not be a placeholder and must be at least 32 characters.")

    cors_origins = [origin.strip() for origin in values.get("CORS_ORIGINS", "").split(",") if origin.strip()]
    if "*" in cors_origins:
        errors.append("CORS_ORIGINS must not contain '*'.")

    operator_channels = {
        channel.strip().lower()
        for channel in values.get("OPERATOR_ALERT_CHANNELS", "").split(",")
        if channel.strip()
    }
    allowed_channels = {"webpush", "telegram", "email"}
    if not operator_channels or not operator_channels.issubset(allowed_channels):
        errors.append(
            "Set OPERATOR_ALERT_CHANNELS to one or more configured operator channels "
            "(webpush, telegram, email); arbitrary channels are not allowed."
        )

    public_url = values.get("PUBLIC_BASE_URL") or values.get("APP_BASE_URL", "")
    trusted_proxies = values.get("TRUSTED_PROXY_CIDRS", "").strip()
    if not public_url.lower().startswith("https://") and not trusted_proxies:
        errors.append("Configure an https:// APP_BASE_URL/PUBLIC_BASE_URL or TRUSTED_PROXY_CIDRS.")

    worker_text = values.get("WEB_CONCURRENCY", values.get("UVICORN_WORKERS", "1"))
    try:
        worker_count = int(worker_text)
        if worker_count < 1:
            raise ValueError
    except ValueError:
        errors.append("WEB_CONCURRENCY/UVICORN_WORKERS must be a positive integer.")
        worker_count = 1

    resolved_database_url = database_url or values.get("DATABASE_URL", "sqlite:///reports/acs.db")
    if worker_count > 1 and resolved_database_url.startswith("sqlite"):
        errors.append("SQLite cannot be used with multiple workers; configure PostgreSQL.")

    insecure_accounts = _contains_default_credentials(users)
    if insecure_accounts:
        errors.append(
            "Remove legacy seeded accounts still using built-in credentials: "
            + ", ".join(sorted(insecure_accounts))
            + "."
        )

    for name in ("INITIAL_ADMIN_PASSWORD", "INITIAL_OPERATOR_PASSWORD", "INITIAL_VIEWER_PASSWORD"):
        supplied = values.get(name)
        if supplied in DEFAULT_CREDENTIALS.get(name.removeprefix("INITIAL_").removesuffix("_PASSWORD").lower(), ()):
            errors.append(f"{name} must not use a built-in development credential.")

    if errors:
        raise RuntimeError("Production startup validation failed:\n- " + "\n- ".join(errors))
