from __future__ import annotations

import base64

import pytest

from backend.app.core.startup_validation import validate_production_environment
from backend.app.core.security import hash_password


def valid_production_env() -> dict[str, str]:
    return {
        "ENVIRONMENT": "production",
        "JWT_SECRET_KEY": "a" * 64,
        "FERNET_KEY": base64.urlsafe_b64encode(b"f" * 32).decode("ascii"),
        "CORS_ORIGINS": "https://operator.example.test",
        "APP_BASE_URL": "https://operator.example.test",
        "OPERATOR_ALERT_CHANNELS": "webpush",
        "WEB_CONCURRENCY": "1",
        "DATABASE_URL": "postgresql+psycopg://acs:secret@db/acs",
    }


def test_valid_production_configuration_passes():
    validate_production_environment(valid_production_env())


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("JWT_SECRET_KEY", ""),
        ("JWT_SECRET_KEY", "change-me"),
        ("JWT_SECRET_KEY", "short"),
        ("FERNET_KEY", ""),
        ("FERNET_KEY", "not-a-fernet-key-that-is-long-enough"),
    ],
)
def test_missing_placeholder_or_short_secrets_are_refused(name: str, value: str):
    env = valid_production_env()
    env[name] = value
    with pytest.raises(RuntimeError, match="Production startup validation failed"):
        validate_production_environment(env)


def test_wildcard_cors_is_refused():
    env = valid_production_env()
    env["CORS_ORIGINS"] = "https://operator.example.test,*"
    with pytest.raises(RuntimeError, match="CORS_ORIGINS"):
        validate_production_environment(env)


def test_missing_https_and_trusted_proxy_settings_are_refused():
    env = valid_production_env()
    env.pop("APP_BASE_URL")
    with pytest.raises(RuntimeError, match="TRUSTED_PROXY_CIDRS"):
        validate_production_environment(env)


def test_sqlite_with_multiple_workers_is_refused():
    env = valid_production_env()
    env["DATABASE_URL"] = "sqlite:///reports/acs.db"
    env["WEB_CONCURRENCY"] = "2"
    with pytest.raises(RuntimeError, match="SQLite cannot be used"):
        validate_production_environment(env)


def test_seeded_default_admin_credentials_are_refused():
    env = valid_production_env()
    users = [("admin", hash_password("AdminSecure2026!#"))]
    with pytest.raises(RuntimeError, match="legacy seeded accounts"):
        validate_production_environment(env, users=users)


def test_builtin_initial_admin_password_is_refused():
    env = valid_production_env()
    env["INITIAL_ADMIN_PASSWORD"] = "AdminSecure2026!#"
    with pytest.raises(RuntimeError, match="INITIAL_ADMIN_PASSWORD"):
        validate_production_environment(env)
