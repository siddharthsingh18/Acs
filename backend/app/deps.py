"""
backend.app.deps
FastAPI dependency providers for configuration and system resources.
"""
from acs.config import AppConfig, config


def get_config() -> AppConfig:
    """Dependency provider returning the active global AppConfig instance."""
    return config
