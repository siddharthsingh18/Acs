"""
backend.app.core.sentry
Error monitoring integration using Sentry SDK.
Safely falls back to local logging when SENTRY_DSN is not configured.
"""
import logging
import os
from typing import Any, Dict, Optional

logger = logging.getLogger("acs.backend.sentry")

_SENTRY_INITIALIZED = False


def _scrub_sensitive_data(event: Dict[str, Any], hint: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Scrubs secrets, tokens, and credentials from Sentry error payloads before transmission."""
    try:
        # Scrub request headers
        request = event.get("request", {})
        headers = request.get("headers", {})
        for secret_header in ("authorization", "cookie", "x-auth-token"):
            if secret_header in headers:
                headers[secret_header] = "[FILTERED]"

        # Scrub query params and sensitive fields
        data = request.get("data")
        if isinstance(data, dict):
            for secret_key in ("password", "token", "secret", "credentials", "jwt"):
                if secret_key in data:
                    data[secret_key] = "[FILTERED]"
    except Exception:
        pass
    return event


def init_sentry() -> bool:
    """Initializes Sentry SDK if SENTRY_DSN environment variable is present."""
    global _SENTRY_INITIALIZED
    dsn = os.getenv("SENTRY_DSN", "").strip()
    if not dsn:
        logger.debug("Sentry DSN not configured; running with local structured error logging.")
        return False

    try:
        import sentry_sdk
        from sentry_sdk.integrations.fastapi import FastApiIntegration
        from sentry_sdk.integrations.logging import LoggingIntegration

        environment = os.getenv("ENVIRONMENT", "production")
        release = os.getenv("APP_VERSION", "1.0.0")
        traces_sample_rate = float(os.getenv("SENTRY_TRACES_SAMPLE_RATE", "0.2"))

        logging_integration = LoggingIntegration(
            level=logging.INFO,
            event_level=logging.ERROR,
        )

        sentry_sdk.init(
            dsn=dsn,
            environment=environment,
            release=release,
            traces_sample_rate=traces_sample_rate,
            integrations=[
                FastApiIntegration(transaction_style="endpoint"),
                logging_integration,
            ],
            before_send=_scrub_sensitive_data,
            send_default_pii=False,
        )
        _SENTRY_INITIALIZED = True
        logger.info("Sentry error monitoring initialized successfully [env: %s, release: %s].", environment, release)
        return True
    except ImportError:
        logger.warning("sentry-sdk package not installed; error monitoring disabled.")
        return False
    except Exception as exc:
        logger.warning("Failed to initialize Sentry error monitoring: %s", exc)
        return False


def capture_error(exc: BaseException, context: Optional[Dict[str, Any]] = None, request_id: Optional[str] = None) -> None:
    """Explicitly captures an exception and reports to Sentry with context tags."""
    if _SENTRY_INITIALIZED:
        try:
            import sentry_sdk
            with sentry_sdk.push_scope() as scope:
                if request_id:
                    scope.set_tag("request_id", request_id)
                if context:
                    for k, v in context.items():
                        scope.set_extra(k, v)
                sentry_sdk.capture_exception(exc)
        except Exception:
            pass
    logger.error("Exception captured: %s", exc, exc_info=True, extra={"request_id": request_id, **(context or {})})
