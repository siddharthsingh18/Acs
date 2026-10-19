"""
backend.app.core.logging_config
Structured logging with correlation request IDs and JSON/Console formatting.
"""
import contextvars
import json
import logging
import os
import sys
import time
from typing import Any, Dict, Optional

# Context variable preserving request ID across async request tasks
request_id_ctx: contextvars.ContextVar[str] = contextvars.ContextVar("request_id", default="-")


def get_request_id() -> str:
    """Returns the current request ID or '-' if outside request context."""
    return request_id_ctx.get()


def set_request_id(req_id: str) -> None:
    """Sets the current request ID for the active context."""
    request_id_ctx.set(req_id)


class RequestIDFilter(logging.Filter):
    """Injects current request_id into standard log records."""

    def filter(self, record: logging.LogRecord) -> bool:
        record.request_id = get_request_id()
        return True


class JSONStructuredFormatter(logging.Formatter):
    """Emits logs as standardized JSON objects for log aggregation (Datadog, Loki, CloudWatch)."""

    def format(self, record: logging.LogRecord) -> str:
        log_entry: Dict[str, Any] = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(record.created))
            + f".{int(record.msecs):03d}Z",
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "request_id": getattr(record, "request_id", get_request_id()),
            "module": record.module,
            "line": record.lineno,
        }

        if record.exc_info:
            log_entry["exception"] = self.formatException(record.exc_info)

        # Include custom extra attributes if passed
        for key, val in record.__dict__.items():
            if key not in (
                "args", "asctime", "created", "exc_info", "exc_text", "filename",
                "funcName", "levelname", "levelno", "lineno", "module", "msecs",
                "message", "msg", "name", "pathname", "process", "processName",
                "relativeCreated", "stack_info", "thread", "threadName", "request_id",
            ) and not key.startswith("_"):
                log_entry[key] = val

        return json.dumps(log_entry)


class ConsoleStructuredFormatter(logging.Formatter):
    """Emits human-readable console logs formatted with timestamp, level, and request ID."""

    def format(self, record: logging.LogRecord) -> str:
        req_id = getattr(record, "request_id", get_request_id())
        req_str = f" [{req_id}]" if req_id and req_id != "-" else ""
        asctime = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(record.created))
        base_msg = f"{asctime} [{record.levelname}]{req_str} {record.name}: {record.getMessage()}"
        if record.exc_info:
            base_msg += "\n" + self.formatException(record.exc_info)
        return base_msg


def setup_structured_logging(
    level: Optional[str] = None,
    json_format: Optional[bool] = None,
) -> None:
    """Configures the root and ACS loggers with structured logging and request ID filters."""
    log_level_str = (level or os.getenv("LOG_LEVEL", "INFO")).upper()
    log_level = getattr(logging, log_level_str, logging.INFO)

    use_json = json_format
    if use_json is None:
        use_json = os.getenv("STRUCTURED_LOGS", "").lower() in ("true", "1", "json") or os.getenv("LOG_FORMAT", "").lower() == "json"

    handler = logging.StreamHandler(sys.stdout)
    handler.addFilter(RequestIDFilter())

    if use_json:
        handler.setFormatter(JSONStructuredFormatter())
    else:
        handler.setFormatter(ConsoleStructuredFormatter())

    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    # Remove existing handlers to avoid duplicates
    root_logger.handlers = [handler]

    # Ensure uvicorn logs propagate or share handler
    for uvicorn_log in ("uvicorn", "uvicorn.access", "uvicorn.error"):
        u_logger = logging.getLogger(uvicorn_log)
        u_logger.handlers = [handler]
        u_logger.propagate = False
