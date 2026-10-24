"""
backend.app.main
FastAPI application entry point for the Automated CCTV Accident Detection System (ACS).

Features:
- CORS middleware configured for Vite React frontend (http://localhost:5173).
- Structured logging with clean formatting.
- Settings and directories initialized from acs.config.
- Modular router inclusion for health and configuration APIs.
- Static file serving for generated report images.
"""
import asyncio
import logging
import os
import time
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from acs.config import config
from backend.app.db.database import init_db
from backend.app.routers import alerts as alerts_router
from backend.app.routers import audit as audit_router
from backend.app.routers import auth as auth_router
from backend.app.routers import cameras as cameras_router
from backend.app.routers import config as config_router
from backend.app.routers import edge as edge_router
from backend.app.routers import health as health_router
from backend.app.routers import incidents as incidents_router
from backend.app.routers import jobs as jobs_router
from backend.app.routers import review as review_router
from backend.app.routers import stats as stats_router
from backend.app.routers import users as users_router
from backend.app.services.camera_manager import camera_manager
from backend.app.services.incident_service import import_csv_to_db
import uuid
from backend.app.core.logging_config import setup_structured_logging, set_request_id
from backend.app.core.sentry import init_sentry, capture_error
from backend.app.core.metrics import record_http_request
from backend.app.routers import drift as drift_router
from backend.app.routers import metrics as metrics_router

from backend.app.services.job_manager import job_manager
from backend.app.core.startup_validation import validate_production_environment

# -----------------------------------------------------------------------------
# Structured Logging & Error Monitoring Initialization
# -----------------------------------------------------------------------------
setup_structured_logging()
init_sentry()
logger = logging.getLogger("acs.backend")


def validate_startup_secrets() -> None:
    """Refuse unsafe production settings before any listeners or workers start."""
    from backend.app.db.database import DATABASE_URL

    validate_production_environment(database_url=DATABASE_URL)


# -----------------------------------------------------------------------------
# Application Lifespan
# -----------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Lifecycle startup and shutdown handler."""
    logger.info("Initializing ACS Backend service...")
    validate_startup_secrets()
    config.ensure_directories()
    init_db()
    from sqlmodel import Session, select
    from backend.app.db.database import engine
    from backend.app.db.models import User

    with Session(engine) as session:
        validate_production_environment(
            database_url=engine.url.render_as_string(hide_password=True),
            users=((user.username, user.hashed_password) for user in session.exec(select(User)).all()),
        )
    imported = import_csv_to_db()
    if imported > 0:
        logger.info("Imported %d historical incidents from CSV.", imported)

    # Register running event loop with job manager and camera manager
    try:
        loop = asyncio.get_running_loop()
        job_manager.set_loop(loop)
        camera_manager.set_loop(loop)
        camera_manager.initialize_cameras()
    except Exception as e:
        logger.warning("Could not initialize camera manager or loop: %s", e)

    logger.info(
        "ACS Backend initialized [Device: %s | Model: %s]",
        config.device,
        config.model_path,
    )
    # Start background alert escalation monitor
    from backend.app.services.alert_escalation_service import run_escalation_worker
    escalation_task = asyncio.create_task(run_escalation_worker(interval_seconds=30.0))

    # Start background automated retention policy worker
    from backend.app.services.retention_service import run_retention_worker
    retention_interval = float(os.getenv("RETENTION_CHECK_INTERVAL_SECONDS", "3600"))
    retention_task = asyncio.create_task(run_retention_worker(interval_seconds=retention_interval))

    yield
    logger.info("Shutting down ACS Backend service and background workers.")
    escalation_task.cancel()
    retention_task.cancel()
    for task in (escalation_task, retention_task):
        try:
            await task
        except asyncio.CancelledError:
            pass
    camera_manager.shutdown()


# -----------------------------------------------------------------------------
# FastAPI Application Factory
# -----------------------------------------------------------------------------
app = FastAPI(
    title="ACS - CCTV Accident Detection API",
    description="Enterprise Automated Surveillance, Tracking & Emergency Dispatch API",
    version="1.0.0",
    lifespan=lifespan,
)

# -----------------------------------------------------------------------------
# Restrictive CORS Middleware
# -----------------------------------------------------------------------------
is_prod = os.getenv("ENVIRONMENT", "").lower() in ("production", "prod")
extra_origins = os.getenv("CORS_ORIGINS")

if is_prod and extra_origins:
    origins = [o.strip() for o in extra_origins.split(",") if o.strip()]
else:
    origins = [
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ]
    if extra_origins:
        for o in extra_origins.split(","):
            stripped = o.strip()
            if stripped and stripped not in origins:
                origins.append(stripped)

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS"],
    allow_headers=[
        "Authorization",
        "Content-Type",
        "Accept",
        "Range",
        "Origin",
        "X-Requested-With",
    ],
)


# -----------------------------------------------------------------------------
# Security Headers Middleware
# -----------------------------------------------------------------------------
@app.middleware("http")
async def security_headers_middleware(request: Request, call_next):
    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["X-XSS-Protection"] = "1; mode=block"
    response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
    response.headers["Permissions-Policy"] = "camera=(), microphone=(), geolocation=()"
    response.headers["X-Permitted-Cross-Domain-Policies"] = "none"
    response.headers["Content-Security-Policy"] = (
        "default-src 'self'; "
        "img-src 'self' data: blob: https://*.tile.openstreetmap.org https://unpkg.com; "
        "style-src 'self' 'unsafe-inline' https://unpkg.com; "
        "script-src 'self' 'unsafe-inline' https://unpkg.com; "
        "connect-src 'self' ws: wss:; "
        "media-src 'self' blob:;"
    )
    is_https = request.url.scheme == "https" or request.headers.get("X-Forwarded-Proto") == "https"
    if is_https or os.getenv("ENVIRONMENT") == "production":
        response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains; preload"
    return response


# -----------------------------------------------------------------------------
# Request Timing & Structured Logging Middleware
# -----------------------------------------------------------------------------
# Request Timing & Structured Logging Middleware
# -----------------------------------------------------------------------------
@app.middleware("http")
async def log_requests_middleware(request: Request, call_next):
    # Correlation Request ID
    req_id = request.headers.get("X-Request-ID") or request.headers.get("X-Correlation-ID")
    if not req_id:
        req_id = f"req_{uuid.uuid4().hex[:12]}"
    set_request_id(req_id)

    start_time = time.time()
    try:
        response = await call_next(request)
    except Exception as exc:
        duration_ms = (time.time() - start_time) * 1000.0
        capture_error(exc, request_id=req_id, context={"path": request.url.path, "method": request.method})
        record_http_request(request.method, request.url.path, 500, duration_ms / 1000.0)
        from fastapi.responses import JSONResponse
        return JSONResponse(
            status_code=500,
            content={"detail": "Internal server error", "request_id": req_id},
            headers={"X-Request-ID": req_id},
        )

    duration_ms = (time.time() - start_time) * 1000.0
    response.headers["X-Request-ID"] = req_id

    # Record Prometheus HTTP metric
    record_http_request(request.method, request.url.path, response.status_code, duration_ms / 1000.0)

    # Log request unless high-frequency metric probe
    if request.url.path != "/metrics" or response.status_code >= 400:
        logger.info(
            "%s %s - %d (%.1fms)",
            request.method,
            request.url.path,
            response.status_code,
            duration_ms,
            extra={"request_id": req_id, "status_code": response.status_code, "duration_ms": duration_ms},
        )
    return response


# -----------------------------------------------------------------------------
# Mount Static Directories for Images and Keyframes
# -----------------------------------------------------------------------------
config.ensure_directories()
if os.path.exists(config.report_images_dir):
    app.mount(
        "/report_images",
        StaticFiles(directory=config.report_images_dir),
        name="report_images",
    )
    app.mount(
        "/media/keyframes",
        StaticFiles(directory=config.report_images_dir),
        name="media_keyframes",
    )


@app.get("/media/clips/{filename}")
@app.head("/media/clips/{filename}")
def stream_media_clip(filename: str, request: Request):
    """Streams video clips with full HTTP Range request support."""
    from backend.app.routers.incidents import serve_file_with_range_support
    file_path = os.path.join(config.clips_dir, filename)
    return serve_file_with_range_support(file_path, request)

# -----------------------------------------------------------------------------
# Router Registration
# -----------------------------------------------------------------------------
app.include_router(metrics_router.router)
app.include_router(drift_router.router)
app.include_router(health_router.router)
app.include_router(auth_router.router)
app.include_router(users_router.router)
app.include_router(audit_router.router)
app.include_router(config_router.router)
app.include_router(jobs_router.router)
app.include_router(incidents_router.router)
app.include_router(alerts_router.router)
app.include_router(stats_router.router)
app.include_router(cameras_router.router)
app.include_router(review_router.router)
app.include_router(edge_router.router)


# -----------------------------------------------------------------------------
# Mount Frontend Static Build (Production / Docker)
# -----------------------------------------------------------------------------
frontend_dist = os.getenv(
    "FRONTEND_DIST",
    os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "frontend", "dist"),
)
if os.path.exists(frontend_dist):
    from fastapi import HTTPException
    from starlette.responses import FileResponse

    assets_dir = os.path.join(frontend_dist, "assets")
    if os.path.exists(assets_dir):
        app.mount("/assets", StaticFiles(directory=assets_dir), name="static_assets")

    @app.get("/{full_path:path}", include_in_schema=False)
    async def serve_spa(full_path: str):
        if full_path.startswith("api/") or full_path == "api":
            raise HTTPException(status_code=404, detail="API route not found")
        file_path = os.path.join(frontend_dist, full_path)
        if full_path and os.path.isfile(file_path):
            return FileResponse(file_path)
        index_file = os.path.join(frontend_dist, "index.html")
        if os.path.exists(index_file):
            return FileResponse(index_file)
        raise HTTPException(status_code=404, detail="Frontend build not found")

