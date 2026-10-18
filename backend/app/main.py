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
import sys
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
from backend.app.routers import health as health_router
from backend.app.routers import incidents as incidents_router
from backend.app.routers import jobs as jobs_router
from backend.app.routers import review as review_router
from backend.app.routers import stats as stats_router
from backend.app.routers import users as users_router
from backend.app.services.camera_manager import camera_manager
from backend.app.services.incident_service import import_csv_to_db
from backend.app.services.job_manager import job_manager

# -----------------------------------------------------------------------------
# Structured Logging Configuration
# -----------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("acs.backend")


def validate_startup_secrets() -> None:
    """Validates that sensitive secrets are sourced exclusively from environment and not insecure placeholders."""
    is_prod = (
        os.getenv("ENVIRONMENT", "").lower() in ("production", "prod")
        or os.getenv("STRICT_SECRETS", "").lower() in ("true", "1")
    )
    from backend.app.core.security import is_placeholder_secret
    jwt_key = os.getenv("JWT_SECRET_KEY")
    fernet_key = os.getenv("FERNET_KEY")

    errors = []
    if is_prod:
        if not jwt_key or is_placeholder_secret(jwt_key):
            errors.append(
                "JWT_SECRET_KEY is missing or contains an insecure placeholder. "
                "Generate one using: python -c 'import secrets; print(secrets.token_hex(32))'"
            )
        if fernet_key and is_placeholder_secret(fernet_key):
            errors.append("FERNET_KEY matches an insecure placeholder pattern.")

    if errors:
        error_msg = "FATAL: Startup secret validation failed:\n  " + "\n  ".join(errors)
        logger.error(error_msg)
        raise RuntimeError(error_msg)


# -----------------------------------------------------------------------------
# Application Lifespan
# -----------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Lifecycle startup and shutdown handler."""
    logger.info("Initializing ACS Backend service...")
    validate_startup_secrets()
    config.ensure_directories()
    # Initialize SQLite database and import existing CSV records
    try:
        init_db()
        imported = import_csv_to_db()
        if imported > 0:
            logger.info("Imported %d historical incidents from CSV to SQLite.", imported)
    except Exception as db_init_err:
        logger.warning("Database init or CSV migration warning: %s", db_init_err)

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
    yield
    logger.info("Shutting down ACS Backend service and camera workers.")
    camera_manager.shutdown()


# Initial module load initialization for storage and database
config.ensure_directories()
try:
    init_db()
    import_csv_to_db()
except Exception as e:
    logger.warning("Initial database bootstrap warning: %s", e)

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
origins = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "http://localhost:3000",
    "http://127.0.0.1:3000",
]

# Allow custom CORS origins via environment variable
extra_origins = os.getenv("CORS_ORIGINS")
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
@app.middleware("http")
async def log_requests_middleware(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    duration_ms = (time.time() - start_time) * 1000.0
    logger.info(
        "%s %s - %d (%.1fms)",
        request.method,
        request.url.path,
        response.status_code,
        duration_ms,
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


