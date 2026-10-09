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
from backend.app.routers import config as config_router
from backend.app.routers import health as health_router
from backend.app.routers import incidents as incidents_router
from backend.app.routers import jobs as jobs_router
from backend.app.routers import stats as stats_router
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


# -----------------------------------------------------------------------------
# Application Lifespan
# -----------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Lifecycle startup and shutdown handler."""
    logger.info("Initializing ACS Backend service...")
    config.ensure_directories()
    # Initialize SQLite database and import existing CSV records
    try:
        init_db()
        imported = import_csv_to_db()
        if imported > 0:
            logger.info("Imported %d historical incidents from CSV to SQLite.", imported)
    except Exception as db_init_err:
        logger.warning("Database init or CSV migration warning: %s", db_init_err)

    # Register running event loop with job manager
    try:
        loop = asyncio.get_running_loop()
        job_manager.set_loop(loop)
    except Exception as e:
        logger.warning("Could not set event loop on job manager: %s", e)

    logger.info(
        "ACS Backend initialized [Device: %s | Model: %s]",
        config.device,
        config.model_path,
    )
    yield
    logger.info("Shutting down ACS Backend service.")


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
# CORS Middleware
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
    allow_methods=["*"],
    allow_headers=["*"],
)


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

# -----------------------------------------------------------------------------
# Router Registration
# -----------------------------------------------------------------------------
app.include_router(health_router.router)
app.include_router(config_router.router)
app.include_router(jobs_router.router)
app.include_router(incidents_router.router)
app.include_router(alerts_router.router)
app.include_router(stats_router.router)

