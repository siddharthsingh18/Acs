# =============================================================================
# Automated CCTV Accident Detection System (ACS)
# Production Container Image (Cloud & Public Demo Deployment)
# =============================================================================

# --- Stage 1: Build Vite React Frontend ---
FROM node:20-slim AS frontend-builder
WORKDIR /app/frontend

COPY frontend/package*.json ./
RUN npm ci

COPY frontend/ ./
RUN npm run build

# --- Stage 2: Production Python Runtime ---
FROM python:3.11-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive \
    PORT=8000 \
    ENVIRONMENT=production \
    DEMO_MODE=true \
    PUBLIC_DEMO=true \
    ALERTS_DRY_RUN=true \
    ALERTS_ENABLED=false \
    STORE_UPLOADS=false \
    CAMERA_WORKER_DEVICE=cpu \
    FRONTEND_DIST=/app/frontend/dist

WORKDIR /app

# Install minimal OS runtime packages for OpenCV and FFmpeg
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    ffmpeg \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install pinned Python dependencies
COPY requirements.txt ./
RUN python -m pip install --no-cache-dir --upgrade pip && \
    python -m pip install --no-cache-dir -r requirements.txt

# Copy backend application code, models, and configs
COPY backend/ ./backend/
COPY acs/ ./acs/
COPY ml/ ./ml/
COPY evaluate.py alerts.py config.py acc_ex.mp4 best.pt ./
COPY reports/ ./reports/

# Copy compiled frontend from builder stage
COPY --from=frontend-builder /app/frontend/dist ./frontend/dist

# Create runtime directories
RUN mkdir -p /app/reports /app/report_images /app/report_clips /app/temp /app/media

# Create non-privileged service user
RUN useradd -m -u 1000 appuser && \
    chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:8000/api/health || exit 1

CMD ["sh", "-c", "uvicorn backend.app.main:app --host 0.0.0.0 --port ${PORT} --workers 1"]
