"""
tests.test_api_health
Tests for FastAPI backend /api/health and /api/config endpoints.
"""
from fastapi.testclient import TestClient
import pytest

from backend.app.main import app


@pytest.fixture
def client():
    """Provides a TestClient instance for backend API testing."""
    return TestClient(app)


def test_api_health_endpoint(client):
    """Verifies GET /api/health returns 200 OK with expected status and timestamp."""
    response = client.get("/api/health")
    assert response.status_code == 200

    data = response.json()
    assert data["status"] == "ok"
    assert data["service"] == "acs-backend"
    assert data["version"] == "1.0.0"
    assert "timestamp" in data
    assert len(data["timestamp"]) > 0


def test_api_config_endpoint(client):
    """Verifies GET /api/config returns models, device status, classes, and defaults."""
    response = client.get("/api/config")
    assert response.status_code == 200

    data = response.json()
    # 1. Models
    assert "available_models" in data
    assert isinstance(data["available_models"], list)
    assert any("best.pt" in m for m in data["available_models"])

    # 2. Device info
    assert "device_info" in data
    assert "device" in data["device_info"]
    assert "cuda_available" in data["device_info"]
    assert "mps_available" in data["device_info"]

    # 3. Defaults
    assert "defaults" in data
    defaults = data["defaults"]
    assert "model_path" in defaults
    assert "confidence_threshold" in defaults
    assert "alerts_enabled" in defaults

    # 4. Supported languages
    assert "supported_languages" in data
    lang_codes = [lang["code"] for lang in data["supported_languages"]]
    assert "en" in lang_codes
    assert "ar" in lang_codes

    # 5. Classes with collision flags and bilingual translations
    assert "classes" in data
    assert len(data["classes"]) > 0
    accident_classes = [c for c in data["classes"] if c["is_accident"]]
    normal_classes = [c for c in data["classes"] if not c["is_accident"]]

    assert len(accident_classes) > 0
    assert len(normal_classes) > 0

    # Verify class item schema
    sample = data["classes"][0]
    assert "name" in sample
    assert "name_ar" in sample
    assert "is_accident" in sample


def test_cors_headers_present(client):
    """Verifies CORS headers are set appropriately for allowed origins."""
    headers = {
        "Origin": "http://localhost:5173",
        "Access-Control-Request-Method": "GET",
    }
    response = client.options("/api/health", headers=headers)
    assert response.headers.get("access-control-allow-origin") == "http://localhost:5173"
