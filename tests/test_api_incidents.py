"""
tests.test_api_incidents
Comprehensive test suite for incidents, dispatch reports, PDF export, alerts, and stats:
- SQLite persistence and one-time CSV migration
- GET /api/incidents (pagination, filters by class/severity/status, sorting)
- GET /api/incidents/{id}
- POST /api/incidents/{id}/dispatch-report (structured JSON, caching in DB)
- GET /api/incidents/{id}/pdf (valid application/pdf binary)
- GET /api/incidents/export.csv (valid text/csv)
- POST /api/incidents/{id}/status (human review: approved/rejected/dispatched)
- Alerts: GET/PUT /api/alerts/settings, POST /api/alerts/test
- Analytics: GET /api/stats (counts today, by class, average confidence, review time)
"""
import pytest
from fastapi.testclient import TestClient

from backend.app.main import app
from backend.app.db.database import init_db
from backend.app.services.incident_service import save_incident_to_db, import_csv_to_db


@pytest.fixture
def client():
    """Provides FastAPI TestClient."""
    init_db()
    import_csv_to_db()
    return TestClient(app)


@pytest.fixture
def sample_incident():
    """Ensures at least one known incident exists in SQLite for tests."""
    init_db()
    return save_incident_to_db(
        incident_id="TEST-INC-101",
        start_timestamp="2026-10-10 12:00:00 UTC",
        end_timestamp="2026-10-10 12:00:05 UTC",
        class_name="car_car_accident",
        max_confidence=0.885,
        image_path="report_images/INC-3-001_keyframe.jpg",
        duration=5.0,
    )


def test_list_incidents_pagination_and_filter(client, sample_incident):
    """Verifies GET /api/incidents supports pagination and filtering."""
    # List all
    res = client.get("/api/incidents?page=1&page_size=10")
    assert res.status_code == 200
    data = res.json()
    assert "items" in data
    assert "total" in data
    assert data["total"] >= 1
    assert data["page"] == 1
    assert data["page_size"] == 10

    # Filter by class matching
    res_cls = client.get("/api/incidents?class=car_car_accident")
    assert res_cls.status_code == 200
    for item in res_cls.json()["items"]:
        assert item["class_name"] == "car_car_accident"

    # Filter by class with no matches
    res_empty = client.get("/api/incidents?class=non_existent_class")
    assert res_empty.status_code == 200
    assert res_empty.json()["total"] == 0
    assert len(res_empty.json()["items"]) == 0


def test_get_incident_by_id(client, sample_incident):
    """Verifies GET /api/incidents/{id} returns details or 404 for unknown ID."""
    res = client.get(f"/api/incidents/{sample_incident.incident_id}")
    assert res.status_code == 200
    data = res.json()
    assert data["incident_id"] == sample_incident.incident_id
    assert data["class_name"] == "car_car_accident"
    assert data["status"] in ("pending", "approved", "rejected", "dispatched")

    # Non-existent ID -> 404
    res_404 = client.get("/api/incidents/UNKNOWN-ID-999")
    assert res_404.status_code == 404


def test_update_incident_status_human_review(client, sample_incident):
    """Verifies POST /api/incidents/{id}/status updates status and reviewer timestamp."""
    # 1. Approve
    res_app = client.post(
        f"/api/incidents/{sample_incident.incident_id}/status",
        json={"status": "approved", "reviewer_name": "Supervisor Sarah"},
    )
    assert res_app.status_code == 200
    data_app = res_app.json()
    assert data_app["status"] == "approved"
    assert data_app["reviewer_name"] == "Supervisor Sarah"
    assert data_app["reviewed_at"] is not None

    # 2. Reject
    res_rej = client.post(
        f"/api/incidents/{sample_incident.incident_id}/status",
        json={"status": "rejected", "reviewer_name": "Officer Dave"},
    )
    assert res_rej.status_code == 200
    assert res_rej.json()["status"] == "rejected"
    assert res_rej.json()["reviewer_name"] == "Officer Dave"

    # 3. Invalid status -> 422 Unprocessable Entity
    res_bad = client.post(
        f"/api/incidents/{sample_incident.incident_id}/status",
        json={"status": "invalid_status"},
    )
    assert res_bad.status_code == 422


def test_dispatch_report_caching(client, sample_incident):
    """Verifies generating and caching multimodal dispatch report."""
    # First call generates report and writes cache to SQLite
    res1 = client.post(f"/api/incidents/{sample_incident.incident_id}/dispatch-report?lang=en")
    assert res1.status_code == 200
    report1 = res1.json()
    assert "vehicles" in report1
    assert "severity" in report1
    assert "recommended_units" in report1

    # Second call fetches directly from cached SQLite column
    res2 = client.post(f"/api/incidents/{sample_incident.incident_id}/dispatch-report?lang=en")
    assert res2.status_code == 200
    report2 = res2.json()
    assert report1 == report2


def test_incident_pdf_download(client, sample_incident):
    """Verifies GET /api/incidents/{id}/pdf generates valid PDF bytes."""
    res = client.get(f"/api/incidents/{sample_incident.incident_id}/pdf")
    assert res.status_code == 200
    assert "application/pdf" in res.headers.get("content-type", "")
    assert res.content.startswith(b"%PDF")
    assert len(res.content) > 1000


def test_incidents_export_csv(client, sample_incident):
    """Verifies GET /api/incidents/export.csv returns CSV formatted data."""
    res = client.get("/api/incidents/export.csv")
    assert res.status_code == 200
    assert "text/csv" in res.headers.get("content-type", "")
    lines = res.text.strip().splitlines()
    assert len(lines) >= 2  # Header + at least 1 record
    header = lines[0]
    assert "incident_id" in header
    assert "class" in header
    assert "status" in header


def test_alert_settings_and_test(client):
    """Verifies GET/PUT /api/alerts/settings and POST /api/alerts/test."""
    # 1. Get settings
    res_get = client.get("/api/alerts/settings")
    assert res_get.status_code == 200
    assert "enabled" in res_get.json()
    assert "channels" in res_get.json()

    # 2. Update settings
    res_put = client.put(
        "/api/alerts/settings",
        json={"enabled": True, "dry_run": True},
    )
    assert res_put.status_code == 200
    assert res_put.json()["enabled"] is True
    assert res_put.json()["dry_run"] is True

    # 3. Test alert trigger
    res_test = client.post(
        "/api/alerts/test",
        json={"dry_run": True, "lang": "en"},
    )
    assert res_test.status_code == 200
    assert "results" in res_test.json()
    assert res_test.json()["results"].get("console") is True


def test_incident_stats(client, sample_incident):
    """Verifies GET /api/stats computes summary KPIs."""
    # Ensure at least one reviewed incident for review time metric
    client.post(
        f"/api/incidents/{sample_incident.incident_id}/status",
        json={"status": "approved", "reviewer_name": "Chief Anderson"},
    )

    res = client.get("/api/stats")
    assert res.status_code == 200
    data = res.json()
    assert "total_incidents" in data
    assert data["total_incidents"] >= 1
    assert "by_class" in data
    assert "by_status" in data
    assert "average_confidence" in data
    assert data["average_confidence"] > 0.0
    assert "average_review_time_seconds" in data
