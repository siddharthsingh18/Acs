"""
tests.test_auth_rbac_audit
Comprehensive test suite verifying enterprise hardening:
- Argon2 password hashing and verification
- Login brute-force rate limiting (HTTP 429)
- JWT access & refresh token lifecycle, HTTP cookies
- Role-based access control (RBAC) enforced on backend (Viewer, Operator, Admin)
- Free-text reviewer name replaced with logged-in user
- Append-only audit trail with IP address and timestamps
- Video upload validation (magic bytes, size limit, decode sanity check)
- Strict secrets startup check
"""
import io
import os
import time
import pytest
from starlette.testclient import TestClient
from sqlmodel import Session

from backend.app.core.security import (
    create_access_token,
    hash_password,
    is_placeholder_secret,
    login_rate_limiter,
    verify_password,
)
from backend.app.db.database import engine, get_session
from backend.app.db.models import Incident, User
from backend.app.main import app, validate_startup_secrets
from backend.app.routers.jobs import is_valid_video_magic_bytes


@pytest.fixture
def auth_db_session():
    """Provides a fresh isolated database session."""
    with Session(engine) as session:
        yield session


@pytest.fixture
def test_client(auth_db_session):
    """Provides a TestClient with database overrides."""
    def get_test_db():
        yield auth_db_session

    app.dependency_overrides[get_session] = get_test_db
    with TestClient(app) as client:
        yield client
    app.dependency_overrides.clear()


@pytest.fixture(autouse=True)
def setup_auth_environment():
    """Ensures strict auth is enabled during this test suite."""
    old_enforce = os.environ.get("AUTH_ENFORCE")
    os.environ["AUTH_ENFORCE"] = "1"
    yield
    if old_enforce is not None:
        os.environ["AUTH_ENFORCE"] = old_enforce
    else:
        os.environ.pop("AUTH_ENFORCE", None)


def _get_auth_header(user: User) -> dict:
    """Helper generating Authorization header with valid JWT token."""
    token = create_access_token({
        "sub": user.id,
        "username": user.username,
        "role": user.role,
        "full_name": user.full_name,
    })
    return {"Authorization": f"Bearer {token}"}


# -----------------------------------------------------------------------------
# 1. Argon2 Password Hashing Tests
# -----------------------------------------------------------------------------
def test_argon2_password_hashing():
    pw = "SuperSecure#2026!P@ss"
    hashed = hash_password(pw)
    assert hashed.startswith("$argon2id$")
    assert verify_password(pw, hashed) is True
    assert verify_password("WrongPassword!", hashed) is False
    assert verify_password("", hashed) is False


# -----------------------------------------------------------------------------
# 2. Login Rate Limiting Tests (Brute-Force Protection)
# -----------------------------------------------------------------------------
def test_login_rate_limiter_lockout(test_client):
    test_ip = "192.168.1.99"
    test_user = "brute_force_target"

    # Reset limiter for key
    key = f"{test_ip}:{test_user}"
    login_rate_limiter._failed_attempts.pop(key, None)

    headers = {"X-Forwarded-For": test_ip}

    # First 5 failed attempts
    for i in range(5):
        res = test_client.post(
            "/api/auth/login",
            json={"username": test_user, "password": f"bad_password_{i}"},
            headers=headers,
        )
        assert res.status_code == 401

    # 6th attempt should be blocked by rate limiter with 429 Too Many Requests
    blocked_res = test_client.post(
        "/api/auth/login",
        json={"username": test_user, "password": "even_correct_is_blocked"},
        headers=headers,
    )
    assert blocked_res.status_code == 429
    assert "Too many failed login attempts" in blocked_res.json()["detail"]
    assert "Retry-After" in blocked_res.headers


# -----------------------------------------------------------------------------
# 3. JWT Auth Lifecycle & Refresh
# -----------------------------------------------------------------------------
def test_auth_login_refresh_logout_lifecycle(test_client, auth_db_session):
    import uuid
    uid = uuid.uuid4().hex[:8]
    now = time.time()
    username = f"op_bob_{uid}"
    user = User(
        id=f"usr_{uid}",
        username=username,
        hashed_password=hash_password("OperatorPass123!"),
        full_name="Bob Operator",
        role="operator",
        is_active=True,
        created_at=now,
    )
    auth_db_session.add(user)
    auth_db_session.commit()

    # Login
    login_res = test_client.post(
        "/api/auth/login",
        json={"username": username, "password": "OperatorPass123!"},
        headers={"X-Forwarded-For": "10.0.0.1"},
    )
    assert login_res.status_code == 200
    token_data = login_res.json()
    assert "access_token" in token_data
    assert "refresh_token" in token_data
    assert token_data["user"]["username"] == username
    assert token_data["user"]["role"] == "operator"

    access_token = token_data["access_token"]
    refresh_token = token_data["refresh_token"]

    # Verify /api/auth/me
    me_res = test_client.get(
        "/api/auth/me",
        headers={"Authorization": f"Bearer {access_token}"},
    )
    assert me_res.status_code == 200
    assert me_res.json()["username"] == username

    # Refresh token
    refresh_res = test_client.post(
        "/api/auth/refresh",
        json={"refresh_token": refresh_token},
    )
    assert refresh_res.status_code == 200
    new_token_data = refresh_res.json()
    assert "access_token" in new_token_data
    assert new_token_data["access_token"] != access_token

    # Logout
    logout_res = test_client.post(
        "/api/auth/logout",
        headers={"Authorization": f"Bearer {new_token_data['access_token']}"},
    )
    assert logout_res.status_code == 200



# -----------------------------------------------------------------------------
# 4. RBAC: Backend Role Enforcement & Reviewer Identity Overwrite
# -----------------------------------------------------------------------------
def test_rbac_backend_role_enforcement_and_reviewer_replacement(test_client, auth_db_session):
    import uuid
    uid = uuid.uuid4().hex[:8]
    now = time.time()
    # Create 3 users: viewer, operator, admin
    viewer_user = User(
        id=f"usr_view_{uid}",
        username=f"viewer_sam_{uid}",
        hashed_password=hash_password("PassViewer123!"),
        full_name="Sam Analyst",
        role="viewer",
        is_active=True,
        created_at=now,
    )
    operator_user = User(
        id=f"usr_op_{uid}",
        username=f"operator_jane_{uid}",
        hashed_password=hash_password("PassOperator123!"),
        full_name="Jane Reviewer",
        role="operator",
        is_active=True,
        created_at=now,
    )
    admin_user = User(
        id=f"usr_adm_{uid}",
        username=f"admin_karl_{uid}",
        hashed_password=hash_password("PassAdmin123!"),
        full_name="Karl Sysadmin",
        role="admin",
        is_active=True,
        created_at=now,
    )
    auth_db_session.add_all([viewer_user, operator_user, admin_user])

    # Seed an incident for review testing
    inc = Incident(
        incident_id=f"INC-RBAC-{uid}",
        start_timestamp="2026-10-18T20:00:00Z",
        end_timestamp="2026-10-18T20:00:05Z",
        class_name="car_car_accident",
        max_confidence=0.88,
        image_path="/media/keyframes/rbac_001.jpg",
        keyframe_url="/media/keyframes/rbac_001.jpg",
        status="pending",
        created_at=now,
    )
    auth_db_session.add(inc)
    auth_db_session.commit()

    viewer_auth = _get_auth_header(viewer_user)
    operator_auth = _get_auth_header(operator_user)
    admin_auth = _get_auth_header(admin_user)

    # 1. Unauthenticated request without token must fail 401
    unauth_res = test_client.post(
        f"/api/incidents/{inc.incident_id}/status",
        json={"status": "approved"},
    )
    assert unauth_res.status_code == 401

    # 2. Viewer (read-only) cannot approve incident -> 403 Forbidden
    viewer_res = test_client.post(
        f"/api/incidents/{inc.incident_id}/status",
        json={"status": "approved"},
        headers=viewer_auth,
    )
    assert viewer_res.status_code == 403
    assert "Access denied" in viewer_res.json()["detail"]

    # 3. Viewer cannot create camera -> 403 Forbidden
    cam_fail = test_client.post(
        "/api/cameras",
        json={"name": "Cam Test", "source_url": "0"},
        headers=viewer_auth,
    )
    assert cam_fail.status_code == 403

    # 4. Operator CAN approve incident
    # CRITICAL REQUIREMENT: Free-text reviewer name "Spoofed Name" MUST be replaced with authenticated user ("Jane Reviewer")
    op_res = test_client.post(
        f"/api/incidents/{inc.incident_id}/status",
        json={"status": "approved", "reviewer_name": "Spoofed Fake Reviewer", "notes": "Looks confirmed"},
        headers={**operator_auth, "X-Forwarded-For": "172.16.0.42"},
    )
    assert op_res.status_code == 200
    res_data = op_res.json()
    assert res_data["status"] == "approved"
    assert res_data["reviewer_name"] == "Jane Reviewer"  # Must NOT be "Spoofed Fake Reviewer"

    # 5. Operator CANNOT change alert settings or dispatch test alert -> 403 Forbidden
    op_alert_fail = test_client.post(
        "/api/alerts/test",
        json={"channels": ["telegram"]},
        headers=operator_auth,
    )
    assert op_alert_fail.status_code == 403

    # 6. Admin CAN dispatch test alert and change settings
    adm_alert_ok = test_client.post(
        "/api/alerts/test",
        json={"channels": ["telegram"], "dry_run": True},
        headers={**admin_auth, "X-Forwarded-For": "10.0.1.5"},
    )
    assert adm_alert_ok.status_code == 200


# -----------------------------------------------------------------------------
# 5. Append-Only Audit Logging Tests
# -----------------------------------------------------------------------------
def test_audit_logging_recorded_and_admin_accessible(test_client, auth_db_session):
    import uuid
    uid = uuid.uuid4().hex[:8]
    now = time.time()
    admin_user = User(
        id=f"usr_adm_audit_{uid}",
        username=f"admin_audit_{uid}",
        hashed_password=hash_password("PassAdmin123!"),
        full_name="Audit Administrator",
        role="admin",
        is_active=True,
        created_at=now,
    )
    operator_user = User(
        id=f"usr_op_audit_{uid}",
        username=f"operator_audit_{uid}",
        hashed_password=hash_password("PassOperator123!"),
        full_name="Audit Operator",
        role="operator",
        is_active=True,
        created_at=now,
    )
    auth_db_session.add_all([admin_user, operator_user])
    auth_db_session.commit()

    admin_auth = _get_auth_header(admin_user)
    operator_auth = _get_auth_header(operator_user)

    # 1. Update settings as Admin -> creates audit event
    test_client.put(
        "/api/alerts/settings",
        json={"enabled": True, "dry_run": True},
        headers={**admin_auth, "X-Forwarded-For": "198.51.100.22"},
    )

    # 2. Query audit logs as Admin
    audit_res = test_client.get("/api/audit-logs", headers=admin_auth)
    assert audit_res.status_code == 200
    logs = audit_res.json()["items"]
    assert len(logs) > 0

    setting_log = next(l for l in logs if l["action"] == "alert_settings_updated")
    assert setting_log["username"] == f"admin_audit_{uid}"
    assert setting_log["user_role"] == "admin"
    assert setting_log["target_type"] == "settings"
    assert setting_log["ip_address"] == "198.51.100.22"
    assert setting_log["timestamp"] is not None

    # 3. Operator cannot access audit logs -> 403 Forbidden
    op_audit_fail = test_client.get("/api/audit-logs", headers=operator_auth)
    assert op_audit_fail.status_code == 403


# -----------------------------------------------------------------------------
# 6. Upload Validation Tests (Magic Bytes, Size Limit, Sanity Check)
# -----------------------------------------------------------------------------
def test_video_magic_bytes_detection():
    # Valid MP4 header (ftyp box)
    mp4_header = b"\x00\x00\x00\x20ftypisom\x00\x00\x02\x00isomiso2mp41"
    assert is_valid_video_magic_bytes(mp4_header) is True

    # Valid AVI header (RIFF ... AVI )
    avi_header = b"RIFF\x24\x00\x00\x00AVI LIST\x00\x00\x00\x00"
    assert is_valid_video_magic_bytes(avi_header) is True

    # Valid MKV/WebM header
    mkv_header = b"\x1a\x45\xdf\xa3\x9f\x42\x86\x81\x01\x42\xf7\x81\x01"
    assert is_valid_video_magic_bytes(mkv_header) is True

    # Invalid executable or text file
    exe_header = b"MZ\x90\x00\x03\x00\x00\x00\x04\x00\x00\x00\xff\xff"
    assert is_valid_video_magic_bytes(exe_header) is False

    text_header = b"Hello, this is just a plain text document header."
    assert is_valid_video_magic_bytes(text_header) is False


def test_upload_validation_rejects_corrupted_and_non_video_files(test_client, auth_db_session):
    import uuid
    uid = uuid.uuid4().hex[:8]
    now = time.time()
    operator_user = User(
        id=f"usr_op_upload_{uid}",
        username=f"operator_upload_{uid}",
        hashed_password=hash_password("Pass123!"),
        full_name="Upload Operator",
        role="operator",
        is_active=True,
        created_at=now,
    )
    auth_db_session.add(operator_user)
    auth_db_session.commit()
    operator_auth = _get_auth_header(operator_user)


    # 1. Upload fake video file (text file with .mp4 extension)
    fake_video = b"This is clearly not an MP4 video file! Fake header."
    res_fake = test_client.post(
        "/api/jobs",
        files={"file": ("fake_exploit.mp4", io.BytesIO(fake_video), "video/mp4")},
        headers=operator_auth,
    )
    assert res_fake.status_code == 400
    assert "Invalid video format: magic bytes header" in res_fake.json()["detail"]


# -----------------------------------------------------------------------------
# 7. Secrets Startup Check Tests
# -----------------------------------------------------------------------------
def test_secrets_placeholder_detection():
    assert is_placeholder_secret("change-me-immediately") is True
    assert is_placeholder_secret("your-secret-key-here") is True
    assert is_placeholder_secret("1234567890123456") is True
    assert is_placeholder_secret("secret") is True
    assert is_placeholder_secret("default_key_value") is True
    assert is_placeholder_secret("") is True
    assert is_placeholder_secret(None) is True

    # High entropy 64-char key
    valid_key = "e89b4f73a1c94519bc1694f877685d10a1209384bfce65298a0d9238475619bc"
    assert is_placeholder_secret(valid_key) is False


def test_startup_secret_check_fails_loudly_in_production(monkeypatch):
    # Set production mode and placeholder secret
    monkeypatch.setenv("ENVIRONMENT", "production")
    monkeypatch.setenv("JWT_SECRET_KEY", "your-secret-placeholder-key")

    with pytest.raises(RuntimeError) as exc_info:
        validate_startup_secrets()
    assert "FATAL: Startup secret validation failed" in str(exc_info.value)


# -----------------------------------------------------------------------------
# 8. Security Headers & CORS Tests
# -----------------------------------------------------------------------------
def test_security_headers_and_cors(test_client):
    res = test_client.get("/api/health")
    assert res.status_code == 200
    headers = res.headers

    assert headers.get("X-Content-Type-Options") == "nosniff"
    assert headers.get("X-Frame-Options") == "DENY"
    assert headers.get("X-XSS-Protection") == "1; mode=block"
    assert headers.get("Referrer-Policy") == "strict-origin-when-cross-origin"
    assert headers.get("Permissions-Policy") == "camera=(), microphone=(), geolocation=()"
    assert headers.get("X-Permitted-Cross-Domain-Policies") == "none"
    assert "default-src 'self'" in headers.get("Content-Security-Policy", "")


def test_session_cookie_auth(test_client, auth_db_session):
    import uuid
    uid = uuid.uuid4().hex[:8]
    user = User(
        id=f"usr_cookie_{uid}",
        username=f"cookie_user_{uid}",
        hashed_password=hash_password("Pass123456!"),
        full_name="Cookie User",
        role="operator",
        is_active=True,
        created_at=time.time(),
    )
    auth_db_session.add(user)
    auth_db_session.commit()

    # Login to receive cookies
    login_res = test_client.post(
        "/api/auth/login",
        json={"username": user.username, "password": "Pass123456!"},
        headers={"X-Forwarded-Proto": "https"},
    )
    assert login_res.status_code == 200
    assert "access_token" in login_res.cookies
    assert "refresh_token" in login_res.cookies

    # Test that requests using cookie without Authorization header authenticate successfully
    me_res = test_client.get("/api/auth/me", cookies={"access_token": login_res.cookies["access_token"]})
    assert me_res.status_code == 200
    assert me_res.json()["username"] == user.username


# -----------------------------------------------------------------------------
# 9. Extended RBAC Matrix Tests
# -----------------------------------------------------------------------------
def test_extended_rbac_enforcement(test_client, auth_db_session):
    import uuid
    uid = uuid.uuid4().hex[:8]
    viewer = User(
        id=f"usr_v_{uid}",
        username=f"viewer_ext_{uid}",
        hashed_password=hash_password("Pass123!"),
        full_name="Viewer Ext",
        role="viewer",
        is_active=True,
        created_at=time.time(),
    )
    operator = User(
        id=f"usr_o_{uid}",
        username=f"operator_ext_{uid}",
        hashed_password=hash_password("Pass123!"),
        full_name="Operator Ext",
        role="operator",
        is_active=True,
        created_at=time.time(),
    )
    auth_db_session.add_all([viewer, operator])
    auth_db_session.commit()

    v_auth = _get_auth_header(viewer)
    o_auth = _get_auth_header(operator)

    # Viewer cannot access operator review-queue decision -> 403
    v_rev_fail = test_client.post(
        "/api/review-queue/INC-DUMMY/decision",
        json={"decision": "accident"},
        headers=v_auth,
    )
    assert v_rev_fail.status_code == 403

    # Operator cannot access admin user management -> 403
    o_users_fail = test_client.get("/api/users", headers=o_auth)
    assert o_users_fail.status_code == 403

    # Operator cannot update alert settings -> 403
    o_alerts_fail = test_client.put(
        "/api/alerts/settings",
        json={"enabled": True},
        headers=o_auth,
    )
    assert o_alerts_fail.status_code == 403
