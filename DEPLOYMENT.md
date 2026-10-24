# ACS Research Prototype Deployment Guide

> **Not a production-readiness statement.** This guide describes configuration examples only. The model and its reported performance have not passed reproducible promotion gates. Use only in isolated demos or research; a human must verify every alert.

This document specifies the deployment architecture, HTTPS reverse-proxy configuration, secrets management, and role-based operational controls for the **Automated CCTV Accident Detection & Emergency Dispatch System (ACS)**.

---

## 1. Security Architecture & Threat Model

```
                    Internet / Corporate Network
                               │
                       [HTTPS :443 / WSS]
                               ▼
               ┌───────────────────────────────┐
               │    TLS Termination Proxy      │
               │      (Nginx or Caddy)         │
               │   - TLS 1.3 / Strict HSTS     │
               │   - Security Headers / CSP    │
               │   - 500MB Client Max Body     │
               └───────────────┬───────────────┘
                               │
            ┌──────────────────┴──────────────────┐
            ▼                                     ▼
┌───────────────────────┐             ┌───────────────────────┐
│   FastAPI Backend     │             │     MediaMTX RTSP     │
│   (127.0.0.1:8000)    │             │   (127.0.0.1:8554)    │
│  - Argon2id Auth      │             │  - CCTV RTSP Relays   │
│  - JWT Access/Refresh │             │  - Authenticated WSS  │
│  - RBAC (3 Roles)     │             └───────────────────────┘
│  - Append-Only Audit  │
│  - Magic Byte Check   │
└───────────┬───────────┘
            │
            ▼
┌───────────────────────┐
│     SQLite DB &       │
│  Rolling Media Buffer │
└───────────────────────┘
```

---

## 2. HTTPS Reverse Proxy Configuration

In real production environments, **ACS must run behind a TLS-terminating reverse proxy**. Never expose the Uvicorn ASGI server directly to the public internet.

### Option A: Nginx Configuration (`/etc/nginx/sites-available/acs.conf`)

```nginx
# Rate limiting zone for authentication endpoints
limit_req_zone $binary_remote_addr zone=acs_auth_limit:10m rate=5r/m;

# HTTP -> HTTPS Redirect
server {
    listen 80;
    listen [::]:80;
    server_name acs.example.com;
    return 301 https://$host$request_uri;
}

server {
    listen 443 ssl http2;
    listen [::]:443 ssl http2;
    server_name acs.example.com;

    # SSL Certificates (Certbot / Let's Encrypt)
    ssl_certificate /etc/letsencrypt/live/acs.example.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/acs.example.com/privkey.pem;

    # TLS Hardening
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers ECDHE-ECDSA-AES128-GCM-SHA256:ECDHE-RSA-AES128-GCM-SHA256:ECDHE-ECDSA-AES256-GCM-SHA384:ECDHE-RSA-AES256-GCM-SHA384;
    ssl_prefer_server_ciphers off;
    ssl_session_cache shared:SSL:10m;
    ssl_session_timeout 1d;
    ssl_session_tickets off;

    # HSTS (Preload eligible)
    add_header Strict-Transport-Security "max-age=63072000; includeSubDomains; preload" always;

    # Security Headers
    add_header X-Content-Type-Options "nosniff" always;
    add_header X-Frame-Options "DENY" always;
    add_header X-XSS-Protection "1; mode=block" always;
    add_header Referrer-Policy "strict-origin-when-cross-origin" always;
    add_header Content-Security-Policy "default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline' https://fonts.googleapis.com; font-src 'self' https://fonts.gstatic.com data:; img-src 'self' data: blob: https://*.tile.openstreetmap.org; media-src 'self' blob:; connect-src 'self' ws: wss:; frame-ancestors 'none';" always;

    # Max upload limit (500MB video evidence)
    client_max_body_size 500M;

    # Frontend Single Page App static assets
    root /var/www/acs/frontend/dist;
    index index.html;

    location / {
        try_files $uri $uri/ /index.html;
    }

    # Authentication Rate Limiting
    location /api/auth/login {
        limit_req zone=acs_auth_limit burst=3 nodelay;
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }

    # REST API Reverse Proxy
    location /api/ {
        proxy_pass http://127.0.0.1:8000;
        proxy_http_version 1.1;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_read_timeout 300s;
        proxy_connect_timeout 75s;
    }

    # Live CCTV WebSocket Feeds
    location /api/cameras/ {
        proxy_pass http://127.0.0.1:8000;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_read_timeout 3600s;
        proxy_send_timeout 3600s;
    }
}
```

### Option B: Caddyfile (`/etc/caddy/Caddyfile`)

```caddy
acs.example.com {
    # Automatic HTTPS with Let's Encrypt
    header {
        Strict-Transport-Security "max-age=63072000; includeSubDomains; preload"
        X-Content-Type-Options "nosniff"
        X-Frame-Options "DENY"
        Referrer-Policy "strict-origin-when-cross-origin"
    }

    request_body {
        max_size 500MB
    }

    handle /api/* {
        reverse_proxy 127.0.0.1:8000
    }

    handle {
        root * /var/www/acs/frontend/dist
        try_files {path} /index.html
        file_server
    }
}
```

---

## 3. Environment Variables & Secrets Management

Secrets must **NEVER** be committed into git or hardcoded. They are loaded exclusively from the environment.

### Production Environment Variables

| Variable | Description | Required in Production | Default / Example |
| :--- | :--- | :--- | :--- |
| `ENVIRONMENT` | Environment stage (`production` or `development`) | **YES** | `production` |
| `JWT_SECRET_KEY` | High-entropy 256-bit key for JWT signing | **YES** | `openssl rand -hex 32` |
| `FERNET_KEY` | 32-byte URL-safe base64 key for camera credential encryption | **YES** | Generated via `Fernet.generate_key()` |
| `INITIAL_ADMIN_PASSWORD` | Bootstrap password for initial root administrator | **YES** | Strong password (min 12 chars) |
| `CORS_ORIGINS` | Comma-separated list of permitted web origins | **YES** | `https://acs.example.com` |
| `MAX_UPLOAD_SIZE_BYTES` | Video file upload cap in bytes | Optional | `524288000` (500 MB) |
| `TELEGRAM_BOT_TOKEN` | Bot token if Telegram alert channel is enabled | If channel active | Real bot token from @BotFather |
| `TELEGRAM_CHAT_ID` | Telegram chat/channel target ID | If channel active | Real target chat ID |
| `SMTP_APP_PASSWORD` | Mail dispatch app credential | If channel active | Real SMTP credential |
| `DATABASE_URL` | SQLite file path or PostgreSQL DSN | Optional | `sqlite:///acs_database.db` |

### Startup Fail-Loud Secret Validator

When `ENVIRONMENT=production`, the application executes `validate_startup_secrets()` during lifespan initialization.

If any mandatory secret:
1. Is unset or empty, OR
2. Matches known default/insecure placeholders (e.g. `your_telegram_bot_token`, `changeme`, `secret`, `admin`, `placeholder`, `<...>`),

the backend **immediately halts with a fatal `RuntimeError`** before binding to any network ports.

---

## 4. Role-Based Access Control (RBAC) Matrix

Backend RBAC is strictly enforced across all REST endpoints via FastAPI dependency injection:

| Capability / API Route | Viewer | Operator | Administrator |
| :--- | :---: | :---: | :---: |
| View live CCTV streams & camera wall | Yes | Yes | Yes |
| View incident history & replay maps | Yes | Yes | Yes |
| Download accident summary PDFs | Yes | Yes | Yes |
| Review queue: Approve / Reject / Classify | No (403) | **Yes** | **Yes** |
| Dispatch emergency alerts for incident | No (403) | **Yes** | **Yes** |
| Upload video footage for inference | No (403) | **Yes** | **Yes** |
| Register, edit, or delete CCTV cameras | No (403) | No (403) | **Yes** |
| Modify alert channel keys & dispatch settings | No (403) | No (403) | **Yes** |
| Trigger test alert dispatches | No (403) | No (403) | **Yes** |
| Adjust retention policies & rolling buffer caps | No (403) | No (403) | **Yes** |
| View forensic Security Audit Log (`/api/audit-logs`) | No (403) | No (403) | **Yes** |
| Manage user accounts & assign roles (`/api/users`) | No (403) | No (403) | **Yes** |

---

## 5. Defense-in-Depth Measures

### Password Hashing (Argon2id)
Passwords are salted and hashed using Argon2id (`argon2-cffi`), configured per OWASP specifications:
- Memory Cost: 65,536 KiB (64 MiB)
- Time Cost: 3 iterations
- Parallelism: 4 threads

### Brute-Force Rate Limiting
The login endpoint (`/api/auth/login`) tracks IP address and target account failures:
- Maximum 5 failed attempts per 300-second window.
- Exceeding the threshold triggers HTTP `429 Too Many Requests` with a `Retry-After: 300` header.
- Successful authentication resets the failure count for that IP.

### Immutable Append-Only Audit Trail
Every security-sensitive state modification records an immutable entry in the `AuditLog` table containing:
- Timestamp (UTC ISO-8601 & Unix epoch)
- Actor ID, Username, and Active Role
- Action type (`incident.approved`, `camera.created`, `settings.alerts.updated`, etc.)
- Target entity type and ID
- Client IP address (with `X-Forwarded-For` proxy traversal)
- Structured JSON snapshot of modified fields

### Video Upload Validation
All incoming footage to `/api/jobs/upload` must pass three validation gates:
1. **Container Magic Bytes**: Inspects file header signatures before saving (`ftyp` for MP4, `RIFF` for AVI, `\x1a\x45\xdf\xa3` for MKV/WebM, `0x47` for MPEG-TS).
2. **Streaming Chunk Size Limit**: Aborts immediately if the payload exceeds 500 MB, preventing memory exhaustion and disk-filling DoS attacks.
3. **Codec / Decode Sanity Check**: Runs an initial frame decode via OpenCV / ffprobe to reject corrupted or malformed payloads.

---

## 6. Systemd Deployment (Ubuntu / Debian Linux)

### Backend Service (`/etc/systemd/system/acs-backend.service`)

```ini
[Unit]
Description=ACS Automated CCTV Accident Detection Backend
After=network.target

[Service]
Type=simple
User=acs
Group=acs
WorkingDirectory=/var/www/acs
Environment="PATH=/var/www/acs/venv/bin"
EnvironmentFile=/etc/acs/acs.env
ExecStart=/var/www/acs/venv/bin/uvicorn backend.app.main:app --host 127.0.0.1 --port 8000 --workers 4
Restart=always
RestartSec=5
LimitNOFILE=65535

[Install]
WantedBy=multi-user.target
```

### Initial Setup Commands

```bash
# 1. Create secure system environment file
sudo mkdir -p /etc/acs
sudo touch /etc/acs/acs.env
sudo chmod 600 /etc/acs/acs.env

# 2. Populate secrets
cat <<EOF | sudo tee /etc/acs/acs.env
ENVIRONMENT=production
JWT_SECRET_KEY=$(openssl rand -hex 32)
FERNET_KEY=$(python3 -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())")
INITIAL_ADMIN_PASSWORD=SuperAdminPass2026!#
CORS_ORIGINS=https://acs.example.com
EOF

# 3. Enable and start daemon
sudo systemctl daemon-reload
sudo systemctl enable acs-backend
sudo systemctl start acs-backend
sudo systemctl status acs-backend
```

---

## 7. Vulnerability Scanning

Both dependency trees are continuously audited:

```bash
# Python backend dependencies
./venv/bin/pip-audit

# Node.js frontend dependencies
cd frontend && npm audit
```

---

## 8. Surveillance Privacy, Vault Hardening & Retention

### Vault Directory Permissions
Unredacted original footage is saved to `media/unredacted/` and deliberately not mounted to any public static web paths. In production, ensure restricted operating system permissions:

```bash
# Secure media directories: only the backend system user can access unredacted media
chmod 700 /var/www/acs/media/unredacted
chmod 750 /var/www/acs/media/clips
chmod 750 /var/www/acs/media/keyframes
```

### Scheduled Lifecycle Retention Job
In addition to runtime retention checks, configure a daily systemd timer or cron job to execute automated cleanup:

```bash
# Optional daily automated cleanup trigger via curl
0 3 * * * curl -X POST -H "Authorization: Bearer $ADMIN_TOKEN" https://127.0.0.1:8000/api/incidents/retention/cleanup
```

### Legal & Regulatory Compliance
CCTV operations recording individuals and vehicles require strict adherence to local surveillance legislation (e.g. GDPR, CCPA, UAE Personal Data Protection Law) and the physical premises owner's policy. Ensure public disclosure signage is installed and that unredacted vault footage is accessed exclusively for formal law enforcement or judicial inquiries.
