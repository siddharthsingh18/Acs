# Security Policy

## Supported Versions

The Automated CCTV Accident Detection System (ACS) team provides security maintenance and vulnerability updates for the following release branches:

| Version | Supported | Notes |
| :--- | :---: | :--- |
| `1.0.x` (current `main`) | :white_check_mark: | Active production and cloud demo releases |
| `< 1.0.0` | :x: | Legacy prototyping scripts, no longer supported |

---

## Reporting a Vulnerability

We take the security and integrity of critical infrastructure and surveillance telemetry seriously. If you discover a security vulnerability in ACS, please report it via private coordinated disclosure.

- **Private Reporting Email**: `security@acs-surveillance.local` (or file a private GitHub Advisory).
- **Required Details**:
  - Detailed description of the vulnerability, including attack vector and potential impact.
  - Reproducible proof-of-concept (PoC) code or step-by-step reproduction steps.
  - Affected components (backend API, RTSP stream pipeline, edge agent, frontend dashboard, or database).
- **Response Timeline**:
  - **Initial Acknowledgment**: Within 48 hours of receipt.
  - **Triage & Status Assessment**: Within 5 business days.
  - **Target Fix & Public Disclosure**: Coordinated within 30 days of confirmed validation.

Please do **NOT** open public GitHub issues for undisclosed security vulnerabilities.

---

## Security Architecture & Defenses

ACS incorporates defense-in-depth principles across the entire edge-to-cloud computer vision pipeline:

### 1. Zero-Trust Startup Secret Validation
In production environments (`ENVIRONMENT=production` or `STRICT_SECRETS=true`), ACS validates that critical operational secrets are configured with cryptographically strong keys:
- `JWT_SECRET_KEY` (minimum 32 bytes hex/base64, rejecting default placeholders like `change-me`).
- `FERNET_KEY` (valid URL-safe base64 32-byte key for encryption at rest).
- `INITIAL_ADMIN_PASSWORD` (rejecting common default passwords).
- Refuses to boot if live alerts are enabled without valid provider credentials.

### 2. Role-Based Access Control (RBAC) & Authentication
- **Authentication**: Stateless HMAC-SHA256 JWT tokens with configurable expiration and automatic rotation.
- **Role Hierarchy**:
  - `admin`: User provisioning, camera configuration, retention settings, and audit log access.
  - `operator`: Video job submission, review queue triage, manual alert dispatch, and model tuning.
  - `viewer`: Read-only telemetry, camera grid viewing, and analytics dashboard access.
  - `edge_node`: Dedicated API keys (`X-Edge-Api-Key`) restricted exclusively to collision event ingestion.

### 3. Media Ingestion & Upload Protections
- **Magic Bytes Verification**: File headers are strictly inspected for valid container signatures (`ftyp`, `moov`, `RIFF/AVI`, Matroska/WebM, MPEG-TS) before decoding.
- **Upload Size Limits**: Multi-part uploads are strictly capped at 500 MB to prevent disk exhaustion denial-of-service (DoS).
- **Path Traversal Defense**: All filenames are passed through `os.path.basename` and mapped to sanitized identifiers, rejecting directory traversal (`../`) sequences.
- **Ephemeral Demo Storage**: When `STORE_UPLOADS=false` or `PUBLIC_DEMO=true`, uploaded media files are processed in-memory or purged immediately upon job completion.

### 4. Privacy & Regulatory Compliance (GDPR / PII Protection)
- **Automated Anonymization**: Bounding-box detection triggers automated Gaussian blurring for human faces and vehicle license plates prior to external transmission.
- **Tiered Access to Unredacted Media**: Original, unblurred high-resolution media is restricted to authorized administrative users with full cryptographic audit trail logging.
- **Data Retention & Auto-Purge**: Video clips older than `CLIP_RETENTION_DAYS` (default 7 days) and incident metadata older than `INCIDENT_RETENTION_DAYS` (default 30 days) are automatically deleted by a background retention worker.

### 5. Alert Storm Dampening & Anti-Exhaustion Controls
- **Rate-Limiting Interlock**: Dispatches to external notification gateways (Telegram, SMTP, Web Push) are metered by a sliding window rate limiter (`STORM_RATE_LIMIT_SECONDS`, `STORM_MAX_ALERTS_PER_WINDOW`).
- **Demo Interlock**: In `PUBLIC_DEMO=true` or `DEMO_MODE=true`, `alerts_dry_run` is mathematically enforced, preventing simulated incidents from generating external phone/SMS/email alerts.

### 6. Network Security & Modern Headers
- **Content Security Policy (CSP)**: Restricts script, style, and media sources to authorized origins.
- **Standard Defensive Headers**: `X-Content-Type-Options: nosniff`, `X-Frame-Options: DENY`, `X-XSS-Protection: 1; mode=block`, `Referrer-Policy: strict-origin-when-cross-origin`, `Strict-Transport-Security` (HSTS).
