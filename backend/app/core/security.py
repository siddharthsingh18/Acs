"""
backend.app.core.security
Enterprise authentication, cryptographic hashing (Argon2), JWT token handling,
login rate limiting, and secret validation.
"""
import os
import re
import secrets
import threading
import time
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, Optional, Tuple

import jwt
from argon2 import PasswordHasher

# -----------------------------------------------------------------------------
# Argon2 Password Hashing
# -----------------------------------------------------------------------------
# Default parameters: t=3 (iterations), m=65536 (64 MiB RAM), p=4 (parallelism)
ph = PasswordHasher(time_cost=3, memory_cost=65536, parallelism=4)


def hash_password(password: str) -> str:
    """Hashes a plaintext password using Argon2id."""
    return ph.hash(password)


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verifies a plaintext password against an Argon2 hash.
    
    Returns True if valid, False otherwise.
    """
    try:
        return ph.verify(hashed_password, plain_password)
    except Exception:
        return False


# -----------------------------------------------------------------------------
# Secrets Validation & Secret Management
# -----------------------------------------------------------------------------
PLACEHOLDER_PATTERNS = [
    r"^your-secret.*$",
    r"^change-?me.*$",
    r"^secret.*$",
    r"^password.*$",
    r"^123456.*$",
    r"^placeholder.*$",
    r"^default.*$",
    r"^admin.*$",
    r"^test.*$",
]


def is_placeholder_secret(value: Optional[str]) -> bool:
    """Detects whether a secret matches common placeholder or insecure patterns."""
    if not value or len(value.strip()) < 16:
        return True
    lowered = value.strip().lower()
    for pattern in PLACEHOLDER_PATTERNS:
        if re.match(pattern, lowered):
            return True
    return False


def get_jwt_secret_key() -> str:
    """Retrieves JWT secret key from environment, with validation."""
    key = os.getenv("JWT_SECRET_KEY") or os.getenv("SECRET_KEY")
    is_prod = (
        os.getenv("ENVIRONMENT", "").lower() in ("production", "prod")
        or os.getenv("STRICT_SECRETS", "").lower() in ("true", "1")
    )

    if not key or is_placeholder_secret(key):
        if is_prod:
            raise RuntimeError(
                "FATAL: Insecure or missing JWT_SECRET_KEY in production! "
                "JWT_SECRET_KEY must be set in environment with at least 32 high-entropy characters. "
                "Generate one using: python -c 'import secrets; print(secrets.token_hex(32))'"
            )
        # Development fallback: generate deterministic or ephemeral dev key
        return os.getenv("DEV_JWT_SECRET_KEY", "acs-dev-ephemeral-jwt-signing-secret-key-2026-v2")
    return key


JWT_ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "30"))
REFRESH_TOKEN_EXPIRE_DAYS = int(os.getenv("REFRESH_TOKEN_EXPIRE_DAYS", "7"))


def create_access_token(data: Dict[str, Any], expires_delta: Optional[timedelta] = None) -> str:
    """Creates a signed JWT access token containing subject, username, and role."""
    to_encode = data.copy()
    now = datetime.now(timezone.utc)
    if expires_delta:
        expire = now + expires_delta
    else:
        expire = now + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({
        "exp": expire,
        "iat": now,
        "type": "access",
        "jti": secrets.token_hex(16),
    })
    return jwt.encode(to_encode, get_jwt_secret_key(), algorithm=JWT_ALGORITHM)



def create_refresh_token(subject: str, expires_delta: Optional[timedelta] = None) -> str:
    """Creates a signed JWT refresh token."""
    now = datetime.now(timezone.utc)
    if expires_delta:
        expire = now + expires_delta
    else:
        expire = now + timedelta(days=REFRESH_TOKEN_EXPIRE_DAYS)
    payload = {
        "sub": str(subject),
        "exp": expire,
        "iat": now,
        "type": "refresh",
        "jti": secrets.token_hex(16),
    }
    return jwt.encode(payload, get_jwt_secret_key(), algorithm=JWT_ALGORITHM)


def decode_token(token: str) -> Dict[str, Any]:
    """Decodes and cryptographically verifies a JWT token.
    
    Raises:
        jwt.PyJWTError on expiration or invalid signature.
    """
    return jwt.decode(token, get_jwt_secret_key(), algorithms=[JWT_ALGORITHM])


# -----------------------------------------------------------------------------
# Login Rate Limiter (Brute-Force Protection)
# -----------------------------------------------------------------------------
class LoginRateLimiter:
    """Thread-safe in-memory login rate limiter.
    
    Blocks IP/username combinations exceeding max failed attempts within the lockout window.
    """

    def __init__(self, max_attempts: int = 5, lockout_seconds: int = 300):
        self.max_attempts = max_attempts
        self.lockout_seconds = lockout_seconds
        self._failed_attempts: Dict[str, list] = {}
        self._lock = threading.Lock()

    def _make_key(self, ip: Optional[str], username: str) -> str:
        clean_ip = (ip or "unknown").strip()
        clean_user = username.strip().lower()
        return f"{clean_ip}:{clean_user}"

    def is_locked(self, ip: Optional[str], username: str) -> Tuple[bool, int]:
        """Checks if the key is currently locked out.
        
        Returns (is_locked, remaining_seconds).
        """
        key = self._make_key(ip, username)
        now = time.time()
        cutoff = now - self.lockout_seconds

        with self._lock:
            attempts = [t for t in self._failed_attempts.get(key, []) if t > cutoff]
            self._failed_attempts[key] = attempts
            if len(attempts) >= self.max_attempts:
                oldest_in_window = min(attempts)
                remaining = int(self.lockout_seconds - (now - oldest_in_window))
                return True, max(1, remaining)
            return False, 0

    def record_failure(self, ip: Optional[str], username: str) -> int:
        """Records a failed authentication attempt.
        
        Returns total active failed attempts.
        """
        key = self._make_key(ip, username)
        now = time.time()
        cutoff = now - self.lockout_seconds

        with self._lock:
            attempts = [t for t in self._failed_attempts.get(key, []) if t > cutoff]
            attempts.append(now)
            self._failed_attempts[key] = attempts
            return len(attempts)

    def record_success(self, ip: Optional[str], username: str) -> None:
        """Clears failed attempts upon successful authentication."""
        key = self._make_key(ip, username)
        with self._lock:
            self._failed_attempts.pop(key, None)


login_rate_limiter = LoginRateLimiter(max_attempts=5, lockout_seconds=300)
