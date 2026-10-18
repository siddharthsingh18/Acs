"""
backend.app.routers.auth
Authentication endpoints: login, token refresh, current user profile, and logout.
Features Argon2 verification, brute-force rate limiting, JWT generation, and audit logging.
"""
import time
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Request, Response, status
from sqlmodel import Session, select

from backend.app.core.security import (
    create_access_token,
    create_refresh_token,
    decode_token,
    login_rate_limiter,
    verify_password,
)
from backend.app.db.database import get_session
from backend.app.db.models import User
from backend.app.deps import extract_client_ip, get_current_user
from backend.app.schemas import LoginRequest, RefreshTokenRequest, TokenResponse, UserResponse
from backend.app.services.audit_service import log_audit_event

router = APIRouter(prefix="/api/auth", tags=["Authentication"])


@router.post("/login", response_model=TokenResponse)
def login(
    req: LoginRequest,
    request: Request,
    response: Response,
    session: Session = Depends(get_session),
):
    """Authenticates credentials against Argon2 hash, enforcing brute-force rate limits."""
    client_ip = extract_client_ip(request)
    user_agent = request.headers.get("User-Agent")
    clean_username = req.username.strip().lower()

    # 1. Rate limiter check (5 failed attempts per 5 minutes)
    locked, remaining = login_rate_limiter.is_locked(client_ip, clean_username)
    if locked:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=f"Too many failed login attempts. Locked out for {remaining} seconds.",
            headers={"Retry-After": str(remaining)},
        )

    # 2. Look up user
    user = session.exec(select(User).where(User.username == clean_username)).first()
    if not user or not verify_password(req.password, user.hashed_password):
        login_rate_limiter.record_failure(client_ip, clean_username)
        # Record failed audit log
        log_audit_event(
            session=session,
            username=clean_username,
            user_role="unknown",
            action="auth_login_failed",
            target_type="auth",
            target_id=clean_username,
            ip_address=client_ip,
            user_agent=user_agent,
            details={"reason": "Invalid credentials", "ip": client_ip},
        )
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="User account has been deactivated.",
        )

    # 3. Success: clear failed rate limiting attempts and record login
    login_rate_limiter.record_success(client_ip, clean_username)
    user.last_login_at = time.time()
    session.add(user)
    session.commit()
    session.refresh(user)

    # 4. Generate JWT tokens
    token_claims = {
        "sub": user.id,
        "username": user.username,
        "role": user.role,
        "full_name": user.full_name,
    }
    access_token = create_access_token(token_claims)
    refresh_token = create_refresh_token(subject=user.id)

    # 5. Set HTTP-only cookies
    response.set_cookie(
        key="access_token",
        value=access_token,
        httponly=True,
        samesite="lax",
        secure=False,  # Set to True in HTTPS production
        max_age=1800,
    )
    response.set_cookie(
        key="refresh_token",
        value=refresh_token,
        httponly=True,
        samesite="lax",
        secure=False,
        max_age=7 * 86400,
    )

    # 6. Audit log
    log_audit_event(
        session=session,
        username=user.username,
        user_role=user.role,
        action="auth_login_success",
        target_type="auth",
        target_id=user.id,
        ip_address=client_ip,
        user_agent=user_agent,
        user_id=user.id,
    )

    return TokenResponse(
        access_token=access_token,
        refresh_token=refresh_token,
        token_type="bearer",
        user=UserResponse(
            id=user.id,
            username=user.username,
            full_name=user.full_name,
            role=user.role,
            is_active=user.is_active,
            created_at=user.created_at,
            last_login_at=user.last_login_at,
        ),
    )


@router.post("/refresh", response_model=TokenResponse)
def refresh_token(
    request: Request,
    response: Response,
    req: Optional[RefreshTokenRequest] = None,
    session: Session = Depends(get_session),
):
    """Exchanges a valid refresh token for a fresh access token."""
    token = None
    if req and req.refresh_token:
        token = req.refresh_token
    else:
        token = request.cookies.get("refresh_token")

    if not token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Refresh token required.",
        )

    try:
        payload = decode_token(token)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Invalid or expired refresh token: {str(e)}",
        )

    if payload.get("type") != "refresh":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token type (expected refresh token).",
        )

    user_id = payload.get("sub")
    user = session.get(User, user_id)
    if not user or not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User account no longer active or exists.",
        )

    token_claims = {
        "sub": user.id,
        "username": user.username,
        "role": user.role,
        "full_name": user.full_name,
    }
    new_access_token = create_access_token(token_claims)
    new_refresh_token = create_refresh_token(subject=user.id)

    response.set_cookie(
        key="access_token",
        value=new_access_token,
        httponly=True,
        samesite="lax",
        secure=False,
        max_age=1800,
    )
    response.set_cookie(
        key="refresh_token",
        value=new_refresh_token,
        httponly=True,
        samesite="lax",
        secure=False,
        max_age=7 * 86400,
    )

    return TokenResponse(
        access_token=new_access_token,
        refresh_token=new_refresh_token,
        token_type="bearer",
        user=UserResponse(
            id=user.id,
            username=user.username,
            full_name=user.full_name,
            role=user.role,
            is_active=user.is_active,
            created_at=user.created_at,
            last_login_at=user.last_login_at,
        ),
    )


@router.get("/me", response_model=UserResponse)
def get_me(current_user: User = Depends(get_current_user)):
    """Returns profile of currently authenticated user."""
    return UserResponse(
        id=current_user.id,
        username=current_user.username,
        full_name=current_user.full_name,
        role=current_user.role,
        is_active=current_user.is_active,
        created_at=current_user.created_at,
        last_login_at=current_user.last_login_at,
    )


@router.post("/logout")
def logout(
    request: Request,
    response: Response,
    current_user: User = Depends(get_current_user),
    session: Session = Depends(get_session),
):
    """Logs out user, clearing access and refresh cookies, and auditing event."""
    client_ip = extract_client_ip(request)
    response.delete_cookie("access_token")
    response.delete_cookie("refresh_token")

    log_audit_event(
        session=session,
        username=current_user.username,
        user_role=current_user.role,
        action="auth_logout",
        target_type="auth",
        target_id=current_user.id,
        ip_address=client_ip,
        user_id=current_user.id,
    )

    return {"status": "ok", "message": "Successfully logged out"}
