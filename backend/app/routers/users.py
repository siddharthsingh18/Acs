"""
backend.app.routers.users
Admin-only User Management endpoints.
Supports creating, listing, modifying, and deactivating user accounts with role enforcement and audit logging.
"""
import time
import uuid
from typing import List

from fastapi import APIRouter, Depends, HTTPException, Request, status
from sqlmodel import Session, func, select

from backend.app.core.security import hash_password
from backend.app.db.database import get_session
from backend.app.db.models import User
from backend.app.deps import extract_client_ip, require_admin
from backend.app.schemas import UserCreateRequest, UserResponse, UserUpdateRequest
from backend.app.services.audit_service import log_audit_event

router = APIRouter(prefix="/api/users", tags=["User Management"])


@router.get("", response_model=List[UserResponse])
def list_users(
    session: Session = Depends(get_session),
    admin_user: User = Depends(require_admin),
):
    """Lists all user accounts (Admin only)."""
    users = session.exec(select(User).order_by(User.created_at.desc())).all()
    return [
        UserResponse(
            id=u.id,
            username=u.username,
            full_name=u.full_name,
            role=u.role,
            is_active=u.is_active,
            created_at=u.created_at,
            last_login_at=u.last_login_at,
        )
        for u in users
    ]


@router.post("", response_model=UserResponse, status_code=status.HTTP_201_CREATED)
def create_user(
    req: UserCreateRequest,
    request: Request,
    session: Session = Depends(get_session),
    admin_user: User = Depends(require_admin),
):
    """Registers a new user account with Argon2 password hashing (Admin only)."""
    clean_username = req.username.strip().lower()
    if req.role not in ("viewer", "operator", "admin"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Role must be one of: 'viewer', 'operator', 'admin'.",
        )

    existing = session.exec(select(User).where(User.username == clean_username)).first()
    if existing:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Username '{clean_username}' already exists.",
        )

    new_user = User(
        id=f"usr_{uuid.uuid4().hex[:12]}",
        username=clean_username,
        hashed_password=hash_password(req.password),
        full_name=req.full_name.strip(),
        role=req.role,
        is_active=True,
        created_at=time.time(),
    )
    session.add(new_user)
    session.commit()
    session.refresh(new_user)

    client_ip = extract_client_ip(request)
    log_audit_event(
        session=session,
        username=admin_user.username,
        user_role=admin_user.role,
        action="user_created",
        target_type="user",
        target_id=new_user.id,
        ip_address=client_ip,
        details={"username": new_user.username, "role": new_user.role},
        user_id=admin_user.id,
    )

    return UserResponse(
        id=new_user.id,
        username=new_user.username,
        full_name=new_user.full_name,
        role=new_user.role,
        is_active=new_user.is_active,
        created_at=new_user.created_at,
        last_login_at=new_user.last_login_at,
    )


@router.put("/{user_id}", response_model=UserResponse)
def update_user(
    user_id: str,
    req: UserUpdateRequest,
    request: Request,
    session: Session = Depends(get_session),
    admin_user: User = Depends(require_admin),
):
    """Modifies user details, role, status, or credentials (Admin only)."""
    user = session.get(User, user_id)
    if not user:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found.")

    changes = {}

    # Check for last admin protection
    if (req.is_active is False or (req.role and req.role != "admin")) and user.role == "admin":
        active_admin_count = session.exec(
            select(func.count(User.id)).where(User.role == "admin", User.is_active == True)
        ).one()
        if active_admin_count <= 1:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Cannot demote or deactivate the last remaining active Administrator.",
            )

    if req.full_name is not None:
        user.full_name = req.full_name.strip()
        changes["full_name"] = user.full_name

    if req.role is not None:
        if req.role not in ("viewer", "operator", "admin"):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Role must be one of: 'viewer', 'operator', 'admin'.",
            )
        user.role = req.role
        changes["role"] = user.role

    if req.is_active is not None:
        user.is_active = req.is_active
        changes["is_active"] = user.is_active

    if req.password is not None:
        user.hashed_password = hash_password(req.password)
        changes["password_reset"] = True

    session.add(user)
    session.commit()
    session.refresh(user)

    client_ip = extract_client_ip(request)
    log_audit_event(
        session=session,
        username=admin_user.username,
        user_role=admin_user.role,
        action="user_updated",
        target_type="user",
        target_id=user.id,
        ip_address=client_ip,
        details=changes,
        user_id=admin_user.id,
    )

    return UserResponse(
        id=user.id,
        username=user.username,
        full_name=user.full_name,
        role=user.role,
        is_active=user.is_active,
        created_at=user.created_at,
        last_login_at=user.last_login_at,
    )


@router.delete("/{user_id}")
def delete_user(
    user_id: str,
    request: Request,
    session: Session = Depends(get_session),
    admin_user: User = Depends(require_admin),
):
    """Deletes a user account (Admin only)."""
    user = session.get(User, user_id)
    if not user:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found.")

    if user.id == admin_user.id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Administrators cannot delete their own active account.",
        )

    if user.role == "admin":
        active_admin_count = session.exec(
            select(func.count(User.id)).where(User.role == "admin", User.is_active == True)
        ).one()
        if active_admin_count <= 1:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Cannot delete the last remaining Administrator account.",
            )

    username_deleted = user.username
    session.delete(user)
    session.commit()

    client_ip = extract_client_ip(request)
    log_audit_event(
        session=session,
        username=admin_user.username,
        user_role=admin_user.role,
        action="user_deleted",
        target_type="user",
        target_id=user_id,
        ip_address=client_ip,
        details={"deleted_username": username_deleted},
        user_id=admin_user.id,
    )

    return {"status": "ok", "message": f"User {username_deleted} successfully deleted."}
