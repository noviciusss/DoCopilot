import re
from typing import Optional
from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, EmailStr
from sqlalchemy.ext.asyncio import AsyncSession

from backend.db.session import get_db, engine, Base
from backend.db import crud
from backend.db.models import User
from backend.auth.security import hash_password, verify_password, create_access_token
from backend.auth.dependencies import get_current_user, get_tenant_context, TenantContext

router = APIRouter(prefix="/auth", tags=["Authentication"])

class RegisterRequest(BaseModel):
    email: EmailStr
    password: str
    full_name: Optional[str] = None
    tenant_name: Optional[str] = None

class LoginRequest(BaseModel):
    email: EmailStr
    password: str

class AuthTokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user_id: str
    email: str
    tenant_id: str

class UserProfileResponse(BaseModel):
    id: str
    email: str
    full_name: Optional[str]
    tenant_id: str
    role: str

@router.post("/register", response_model=AuthTokenResponse, status_code=status.HTTP_201_CREATED)
async def register(body: RegisterRequest, db: AsyncSession = Depends(get_db)):
    """Registers new user and automatically provisions a primary default workspace/tenant."""
    if len(body.password) < 6:
        raise HTTPException(status_code=400, detail="Password must be at least 6 characters")

    existing_user = await crud.get_user_by_email(db, body.email)

    if existing_user:
        raise HTTPException(status_code=400, detail="Email already registered")

    # 1. Create User
    hashed_pw = hash_password(body.password)
    user = await crud.create_user(db, email=body.email, hashed_password=hashed_pw, full_name=body.full_name)

    # 2. Provision Tenant / Workspace
    t_name = body.tenant_name or f"{user.email.split('@')[0]}'s Workspace"
    clean_slug = re.sub(r'[^a-z0-9]', '-', t_name.lower()).strip('-') + f"-{str(user.id)[:6]}"
    tenant = await crud.create_tenant(db, name=t_name, slug=clean_slug)

    # 3. Create Admin Membership
    membership = await crud.add_tenant_member(db, user_id=user.id, tenant_id=tenant.id, role="admin")

    # 4. Generate JWT Token
    token = create_access_token(user_id=user.id, tenant_id=tenant.id, role=membership.role)

    return AuthTokenResponse(
        access_token=token,
        token_type="bearer",
        user_id=str(user.id),
        email=user.email,
        tenant_id=str(tenant.id)
    )


@router.post("/login", response_model=AuthTokenResponse)
async def login(body: LoginRequest, db: AsyncSession = Depends(get_db)):
    """Authenticates credentials and returns a JWT access token with tenant context."""
    user = await crud.get_user_by_email(db, body.email)
    if not user or not verify_password(body.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # Get user memberships
    memberships = await crud.get_user_memberships(db, user.id)
    if not memberships:
        raise HTTPException(status_code=400, detail="User has no tenant membership")

    primary_membership = memberships[0]
    token = create_access_token(
        user_id=user.id,
        tenant_id=primary_membership.tenant_id,
        role=primary_membership.role
    )

    return AuthTokenResponse(
        access_token=token,
        token_type="bearer",
        user_id=str(user.id),
        email=user.email,
        tenant_id=str(primary_membership.tenant_id)
    )


@router.get("/me", response_model=UserProfileResponse)
async def get_me(context: TenantContext = Depends(get_tenant_context)):
    """Returns profile information for the verified authenticated user and tenant scope."""
    return UserProfileResponse(
        id=str(context.user.id),
        email=context.user.email,
        full_name=context.user.full_name,
        tenant_id=str(context.tenant_id),
        role=context.role
    )
