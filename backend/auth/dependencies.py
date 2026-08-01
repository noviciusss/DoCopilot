import uuid
from dataclasses import dataclass
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from sqlalchemy.ext.asyncio import AsyncSession

from backend.db.session import get_db
from backend.db.models import User
from backend.db import crud
from backend.auth.security import decode_access_token

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/login")

@dataclass
class TenantContext:
    user: User
    tenant_id: uuid.UUID
    tenant_slug: str
    role: str

async def get_current_user(
    token: str = Depends(oauth2_scheme),
    db: AsyncSession = Depends(get_db)
) -> User:
    """Dependency that extracts user from JWT Bearer token."""
    payload = decode_access_token(token)
    user_id_str = payload.get("sub")
    try:
        user_id = uuid.UUID(user_id_str)
    except ValueError:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid User UUID in token")
        
    user = await crud.get_user_by_id(db, user_id)
    if not user:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="User not found")
    return user

async def get_tenant_context(
    token: str = Depends(oauth2_scheme),
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
) -> TenantContext:
    """
    Core security dependency: Verifies user membership in tenant derived strictly from token.
    Prevents cross-tenant access attacks.
    """
    payload = decode_access_token(token)
    tenant_id_str = payload.get("tenant_id")
    role = payload.get("role", "member")
    
    try:
        tenant_id = uuid.UUID(tenant_id_str)
    except ValueError:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid Tenant UUID in token")

    membership = await crud.get_membership(db, user_id=current_user.id, tenant_id=tenant_id)
    if not membership:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access forbidden: User does not belong to the verified tenant"
        )

    return TenantContext(
        user=current_user,
        tenant_id=tenant_id,
        tenant_slug=str(tenant_id),
        role=membership.role
    )
