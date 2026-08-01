import os
import uuid
import bcrypt
from datetime import datetime, timedelta, timezone
from typing import Dict, Any
from jose import jwt, JWTError
from fastapi import HTTPException, status

# Max bytes bcrypt can handle is 72 — truncate before hashing
_BCRYPT_MAX_BYTES = 72

JWT_SECRET_KEY = os.getenv("JWT_SECRET_KEY", "super_secret_dev_key_change_in_production_12345")
JWT_ALGORITHM = os.getenv("JWT_ALGORITHM", "HS256")
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "1440"))  # 24 hours default

def hash_password(password: str) -> str:
    """Hash plain password using bcrypt directly (bypasses passlib incompatibility)."""
    pw_bytes = password.encode("utf-8")[:_BCRYPT_MAX_BYTES]
    salt = bcrypt.gensalt()
    return bcrypt.hashpw(pw_bytes, salt).decode("utf-8")

def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a plain password against a stored bcrypt hash."""
    pw_bytes = plain_password.encode("utf-8")[:_BCRYPT_MAX_BYTES]
    hash_bytes = hashed_password.encode("utf-8")
    return bcrypt.checkpw(pw_bytes, hash_bytes)

def create_access_token(user_id: uuid.UUID, tenant_id: uuid.UUID, role: str = "member") -> str:
    """Create a signed JWT token containing user_id, tenant_id, role, and expiration."""
    expire = datetime.now(timezone.utc) + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode: Dict[str, Any] = {
        "sub": str(user_id),
        "tenant_id": str(tenant_id),
        "role": role,
        "exp": int(expire.timestamp())
    }
    encoded_jwt = jwt.encode(to_encode, JWT_SECRET_KEY, algorithm=JWT_ALGORITHM)
    return encoded_jwt

def decode_access_token(token: str) -> Dict[str, Any]:
    """Decode and validate a JWT access token."""
    try:
        payload = jwt.decode(token, JWT_SECRET_KEY, algorithms=[JWT_ALGORITHM])
        user_id: str = payload.get("sub")
        tenant_id: str = payload.get("tenant_id")
        if not user_id or not tenant_id:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token payload credentials",
                headers={"WWW-Authenticate": "Bearer"},
            )
        return payload
    except JWTError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate token or token has expired",
            headers={"WWW-Authenticate": "Bearer"},
        )
