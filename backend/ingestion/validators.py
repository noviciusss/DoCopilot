import hashlib
from typing import Tuple, Optional
from sqlalchemy.ext.asyncio import AsyncSession
from fastapi import HTTPException, status
from backend.db import crud
from backend.db.models import Document

ALLOWED_MIME_TYPES = {
    "application/pdf": ".pdf",
    "text/plain": ".txt",
}

def calculate_checksum(content: bytes) -> str:
    """Computes SHA-256 checksum of raw file bytes."""
    return hashlib.sha256(content).hexdigest()

def validate_file_size(content_length: int, max_mb: int = 40) -> None:
    """Enforces upper bounds on document upload sizes."""
    max_bytes = max_mb * 1024 * 1024
    if content_length > max_bytes:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File size exceeds limit of {max_mb}MB."
        )

async def check_idempotency(
    db: AsyncSession,
    tenant_id: str,
    checksum: str
) -> Optional[Document]:
    """
    Checks if a file with the identical SHA-256 checksum already exists for this tenant.
    Returns existing Document if present, else None.
    """
    import uuid
    tenant_uuid = uuid.UUID(tenant_id)
    return await crud.get_document_by_checksum(db, tenant_uuid, checksum)
