import uuid
from typing import Optional, List
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, desc
from sqlalchemy.orm import selectinload
from backend.db.models import User, Tenant, TenantMembership, Document, DocumentVersion, IngestionJob

# User CRUD
async def get_user_by_email(db: AsyncSession, email: str) -> Optional[User]:
    result = await db.execute(select(User).where(User.email == email.lower().strip()))
    return result.scalar_one_or_none()

async def get_user_by_id(db: AsyncSession, user_id: uuid.UUID) -> Optional[User]:
    result = await db.execute(select(User).where(User.id == user_id))
    return result.scalar_one_or_none()

async def create_user(db: AsyncSession, email: str, hashed_password: str, full_name: Optional[str] = None) -> User:
    user = User(email=email.lower().strip(), hashed_password=hashed_password, full_name=full_name)
    db.add(user)
    await db.flush()
    return user

# Tenant CRUD
async def create_tenant(db: AsyncSession, name: str, slug: str) -> Tenant:
    tenant = Tenant(name=name, slug=slug.lower().strip())
    db.add(tenant)
    await db.flush()
    return tenant

async def get_tenant_by_slug(db: AsyncSession, slug: str) -> Optional[Tenant]:
    result = await db.execute(select(Tenant).where(Tenant.slug == slug.lower().strip()))
    return result.scalar_one_or_none()

async def add_tenant_member(db: AsyncSession, user_id: uuid.UUID, tenant_id: uuid.UUID, role: str = "member") -> TenantMembership:
    membership = TenantMembership(user_id=user_id, tenant_id=tenant_id, role=role)
    db.add(membership)
    await db.flush()
    return membership

async def get_membership(db: AsyncSession, user_id: uuid.UUID, tenant_id: uuid.UUID) -> Optional[TenantMembership]:
    result = await db.execute(
        select(TenantMembership).where(
            TenantMembership.user_id == user_id,
            TenantMembership.tenant_id == tenant_id
        )
    )
    return result.scalar_one_or_none()

async def get_user_memberships(db: AsyncSession, user_id: uuid.UUID) -> List[TenantMembership]:
    result = await db.execute(select(TenantMembership).where(TenantMembership.user_id == user_id))
    return list(result.scalars().all())

# Document CRUD
async def get_document_by_checksum(db: AsyncSession, tenant_id: uuid.UUID, checksum: str) -> Optional[Document]:
    result = await db.execute(
        select(Document).where(
            Document.tenant_id == tenant_id,
            Document.checksum == checksum,
            Document.is_deleted == False
        )
    )
    return result.scalar_one_or_none()

async def create_document(
    db: AsyncSession,
    tenant_id: uuid.UUID,
    filename: str,
    checksum: str,
    file_size_bytes: int,
    created_by_id: Optional[uuid.UUID] = None,
    mime_type: str = "application/pdf"
) -> Document:
    doc = Document(
        tenant_id=tenant_id,
        filename=filename,
        checksum=checksum,
        file_size_bytes=file_size_bytes,
        created_by_id=created_by_id,
        mime_type=mime_type
    )
    db.add(doc)
    await db.flush()
    return doc

async def get_tenant_documents(
    db: AsyncSession,
    tenant_id: uuid.UUID,
    created_by_id: Optional[uuid.UUID] = None,
    limit: int = 100,
) -> List[Document]:
    """
    Fetch non-deleted documents for a tenant, ordered newest first.
    When created_by_id is provided, only return that user's documents.
    Eagerly loads versions and ingestion_jobs so callers can access them
    without extra DB round trips.
    """
    q = (
        select(Document)
        .options(
            selectinload(Document.versions),
            selectinload(Document.ingestion_jobs),
        )
        .where(
            Document.tenant_id == tenant_id,
            Document.is_deleted == False,
        )
        .order_by(desc(Document.created_at))
        .limit(limit)
    )
    if created_by_id is not None:
        q = q.where(Document.created_by_id == created_by_id)
    result = await db.execute(q)
    return list(result.scalars().all())

async def soft_delete_document(db: AsyncSession, doc_id: uuid.UUID, tenant_id: uuid.UUID) -> bool:
    """Soft-delete a document (marks is_deleted=True). Returns True if found & updated."""
    result = await db.execute(
        select(Document).where(
            Document.id == doc_id,
            Document.tenant_id == tenant_id,
            Document.is_deleted == False,
        )
    )
    doc = result.scalar_one_or_none()
    if not doc:
        return False
    doc.is_deleted = True
    await db.flush()
    return True

# Ingestion Job CRUD
async def create_ingestion_job(db: AsyncSession, document_id: uuid.UUID) -> IngestionJob:
    job = IngestionJob(document_id=document_id, status="queued")
    db.add(job)
    await db.flush()
    return job

async def get_ingestion_job(db: AsyncSession, job_id: uuid.UUID) -> Optional[IngestionJob]:
    result = await db.execute(select(IngestionJob).where(IngestionJob.id == job_id))
    return result.scalar_one_or_none()
