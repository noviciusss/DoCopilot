import uuid
import logging
from datetime import datetime, timezone
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from backend.db.session import AsyncSessionLocal
from backend.db.models import IngestionJob, Document, DocumentVersion
from backend.rag import index_get_pdf, index_get_txt, index_get_plain_text
from backend.ingestion.worker import celery_app

logger = logging.getLogger(__name__)

async def _process_ingestion(job_id_str: str, document_id_str: str, file_bytes: bytes, filename: str, tenant_id_str: str, is_txt: bool = False, is_plain: bool = False):
    """Core async runner updating state machine in Postgres."""
    job_uuid = uuid.UUID(job_id_str)
    doc_uuid = uuid.UUID(document_id_str)

    async with AsyncSessionLocal() as db:
        # 1. Update job state -> RUNNING
        result = await db.execute(select(IngestionJob).where(IngestionJob.id == job_uuid))
        job = result.scalar_one_or_none()
        if not job:
            logger.error("Job %s not found in Postgres.", job_id_str)
            return

        job.status = "running"
        job.started_at = datetime.now(timezone.utc)
        await db.commit()

        try:
            # 2. Execute document indexing into Qdrant
            if is_txt:
                text_content = file_bytes.decode("utf-8")
                doc_collection_id = index_get_txt(text_content, filename, tenant_id=tenant_id_str, document_id=document_id_str)
            elif is_plain:
                text_content = file_bytes.decode("utf-8")
                doc_collection_id = index_get_plain_text(text_content, tenant_id=tenant_id_str, document_id=document_id_str)
            else:
                doc_collection_id = index_get_pdf(file_bytes, filename, tenant_id=tenant_id_str, document_id=document_id_str)


            # 3. Create DocumentVersion record
            doc_version = DocumentVersion(
                document_id=doc_uuid,
                version_number=1,
                qdrant_collection=doc_collection_id,
                is_active=True
            )
            db.add(doc_version)

            # 4. Update job state -> SUCCEEDED
            job.status = "succeeded"
            job.completed_at = datetime.now(timezone.utc)
            await db.commit()
            logger.info("Ingestion job %s completed successfully.", job_id_str)

        except Exception as exc:
            await db.rollback()
            logger.exception("Ingestion job %s failed: %s", job_id_str, exc)
            
            # Refetch job to update failure state
            res_fail = await db.execute(select(IngestionJob).where(IngestionJob.id == job_uuid))
            fail_job = res_fail.scalar_one_or_none()
            if fail_job:
                fail_job.status = "failed"
                fail_job.retry_count += 1
                fail_job.failure_reason = str(exc)
                fail_job.completed_at = datetime.now(timezone.utc)
                await db.commit()

# Celery Wrapper Task
@celery_app.task(bind=True, max_retries=3, default_retry_delay=10)
def process_document_task(self, job_id_str: str, document_id_str: str, file_bytes_hex: str, filename: str, tenant_id_str: str, is_txt: bool = False, is_plain: bool = False):
    import asyncio
    file_bytes = bytes.fromhex(file_bytes_hex)
    try:
        asyncio.run(_process_ingestion(job_id_str, document_id_str, file_bytes, filename, tenant_id_str, is_txt, is_plain))
    except Exception as exc:
        raise self.retry(exc=exc)
