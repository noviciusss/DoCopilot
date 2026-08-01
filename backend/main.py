import os
import uuid
import logging
import hashlib
from typing import Optional, Any

from fastapi import FastAPI, HTTPException, Request, UploadFile, File, Form, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi import BackgroundTasks
from pydantic import BaseModel
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import text

from backend.db.session import get_db, engine, Base
from backend.auth.router import router as auth_router
from backend.auth.dependencies import get_tenant_context, TenantContext
from backend.rag import (
    query_document,
    stream_answer,
    index_get_pdf,
    index_get_txt,
    index_get_plain_text,
    _get_qdrant_client
)
from backend.ragguardrails import RagGuardrails
from backend.db import crud
from backend.ingestion.validators import calculate_checksum, validate_file_size, check_idempotency
from backend.ingestion.tasks import _process_ingestion, process_document_task
from backend.db.models import IngestionJob

limiter = Limiter(key_func=get_remote_address)




app = FastAPI(
    title="DoCopilot API",
    description="Authenticated Enterprise RAG Platform: Hybrid search, reranking, tenant isolation.",
    version="2.1.0",
)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

logger = logging.getLogger(__name__)

# Include Auth Router
app.include_router(auth_router)

# Database table creation on startup for dev (or use alembic upgrade head in production)
@app.on_event("startup")
async def startup_event():
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    logger.info("Database models verified / initialized.")

# CORS Setup
raw_origins = os.getenv("ALLOWED_ORIGINS", "*").strip()
cors_origins = ["*"] if raw_origins == "*" else [o.strip() for o in raw_origins.split(",") if o.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=(raw_origins != "*"),
    allow_methods=["*"],
    allow_headers=["*"],
)

# Models
class UploadResponse(BaseModel):
    document_id: str
    filename: str


class ChatRequest(BaseModel):
    question: str
    document_id: Optional[str] = None

class ChatResponse(BaseModel):
    answer: str
    sources: list[str]
    blocked: bool = False

class UploadAsyncResponse(BaseModel):
    job_id: str
    document_id: str
    filename: str
    status: str
    message: str
class IngestionJobStatusResponse(BaseModel):
    job_id: str
    document_id: str
    status: str
    retry_count: int
    failure_reason: Optional[str] = None
    started_at: Optional[str] = None
    completed_at: Optional[str] = None

def _coerce_upload(value: Any) -> UploadFile | None:
    if value is None:
        return None
    if isinstance(value, UploadFile):
        return value
    if hasattr(value, "filename") and hasattr(value, "file"):
        return value
    return None

# Endpoints

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

@app.get("/readyz")
async def readiness_check(db: AsyncSession = Depends(get_db)):
    """Readiness probe checking Postgres DB and Qdrant readiness."""
    db_status = "ok"
    qdrant_status = "ok"
    try:
        await db.execute(text("SELECT 1"))
    except Exception as e:
        logger.error("Database readiness failure: %s", e)
        db_status = f"unhealthy: {str(e)}"

    try:
        q_client = _get_qdrant_client()
        q_client.get_collections()
    except Exception as e:
        logger.error("Qdrant readiness failure: %s", e)
        qdrant_status = f"unhealthy: {str(e)}"

    is_ready = db_status == "ok" and qdrant_status == "ok"
    status_code = status.HTTP_200_OK if is_ready else status.HTTP_503_SERVICE_UNAVAILABLE
    
    return {
        "status": "ready" if is_ready else "not_ready",
        "postgres": db_status,
        "qdrant": qdrant_status
    }

@app.post("/upload", response_model=UploadAsyncResponse, status_code=status.HTTP_202_ACCEPTED)
async def upload_document_async(
    background_tasks: BackgroundTasks,
    pdf_file: UploadFile | None = File(default=None),
    txt_file: UploadFile | None = File(default=None),
    plain_text: str | None = Form(default=None),
    context: TenantContext = Depends(get_tenant_context),
    db: AsyncSession = Depends(get_db)
) -> UploadAsyncResponse:
    """
    Asynchronous Upload Endpoint:
    1. Validates size & calculates SHA-256 checksum.
    2. Checks idempotency (if duplicate exists, returns immediately).
    3. Saves Document and IngestionJob (status='queued') to Postgres.
    4. Dispatches background processing task (Celery or BackgroundTasks).
    5. Returns 202 Accepted immediately.
    """
    tenant_str = str(context.tenant_id)
    pdf_upload = _coerce_upload(pdf_file)
    txt_upload = _coerce_upload(txt_file)
    max_mb = int(os.getenv("MAX_UPLOAD_SIZE_MB", "20"))
    
    if pdf_upload is not None:
        contents = await pdf_upload.read()
        validate_file_size(len(contents), max_mb=max_mb)
        filename = pdf_upload.filename or "document.pdf"
        is_txt, is_plain = False, False
    elif txt_upload is not None:
        contents = await txt_upload.read()
        validate_file_size(len(contents), max_mb=max_mb)
        filename = txt_upload.filename or "document.txt"
        is_txt, is_plain = True, False
    elif plain_text is not None and plain_text.strip():
        contents = plain_text.encode("utf-8")
        validate_file_size(len(contents), max_mb=max_mb)
        filename = "plain_text.txt"
        is_txt, is_plain = False, True
    else:
        raise HTTPException(status_code=400, detail="No file or text provided")
    # SHA-256 Checksum Idempotency Guard
    checksum = calculate_checksum(contents)
    existing_doc = await check_idempotency(db, tenant_str, checksum)
    if existing_doc:
        return UploadAsyncResponse(
            job_id="existing",
            document_id=str(existing_doc.id),
            filename=existing_doc.filename,
            status="succeeded",
            message="Document already indexed (idempotency match)."
        )
    # Save Document record to Postgres
    doc = await crud.create_document(
        db,
        tenant_id=context.tenant_id,
        filename=filename,
        checksum=checksum,
        file_size_bytes=len(contents),
        created_by_id=context.user.id
    )
    # Save IngestionJob record (QUEUED) to Postgres
    job = await crud.create_ingestion_job(db, document_id=doc.id)
    await db.commit()
    # Dispatch: Celery if broker URL is explicitly set and non-empty, else use FastAPI BackgroundTasks
    celery_broker = os.getenv("CELERY_BROKER_URL", "").strip()
    use_celery = bool(celery_broker)
    if use_celery:
        try:
            process_document_task.delay(
                str(job.id),
                str(doc.id),
                contents.hex(),
                filename,
                tenant_str,
                is_txt,
                is_plain
            )
        except Exception as exc:
            logger.warning("Celery dispatch failed (%s), falling back to BackgroundTasks.", exc)
            background_tasks.add_task(_process_ingestion, str(job.id), str(doc.id), contents, filename, tenant_str, is_txt, is_plain)
    else:
        background_tasks.add_task(_process_ingestion, str(job.id), str(doc.id), contents, filename, tenant_str, is_txt, is_plain)
    return UploadAsyncResponse(
        job_id=str(job.id),
        document_id=str(doc.id),
        filename=filename,
        status="queued",
        message="Document upload accepted. Processing in background."
    )
@app.get("/ingestion/jobs/{job_id}", response_model=IngestionJobStatusResponse)
async def get_job_status(
    job_id: str,
    context: TenantContext = Depends(get_tenant_context),
    db: AsyncSession = Depends(get_db)
) -> IngestionJobStatusResponse:
    """Poll status of an ingestion job."""
    try:
        job_uuid = uuid.UUID(job_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid job UUID format.")
    job = await crud.get_ingestion_job(db, job_uuid)
    if not job:
        raise HTTPException(status_code=404, detail="Ingestion job not found.")
    return IngestionJobStatusResponse(
        job_id=str(job.id),
        document_id=str(job.document_id),
        status=job.status,
        retry_count=job.retry_count,
        failure_reason=job.failure_reason,
        started_at=job.started_at.isoformat() if job.started_at else None,
        completed_at=job.completed_at.isoformat() if job.completed_at else None
    )
@app.post("/chat", response_model=ChatResponse)
@limiter.limit("20/minute")
async def chat(
    request: Request,
    body: ChatRequest,
    context: TenantContext = Depends(get_tenant_context)
) -> ChatResponse:
    """Non-streaming query protected by verified JWT tenant scope."""
    tenant_str = str(context.tenant_id)
    try:
        result = await query_document(
            document_id=body.document_id,
            question=body.question,
            tenant_id=tenant_str,
        )
        return ChatResponse(
            answer=result["answer"],
            sources=result.get("sources", []),
            blocked=result.get("blocked", False),
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

@app.post("/chat/stream")
@limiter.limit("20/minute")
async def chat_stream(
    request: Request,
    body: ChatRequest,
    context: TenantContext = Depends(get_tenant_context)
) -> StreamingResponse:
    """Streaming SSE query protected by verified JWT tenant scope."""
    tenant_str = str(context.tenant_id)
    is_safe, message = RagGuardrails.check_input(body.question)
    if not is_safe:
        import json
        async def _blocked():
            yield f"data: {json.dumps({'token': message})}\n\n"
            yield f"data: {json.dumps({'done': True, 'sources': [], 'blocked': True, 'answer': message})}\n\n"

        return StreamingResponse(
            _blocked(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    return StreamingResponse(
        stream_answer(body.question, document_id=body.document_id, tenant_id=tenant_str),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
