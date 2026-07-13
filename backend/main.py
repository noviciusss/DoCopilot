import os
import logging
from typing import Optional, Any

from fastapi import FastAPI, HTTPException, Request, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

from .rag import (
    query_document,
    stream_answer,
    index_get_pdf,
    index_get_txt,
    index_get_plain_text,
)
from .ragguardrails import RagGuardrails

# ---------------------------------------------------------------------------
# App + rate limiter setup
# ---------------------------------------------------------------------------

limiter = Limiter(key_func=get_remote_address)

app = FastAPI(
    title="DoCopilot API",
    description="RAG backend: upload documents, ask questions with streaming support.",
    version="2.0.0",
)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# CORS
# ---------------------------------------------------------------------------

raw_origins = os.getenv("ALLOWED_ORIGINS", "*").strip()
if raw_origins == "*":
    cors_origins = ["*"]
    cors_allow_credentials = False
else:
    cors_origins = [o for o in (item.strip() for item in raw_origins.split(",")) if o]
    cors_allow_credentials = True
    if not cors_origins:
        raise RuntimeError("ALLOWED_ORIGINS must list at least one origin or be '*'")

app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=cors_allow_credentials,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

class UploadResponse(BaseModel):
    document_id: str


class ChatRequest(BaseModel):
    question: str
    document_id: Optional[str] = None
    tenant_id: str = "default"


class ChatResponse(BaseModel):
    answer: str
    sources: list[str]
    blocked: bool = False


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _coerce_upload(value: Any) -> UploadFile | None:
    if value is None:
        return None
    if isinstance(value, UploadFile):
        return value
    if hasattr(value, "filename") and hasattr(value, "file"):
        return value
    return None


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
async def health_check():
    return {"status": "healthy"}


@app.post("/upload", response_model=UploadResponse)
async def upload_document(
    pdf_file: UploadFile | None = File(default=None),
    txt_file: UploadFile | None = File(default=None),
    plain_text: str | None = Form(default=None),
    tenant_id: str = Form(default="default"),
) -> UploadResponse:
    try:
        pdf_upload = _coerce_upload(pdf_file)
        txt_upload = _coerce_upload(txt_file)

        max_mb = int(os.getenv("MAX_UPLOAD_SIZE_MB", "20"))
        max_bytes = max_mb * 1024 * 1024

        if pdf_upload is not None:
            logger.info("Processing PDF upload: %s for tenant: %s", pdf_upload.filename, tenant_id)
            contents = await pdf_upload.read()
            if len(contents) > max_bytes:
                raise HTTPException(status_code=413, detail=f"File too large. Max {max_mb}MB.")
            doc_id = index_get_pdf(contents, pdf_upload.filename or "document.pdf", tenant_id=tenant_id)
            return UploadResponse(document_id=doc_id)

        if txt_upload is not None:
            logger.info("Processing TXT upload: %s for tenant: %s", txt_upload.filename, tenant_id)
            contents = await txt_upload.read()
            if len(contents) > max_bytes:
                raise HTTPException(status_code=413, detail=f"File too large. Max {max_mb}MB.")
            doc_id = index_get_txt(contents.decode("utf-8"), txt_upload.filename or "document.txt", tenant_id=tenant_id)
            return UploadResponse(document_id=doc_id)

        if plain_text is not None and plain_text.strip():
            logger.info("Processing plain text upload for tenant: %s", tenant_id)
            if len(plain_text.encode("utf-8")) > max_bytes:
                raise HTTPException(status_code=413, detail=f"Text too long. Max {max_mb}MB.")
            doc_id = index_get_plain_text(plain_text, tenant_id=tenant_id)
            return UploadResponse(document_id=doc_id)

        raise HTTPException(status_code=400, detail="No file or text provided")

    except HTTPException:
        raise
    except ValueError as exc:
        logger.exception("Upload validation error: %s", exc)
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Upload error: %s", exc)
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(exc)}") from exc


@app.post("/chat", response_model=ChatResponse)
@limiter.limit("10/minute")
async def chat(request: Request, body: ChatRequest) -> ChatResponse:
    """Non-streaming chat endpoint (returns full answer at once)."""
    try:
        result = await query_document(
            document_id=body.document_id,
            question=body.question,
            tenant_id=body.tenant_id,
        )
        return ChatResponse(
            answer=result["answer"],
            sources=result.get("sources", []),
            blocked=result.get("blocked", False),
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Chat error: %s", exc)
        raise HTTPException(status_code=500, detail="Unexpected error during chat.") from exc


@app.post("/chat/stream")
@limiter.limit("10/minute")
async def chat_stream(request: Request, body: ChatRequest) -> StreamingResponse:
    """
    Streaming chat endpoint - returns Server-Sent Events (SSE).

    Event format:
      Token chunk : data: {"token": "..."}
      Final event : data: {"done": true, "sources": [...], "answer": "<guardrail-cleaned>"}
      Error event : data: {"error": "..."}
    """
    import json

    is_safe, message = RagGuardrails.check_input(body.question)
    if not is_safe:
        async def _blocked():
            yield f"data: {json.dumps({'token': message})}\n\n"
            yield f"data: {json.dumps({'done': True, 'sources': [], 'blocked': True, 'answer': message})}\n\n"

        return StreamingResponse(
            _blocked(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    return StreamingResponse(
        stream_answer(body.question, document_id=body.document_id, tenant_id=body.tenant_id),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
