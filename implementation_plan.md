# DoCopilot — Implementation Plan
> Based on `project_update_masterplan.md` · Target: Flagship Applied-AI / RAG Platform

## Background & Goal

DoCopilot currently works as a solid RAG demo: PDF/TXT ingestion → Qdrant hybrid retrieval (dense + sparse BM25 + RRF) → Cohere rerank → Groq/Qwen streaming answer with LangSmith traces. The codebase has one FastAPI file, one RAG module, basic guardrails, rate limiting, and Docker compose.

**What is missing (credibility gaps per masterplan):**
- No real persistent storage — Qdrant holds payload but not user/doc ownership or job state
- Tenant ID comes from request body — not from verified identity (trust issue)
- Ingestion blocks the HTTP request — no async job state machine
- Only runs locally — no cloud deployment or CI/CD
- Evaluation is one metric (`correctness`, `relevance` via word-overlap) on 40 Qs
- No structured observability — traces exist via LangSmith but no latency percentiles, cost tracking, or job queue visibility

**End-state pitch after all phases:**
> "Authenticated, cloud-deployed document Q&A with hybrid retrieval, reranking, async ingestion, tenant-safe access control, and versioned stage-aware evaluation."

---

## Scope Boundary (anti-gravity rules)

> [!CAUTION]
> Do NOT add: GraphRAG, HyDE, fine-tuned embeddings, multi-agent RAG, Kubernetes, admin panel. These add no placement value and steal time from DSA/OA/core CS.

---

## Phase Overview

| Phase | Theme | Est. Effort | Deliverable |
|---|---|---|---|
| 0 | Foundation & Cleanup | 1–2 days | Repo consistency, env cleanup |
| 1 | PostgreSQL + Schema | 2–3 days | All 7 tables, Alembic migrations |
| 2 | JWT Auth + Tenant Authorization | 2–3 days | Verified identity, cross-tenant tests |
| 3 | Async Ingestion Pipeline | 3–4 days | Job state machine, worker, idempotency |
| 4 | Cloud Deployment + CI/CD | 3–4 days | Azure Container Apps, Vercel, GitHub Actions |
| 5 | Stage-Aware Evaluation | 2–3 days | 60–70 Q benchmark, 7 metric types |
| 6 | Observability | 1–2 days | Structured logs, latency percentiles, cost tracking |

---

## Open Questions

> [!IMPORTANT]
> **Q1 — Auth provider**: Should users register/login via local JWT (email+password stored in Postgres), or do you want OAuth (Google sign-in)? Local JWT is simpler and sufficient for placement demos.

> [!IMPORTANT]
> **Q2 — Async worker strategy**: For Phase 3, the simplest approach is a FastAPI `BackgroundTask` for in-process async. A more production-like approach uses Celery + Redis (already in masterplan). Which do you prefer? Celery adds Docker complexity but is the stronger placement story.

> [!IMPORTANT]
> **Q3 — Cloud provider**: Masterplan specifies Azure (Azure for Students credit). Do you have an active Azure student subscription? If not, we should fall back to Railway or Render (both have free tiers that work fine for the demo story).

> [!WARNING]
> **Q4 — Public demo corpus**: Masterplan says "create a sanitized public corpus for demos." The current repo uses `aws-overview.pdf` which is a public document — this is fine. Do you want to add 2–3 more public domain documents to the demo set, or keep it single-document?

---

## Proposed Changes

---

### Phase 0 — Foundation & Cleanup (prerequisite)

**Goal:** Clean repo state before adding any new systems.

#### [MODIFY] [.env.example](file:///d:/MadRocket/DoCopilot/.env.example)
- Add `DATABASE_URL`, `JWT_SECRET_KEY`, `JWT_ALGORITHM=HS256`, `CELERY_BROKER_URL`, `AZURE_STORAGE_*` placeholders
- Document each variable with inline comments

#### [MODIFY] [README.md](file:///d:/MadRocket/DoCopilot/README.md)
- Verify metric numbers match `evaluation_results.json` — ensure no numeric discrepancies
- Add architecture diagram section (text-based or Mermaid)
- Update setup section to include Postgres steps

#### [MODIFY] [docker-compose.yml](file:///d:/MadRocket/DoCopilot/docker-compose.yml)
- Add `postgres` service (image: `postgres:16-alpine`)
- Add `redis` service for future Celery use (Phase 3)
- Add named volumes for Postgres data persistence
- Add healthchecks for all services

#### [MODIFY] [backend/requirements.txt](file:///d:/MadRocket/DoCopilot/backend/requirements.txt)
- Add: `sqlalchemy>=2.0`, `alembic`, `psycopg2-binary`, `python-jose[cryptography]`, `passlib[bcrypt]`, `celery`, `redis`

**Tests (Phase 0):**
- Run `pytest tests/` to confirm existing eval tests still pass after env changes
- Manual: `docker compose up` — all 4 services (backend, frontend, qdrant, postgres) healthy

---

### Phase 1 — PostgreSQL as Metadata Source of Truth

**Goal:** Add relational backbone. Qdrant stays for retrieval only. Postgres owns identity, documents, versions, and job state.

#### Schema Design

```sql
-- users
id UUID PK, email TEXT UNIQUE, hashed_password TEXT, created_at TIMESTAMP

-- tenants  
id UUID PK, name TEXT, created_at TIMESTAMP

-- tenant_memberships
user_id UUID FK(users), tenant_id UUID FK(tenants), role TEXT ('member'|'admin'), PRIMARY KEY(user_id, tenant_id)

-- documents
id UUID PK, tenant_id UUID FK, filename TEXT, checksum TEXT, file_size_bytes INT, 
mime_type TEXT, created_by UUID FK(users), created_at TIMESTAMP, is_deleted BOOL DEFAULT FALSE

-- document_versions
id UUID PK, document_id UUID FK(documents), version_number INT, checksum TEXT, 
qdrant_collection TEXT, created_at TIMESTAMP, is_active BOOL DEFAULT TRUE

-- ingestion_jobs
id UUID PK, document_id UUID FK(documents), status TEXT ('queued'|'running'|'succeeded'|'failed'),
retry_count INT DEFAULT 0, failure_reason TEXT, started_at TIMESTAMP, completed_at TIMESTAMP

-- evaluation_runs
id UUID PK, run_name TEXT, dataset_version TEXT, total_cases INT, metrics JSONB, 
created_at TIMESTAMP, created_by UUID FK(users)
```

#### [NEW] `backend/db/` directory

**`backend/db/__init__.py`** — empty init

**`backend/db/models.py`** — SQLAlchemy ORM models for all 7 tables above

**`backend/db/session.py`** — async SQLAlchemy engine + `get_db()` dependency
```python
# Pattern: async engine with connection pool, yields session
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql+asyncpg://...")
engine = create_async_engine(DATABASE_URL, pool_size=10, max_overflow=20)
AsyncSessionLocal = async_sessionmaker(engine, expire_on_commit=False)

async def get_db() -> AsyncSession:
    async with AsyncSessionLocal() as session:
        yield session
```

**`backend/db/crud.py`** — typed CRUD functions:
- `create_user`, `get_user_by_email`
- `create_tenant`, `add_member_to_tenant`, `get_user_tenants`
- `create_document`, `get_document`, `soft_delete_document`
- `create_document_version`, `get_active_version`
- `create_ingestion_job`, `update_job_status`
- `create_evaluation_run`

#### [NEW] `alembic/` directory
- `alembic.ini` — points to `DATABASE_URL`
- `alembic/env.py` — imports all models so autogenerate works
- `alembic/versions/001_initial_schema.py` — first migration (all 7 tables)

#### [MODIFY] [docker-compose.yml](file:///d:/MadRocket/DoCopilot/docker-compose.yml)
- Backend `depends_on` postgres with condition `service_healthy`
- Add `command: alembic upgrade head && uvicorn ...` in backend service

#### [MODIFY] [backend/main.py](file:///d:/MadRocket/DoCopilot/backend/main.py)
- Add startup event to verify DB connection
- Add `GET /health` to include DB and Qdrant status (readiness check)

**Tests (Phase 1):**

```
tests/db/
  test_models.py         — create/read all 7 tables against a test Postgres (pytest-asyncio)
  test_crud.py           — CRUD operations, soft delete, version queries
  test_migrations.py     — alembic upgrade head + downgrade + re-upgrade
```

- Use `pytest-asyncio` + `asyncpg` with a real test Postgres (spin up via Docker in CI)
- Fixture: create fresh schema per test session using `alembic upgrade head`
- Teardown: `alembic downgrade base`

**Fallback (Phase 1):**
- If `DATABASE_URL` is not set, backend logs a warning and continues without Postgres (Qdrant-only mode)
- All new DB calls are wrapped with `try/except` and logged; existing RAG endpoints remain functional

---

### Phase 2 — JWT Auth + Real Tenant Authorization

**Goal:** Replace `tenant_id: str = Form(default="default")` (client trust) with verified JWT-derived tenant scope.

#### Auth Flow
```
POST /auth/register  →  create user in Postgres, return JWT
POST /auth/login     →  verify bcrypt hash, return JWT
JWT payload: { sub: user_id, tenant_id: tenant_id, role: "member"|"admin", exp: ... }
```

#### [NEW] `backend/auth/` directory

**`backend/auth/jwt.py`**
```python
SECRET_KEY = os.getenv("JWT_SECRET_KEY")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 24  # 24h

def create_access_token(data: dict) -> str: ...
def decode_token(token: str) -> dict: ...  # raises HTTPException 401 on bad/expired
```

**`backend/auth/dependencies.py`**
```python
async def get_current_user(token: str = Depends(oauth2_scheme), db=Depends(get_db)):
    payload = decode_token(token)
    user = await crud.get_user(db, payload["sub"])
    if not user:
        raise HTTPException(status_code=401)
    return user

async def get_tenant_context(current_user=Depends(get_current_user), db=Depends(get_db)):
    # Returns (user, tenant_id, role) — tenant_id comes from JWT, NOT request body
    membership = await crud.get_membership(db, current_user.id, payload["tenant_id"])
    if not membership:
        raise HTTPException(status_code=403, detail="Not a member of this tenant")
    return TenantContext(user=current_user, tenant_id=membership.tenant_id, role=membership.role)
```

**`backend/auth/router.py`** — `POST /auth/register`, `POST /auth/login`, `GET /auth/me`

#### [MODIFY] [backend/main.py](file:///d:/MadRocket/DoCopilot/backend/main.py)
- Include `auth_router` with prefix `/auth`
- Add `get_tenant_context` dependency to `/upload`, `/chat`, `/chat/stream`
- Remove `tenant_id` from request bodies — it is now injected from token

#### [MODIFY] [backend/rag.py](file:///d:/MadRocket/DoCopilot/backend/rag.py)
- `index_get_pdf` / `index_get_txt` / `index_get_plain_text` now receive `tenant_id` from `TenantContext`
- `query_document` / `stream_answer` receive `tenant_id` from `TenantContext`
- No logic change to retrieval — only the source of `tenant_id` changes

#### Frontend Auth (minimal)
- Add login/register page in Next.js (simple form, store JWT in `localStorage`)
- Pass `Authorization: Bearer <token>` header with every `/upload` and `/chat` request
- If 401, redirect to login

**Tests (Phase 2):**

```
tests/auth/
  test_register_login.py     — happy path register + login + JWT decode
  test_token_expiry.py       — expired token returns 401
  test_cross_tenant.py       — Tenant A token cannot query Tenant B documents (403)
  test_role_checks.py        — member cannot access admin-only routes
  test_invalid_tokens.py     — tampered token, missing token, wrong algorithm
```

Cross-tenant test pattern:
```python
# Register two tenants, upload doc to tenant_a, query with tenant_b token
# Assert: 403 Forbidden — not a member of tenant_a
```

**Fallback (Phase 2):**
- If `JWT_SECRET_KEY` is not set, auth middleware logs a warning and falls back to `tenant_id="default"` (dev mode only, never production)
- Gate this behind `AUTH_ENABLED=true` env variable

---

### Phase 3 — Async Ingestion Pipeline

**Goal:** Upload request returns immediately with a job ID. Background worker handles parse → chunk → embed → index.

#### Ingestion State Machine

```
QUEUED → RUNNING → SUCCEEDED
                 ↘ FAILED (retry_count < 3) → QUEUED
                 ↘ FAILED (retry_count >= 3) → terminal FAILED
```

#### [NEW] `backend/ingestion/` directory

**`backend/ingestion/tasks.py`** — Celery tasks (or BackgroundTasks fallback)
```python
@celery_app.task(bind=True, max_retries=3, default_retry_delay=30)
def ingest_document_task(self, job_id: str, document_id: str, file_bytes: b64, filename: str, tenant_id: str):
    # 1. Update job status → RUNNING
    # 2. Validate: file type, size, checksum for idempotency
    # 3. Parse (PDF or TXT)
    # 4. Chunk
    # 5. Embed + index to Qdrant
    # 6. Update document_versions in Postgres
    # 7. Update job status → SUCCEEDED
    # On exception: update job status → FAILED, set failure_reason, retry
```

**`backend/ingestion/worker.py`** — Celery app config
```python
celery_app = Celery(
    "docopilot",
    broker=os.getenv("CELERY_BROKER_URL", "redis://localhost:6379/0"),
    backend=os.getenv("CELERY_RESULT_BACKEND", "redis://localhost:6379/1"),
)
```

**`backend/ingestion/validators.py`**
- `validate_file_type(filename, content)` — checks MIME via python-magic
- `validate_file_size(content, max_mb)` — raises if over limit
- `validate_checksum_unique(checksum, tenant_id, db)` — idempotency check against Postgres

#### [MODIFY] [backend/main.py](file:///d:/MadRocket/DoCopilot/backend/main.py)

`POST /upload` changes:
```python
# Before: synchronous index_get_pdf(...) blocks until done
# After:
async def upload_document(...) -> UploadJobResponse:
    checksum = hashlib.sha256(content).hexdigest()
    # 1. Create document record in Postgres
    doc = await crud.create_document(db, tenant_id, filename, checksum)
    # 2. Create ingestion_job record (status=QUEUED)
    job = await crud.create_ingestion_job(db, doc.id)
    # 3. Dispatch to Celery (or BackgroundTasks)
    ingest_document_task.delay(str(job.id), str(doc.id), ...)
    # 4. Return immediately
    return UploadJobResponse(job_id=job.id, document_id=doc.id, status="queued")
```

**New endpoint:** `GET /ingestion/jobs/{job_id}` — returns job status, retry_count, failure_reason

#### [MODIFY] [docker-compose.yml](file:///d:/MadRocket/DoCopilot/docker-compose.yml)
- Add `celery_worker` service (same image as backend, command: `celery -A backend.ingestion.worker worker`)
- Add `redis` service (`redis:7-alpine`)

**Tests (Phase 3):**

```
tests/ingestion/
  test_checksum_idempotency.py   — upload same file twice → second returns existing doc_id, no new job
  test_job_state_machine.py      — mock task: queued → running → succeeded path
  test_job_failure_retry.py      — mock: simulate 3 consecutive failures → terminal FAILED state
  test_file_validation.py        — wrong extension, oversized file, malformed PDF → 400 errors
  test_job_status_endpoint.py    — GET /ingestion/jobs/{id} returns correct status
```

**Fallback (Phase 3):**
- If `CELERY_BROKER_URL` is not set, fall back to `BackgroundTasks` (in-process async, no retry, simpler)
- Env variable `ASYNC_BACKEND=celery|background` controls which path is used
- Document in README: "Use `celery` for production-like deployment, `background` for lightweight local dev"

---

### Phase 4 — Cloud Deployment + CI/CD

**Goal:** Live demo URL; interview-ready deployment story.

#### Architecture
```
GitHub ──push──▶ GitHub Actions
                    ├─ test job (pytest)
                    ├─ build job (docker build)
                    └─ deploy job
                          ├─ FastAPI → Azure Container Apps
                          └─ Next.js → Vercel
                Azure:
                  Container Apps (backend)
                  Azure Blob Storage (public demo docs)
                  Azure Key Vault (secrets)
                  Azure Database for Postgres (managed)
                  Azure Cache for Redis (managed)
```

#### [NEW] `.github/workflows/ci.yml`
```yaml
name: CI
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    services:
      postgres:
        image: postgres:16-alpine
        env: { POSTGRES_DB: docopilot_test, ... }
      qdrant:
        image: qdrant/qdrant:v1.12.1
    steps:
      - pip install -r requirements-dev.txt
      - alembic upgrade head
      - pytest tests/ -v --cov=backend
```

#### [NEW] `.github/workflows/deploy.yml`
```yaml
name: Deploy
on:
  push:
    branches: [main]
jobs:
  deploy-backend:
    steps:
      - az login (federated identity)
      - az acr build (build image to Azure Container Registry)
      - az containerapp update (rolling deploy)
  deploy-frontend:
    steps:
      - vercel deploy --prod
```

#### [MODIFY] [backend/main.py](file:///d:/MadRocket/DoCopilot/backend/main.py)
- Add `GET /readyz` endpoint (checks DB ping + Qdrant ping)
- Add `GET /health` (lightweight, no DB call — for load balancer)

#### [NEW] `backend/Dockerfile` updates
- Multi-stage build: `builder` stage installs deps, `runtime` stage is slim
- Add `HEALTHCHECK` instruction

#### [NEW] `infra/` directory (Azure Bicep or simple shell scripts)
- `infra/deploy.sh` — `az containerapp create` with correct env vars from Key Vault references
- `infra/README.md` — step-by-step manual deploy guide (for reproducibility proof)

**Tests (Phase 4):**
- CI runs on every push (automated)
- Manual: Hit live `GET /readyz` → `{"status": "ready", "db": "ok", "qdrant": "ok"}`
- Manual: Upload a PDF on the live URL, confirm streaming answer works end to end

**Fallback (Phase 4):**
- If Azure is unavailable / cost concern, Railway or Render are drop-in alternatives for the backend
- Frontend always on Vercel (free tier, zero config)
- Local Docker Compose remains the primary dev setup — never broken by cloud work

---

### Phase 5 — Stage-Aware Evaluation

**Goal:** Replace single-metric word-overlap eval with a versioned, multi-stage benchmark.

#### New Benchmark Dataset (`data_v2.jsonl`)

60–70 curated cases across 7 categories:

| Category | Count | Purpose |
|---|---|---|
| direct_factual | 15 | Basic retrieval correctness |
| multi_chunk_synthesis | 12 | Tests context window + chunking |
| ambiguous_query | 8 | Tests clarification behavior |
| insufficient_info | 8 | Tests refusal / "I don't know" |
| conflicting_sources | 5 | Tests LLM judgment |
| citation_validation | 7 | Tests `[c1]` citation format |
| safety_adversarial | 5 | Tests guardrails (prompt injection, PII) |

Each row has: `id`, `question`, `expected_answer`, `expected_citations_count`, `category`, `difficulty`, `corpus_version`

#### [NEW] `backend/evaluation/` directory

**`backend/evaluation/metrics.py`** — all 7 metric implementations:

```python
def recall_at_k(retrieved_ids: list, relevant_ids: list, k: int) -> float: ...
def mean_reciprocal_rank(retrieved_ids: list, relevant_ids: list) -> float: ...
def citation_precision(answer: str, expected_citation_count: int) -> float: ...
def groundedness(answer: str, context_chunks: list[str], llm) -> float: ...  # LLM-as-judge
def answer_completeness(answer: str, expected: str, llm) -> float: ...  # LLM-as-judge
def measure_latency_p50_p95(latencies: list[float]) -> dict: ...
def failure_rate(results: list[dict]) -> float: ...
```

**`backend/evaluation/runner.py`** — orchestrates full eval run, saves to `evaluation_runs` table + JSON file

**`backend/evaluation/report.py`** — generates a markdown report table from eval results

#### [MODIFY] [tests/test_rag_eval.py](file:///d:/MadRocket/DoCopilot/tests/test_rag_eval.py)
- Update fixtures to use `data_v2.jsonl`
- Add per-category metric assertions (e.g., refusal rate for `insufficient_info` category ≥ 0.85)
- Add citation precision assertion

#### [MODIFY] [.github/workflows/eval.yml](file:///d:/MadRocket/DoCopilot/.github/workflows/eval.yml)
- Run full eval weekly (cron: `0 2 * * 0`)
- Upload result JSON as GitHub Actions artifact
- Fail if Recall@5 drops below baseline

**Tests (Phase 5):**
- `tests/evaluation/test_metrics.py` — unit test each metric function with known inputs
- `tests/evaluation/test_runner.py` — mock LLM, run full pipeline, assert output structure
- Weekly CI eval run produces a result artifact

---

### Phase 6 — Observability & Operational Metrics

**Goal:** Move from "interesting demo" to "small but credible platform."

#### What to add

**Structured Logging (already partially done — enhance it):**
- Every request logs: `request_id`, `tenant_id`, `user_id`, `endpoint`, `latency_ms`
- Every RAG pipeline step logs: `retrieved_count`, `reranked_count`, `tokens_estimated`, `stage`
- Log format: JSON (production) vs. human-readable (dev) — controlled by `LOG_FORMAT=json|text`

**Latency Tracking:**
```python
# In rag.py — wrap key stages
with timed_stage("hybrid_search") as t:
    retrieved_docs = hybrid_search(...)
logger.info("stage=hybrid_search latency_ms=%d count=%d", t.ms, len(retrieved_docs))
```

**Cost Tracking (approximate):**
```python
# Groq / Qwen token estimation
def estimate_tokens(text: str) -> int:
    return len(text.split()) * 1.3  # rough GPT-style estimate

# Log per-request: input_tokens, output_tokens, estimated_cost_usd
```

**Job Queue Visibility:**
- `GET /ingestion/jobs` (admin only) — lists recent jobs with status counts
- `GET /ingestion/jobs/{id}` — already planned in Phase 3

#### [NEW] `backend/observability/` directory

**`backend/observability/middleware.py`** — request timing middleware (adds `X-Request-ID`, logs latency)

**`backend/observability/logger.py`** — structured JSON logger setup

**`backend/observability/cost.py`** — token estimation + cost tracking helpers

#### [MODIFY] [backend/rag.py](file:///d:/MadRocket/DoCopilot/backend/rag.py)
- Wrap `hybrid_search`, `rerank_documents`, LLM call with timing context managers
- Emit structured log events at each stage

**Tests (Phase 6):**
- `tests/observability/test_middleware.py` — assert `X-Request-ID` present, latency logged
- `tests/observability/test_cost.py` — unit test token estimation function
- Manual: hit `/chat` and verify JSON logs appear with all fields

---

## Verification Plan

### Automated Tests (full matrix)

```
tests/
  db/
    test_models.py           # Phase 1
    test_crud.py             # Phase 1
    test_migrations.py       # Phase 1
  auth/
    test_register_login.py   # Phase 2
    test_cross_tenant.py     # Phase 2 ← most important
    test_token_expiry.py     # Phase 2
    test_invalid_tokens.py   # Phase 2
  ingestion/
    test_checksum_idempotency.py  # Phase 3
    test_job_state_machine.py     # Phase 3
    test_job_failure_retry.py     # Phase 3
    test_file_validation.py       # Phase 3
  evaluation/
    test_metrics.py          # Phase 5
    test_runner.py           # Phase 5
  observability/
    test_middleware.py       # Phase 6
  test_rag_eval.py           # existing — updated for Phase 5
```

**CI commands:**
```bash
# Run unit + integration tests (no LLM cost)
pytest tests/ -v --ignore=tests/test_rag_eval.py

# Run full LLM eval (weekly, requires GROQ_API_KEY)
RUN_FULL_EVAL=true pytest tests/test_rag_eval.py -v
```

### Manual Verification Checklist

| Phase | Check |
|---|---|
| 0 | `docker compose up` — 4 services healthy |
| 1 | `GET /health` → `{"db": "ok", "qdrant": "ok"}` |
| 2 | Register → Login → Upload PDF → Chat with JWT header |
| 2 | Use Tenant A token to query Tenant B doc → 403 |
| 3 | Upload large PDF → immediate `job_id` response → poll status until `succeeded` |
| 3 | Upload same PDF twice → same `document_id`, no duplicate job |
| 4 | Push to `main` → CI passes → live URL responds to `/readyz` |
| 5 | Run eval script → result JSON with all 7 metrics written |
| 6 | Chat request → JSON log line with `latency_ms`, `tokens_estimated` |

---

## Interview Defense Sentences (after all phases)

> "DoCopilot is an authenticated, cloud-deployed document Q&A platform. Qdrant handles hybrid retrieval with BM25 and dense vectors fused via RRF, followed by Cohere reranking. Tenant scope is derived from verified JWT identity — not accepted from request input. Ingestion is asynchronous with a state machine: queued, running, succeeded, or failed, with idempotency by checksum. The system has a stage-aware evaluation benchmark across 7 case categories measuring retrieval recall, citation precision, groundedness, and latency percentiles."

That sentence maps to every sub-question an interviewer could ask.
