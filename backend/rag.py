##Comments maine hi likha hai for clarity and learning purpose
from __future__ import annotations
import os
import json
import tempfile
import time
from pathlib import Path
from typing import AsyncIterator, Dict, List, Optional, Tuple, TYPE_CHECKING
from uuid import uuid4
import logging
import re

import dotenv
from pathlib import Path as _Path
# override=True ensures a freshly-pasted key is always picked up
_env_path = _Path(__file__).resolve().parent.parent / ".env"
dotenv.load_dotenv(dotenv_path=_env_path, override=True)

from langsmith import Client
from langsmith.run_helpers import traceable

###Basic guadrail functions 
from backend.ragguardrails import RagGuardrails
from backend.observability.timing import timed_stage, estimate_token_cost

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

langsmith_client: Optional[Client] = None
if os.getenv("LANGCHAIN_API_KEY"):
    langsmith_client = Client()
    logger.info("Langsmith client initialized")

if TYPE_CHECKING:
    from qdrant_client import QdrantClient
    from langchain_qdrant import QdrantVectorStore
    from langchain_core.documents import Document

# Chunking config
chunk_size = 2000
chunk_overlap = 400

_splitter = None

def get_splitter():
    global _splitter
    if _splitter is None:
        from langchain_text_splitters import RecursiveCharacterTextSplitter
        _splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
    return _splitter

# Document cache - stores vectorstore and metadata
document_cache: Dict[str, dict] = {}
current_document_id: Optional[str] = None

# Qdrant path config (client created only when needed)
QDRANT_PATH = "./qdrant_data"
logger.info("Qdrant storage path: %s", QDRANT_PATH)

# Global client reference (created once, reused)
_qdrant_client: Optional[QdrantClient] = None

QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_HOST = os.getenv("QDRANT_HOST")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")

def _get_qdrant_client() -> QdrantClient:
    """Get or create the singleton Qdrant client."""
    global _qdrant_client
    if _qdrant_client is None:
        from qdrant_client import QdrantClient
        
        # 1. Explicit QDRANT_URL (production / Azure / cloud)
        if QDRANT_URL:
            logger.info("Connecting to Qdrant server at: %s", QDRANT_URL)
            _qdrant_client = QdrantClient(
                url=QDRANT_URL,
                api_key=QDRANT_API_KEY if QDRANT_API_KEY else None,
                timeout=120.0
            )
        # 2. Explicit QDRANT_HOST
        elif QDRANT_HOST:
            logger.info("Connecting to Qdrant host at: %s:%d", QDRANT_HOST, QDRANT_PORT)
            _qdrant_client = QdrantClient(
                host=QDRANT_HOST,
                port=QDRANT_PORT,
                api_key=QDRANT_API_KEY if QDRANT_API_KEY else None,
                timeout=120.0
            )
        # 3. Check if local Qdrant server is running on http://localhost:6333
        else:
            try:
                logger.info("Checking for running Qdrant server at http://localhost:6333...")
                client_temp = QdrantClient(url="http://localhost:6333", timeout=3.0)
                client_temp.get_collections()
                _qdrant_client = client_temp
                logger.info("Connected to running Qdrant server at http://localhost:6333")
            except Exception:
                # 4. Fallback to local directory mode if no server is listening
                logger.info("Qdrant server not detected. Attempting local directory: %s", QDRANT_PATH)
                try:
                    _qdrant_client = QdrantClient(path=QDRANT_PATH)
                except Exception as exc:
                    logger.warning("Local directory %s locked (%s). Falling back to in-memory Qdrant.", QDRANT_PATH, exc)
                    _qdrant_client = QdrantClient(location=":memory:")
                    
        logger.info("Qdrant client initialized successfully")
    return _qdrant_client



_embeddings = None
_sparse_embeddings = None
_reranker = None

def get_embeddings():
    global _embeddings
    if _embeddings is None:
        _load_start = time.time()

        # 1. Try sentence-transformers (CI / full dev environment)
        try:
            import sentence_transformers  # noqa: F401
            logger.info("sentence-transformers detected — using local embeddings.")
            from langchain_huggingface import HuggingFaceEmbeddings
            _embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2",
                model_kwargs={"device": "cpu"},
                encode_kwargs={"normalize_embeddings": True, "batch_size": 32},
            )
            _embeddings.embed_query("warmup")
        except Exception:
            # 2. Try FastEmbed (lightweight ONNX local embeddings, no HuggingFace API key required)
            try:
                logger.info("sentence-transformers unavailable — using FastEmbed (ONNX local mode).")
                from langchain_community.embeddings import FastEmbedEmbeddings
                _embeddings = FastEmbedEmbeddings(model_name="BAAI/bge-small-en-v1.5")
                _embeddings.embed_query("warmup")
            except Exception as fe_err:
                logger.info("FastEmbed unavailable (%s) — falling back to HuggingFace Endpoint API.", fe_err)
                from langchain_huggingface import HuggingFaceEndpointEmbeddings
                raw_token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACEHUB_API_TOKEN")
                hf_token = raw_token.strip() if raw_token else None
                _embeddings = HuggingFaceEndpointEmbeddings(
                    model="sentence-transformers/all-MiniLM-L6-v2",
                    task="feature-extraction",
                    huggingfacehub_api_token=hf_token,
                )

        logger.info("Embedding model ready in %.2fs", time.time() - _load_start)
    return _embeddings


def get_sparse_embeddings():
    global _sparse_embeddings
    if _sparse_embeddings is None:
        logger.info("Preloading sparse embedding model for Qdrant...")
        _sparse_embed_start = time.time()
        from langchain_qdrant import FastEmbedSparse
        _sparse_embeddings = FastEmbedSparse(model_name="Qdrant/bm25")
        logger.info("Sparse embedding model ready in %.2fs", time.time() - _sparse_embed_start)
    return _sparse_embeddings

def get_reranker():
    # Deprecated: We use Cohere Rerank API over HTTP instead of a local model.
    # OLD AND LOCAL:
    # global _reranker
    # if _reranker is None:
    #     logger.info("Preloading re-ranking model...")
    #     _rerank_start = time.time()
    #     from sentence_transformers import CrossEncoder
    #     _reranker = CrossEncoder(
    #         'cross-encoder/ms-marco-MiniLM-L-6-v2',
    #         device='cpu',
    #         max_length=512,
    #     )
    #     _reranker.predict([("warmup question", "warmup passage")])
    #     logger.info("Re-ranking model ready in %.2fs", time.time() - _rerank_start)
    # return _reranker
    return None

# ============================================
# RETRIEVAL CONFIG
# ============================================
HYBRID_ENABLED = True      # Qdrant built-in hybrid (BM25 + Vector)
RERANK_ENABLED = True      # Rerank after retrieval
INITIAL_K = 20             # Retrieve this many docs
FINAL_K = 5                # Final docs after reranking


def _get_embeddings():
    return get_embeddings()


@traceable(name="create_vector_store")
def create_vector_store(docs: List[Document], collection_name: str = "documents") -> QdrantVectorStore:
    """
    Create Qdrant vector store with BUILT-IN hybrid search.
    """
    if not docs:
        raise ValueError("No documents provided")
    
    t0 = time.time()
    from langchain_qdrant import QdrantVectorStore, RetrievalMode

    q_url = QDRANT_URL
    q_host = QDRANT_HOST
    q_api_key = QDRANT_API_KEY if QDRANT_API_KEY else None

    # Check if local Qdrant server is running at http://localhost:6333
    if not q_url and not q_host:
        try:
            import httpx
            r = httpx.get("http://localhost:6333", timeout=1.5)
            if r.status_code == 200:
                q_url = "http://localhost:6333"
        except Exception:
            pass

    if q_url:
        logger.info("Indexing into Qdrant server URL: %s", q_url)
        vectorstore = QdrantVectorStore.from_documents(
            docs,
            embedding=get_embeddings(),
            sparse_embedding=get_sparse_embeddings(),
            url=q_url,
            api_key=q_api_key,
            timeout=120.0,
            collection_name=collection_name,
            retrieval_mode=RetrievalMode.HYBRID,
        )
    elif q_host:
        logger.info("Indexing into Qdrant host: %s:%d", q_host, QDRANT_PORT)
        vectorstore = QdrantVectorStore.from_documents(
            docs,
            embedding=get_embeddings(),
            sparse_embedding=get_sparse_embeddings(),
            host=q_host,
            port=QDRANT_PORT,
            api_key=q_api_key,
            timeout=120.0,
            collection_name=collection_name,
            retrieval_mode=RetrievalMode.HYBRID,
        )
    else:
        logger.info("Attempting local Qdrant directory: %s", QDRANT_PATH)
        try:
            vectorstore = QdrantVectorStore.from_documents(
                docs,
                embedding=get_embeddings(),
                sparse_embedding=get_sparse_embeddings(),
                path=QDRANT_PATH,
                collection_name=collection_name,
                retrieval_mode=RetrievalMode.HYBRID,
            )
        except Exception as exc:
            logger.warning("Local Qdrant directory %s locked (%s). Using in-memory mode.", QDRANT_PATH, exc)
            vectorstore = QdrantVectorStore.from_documents(
                docs,
                embedding=get_embeddings(),
                sparse_embedding=get_sparse_embeddings(),
                location=":memory:",
                collection_name=collection_name,
                retrieval_mode=RetrievalMode.HYBRID,
            )


    
    # Create keyword payload index for tenant_id filtering
    try:
        from qdrant_client.models import PayloadSchemaType
        client = _get_qdrant_client()
        client.create_payload_index(
            collection_name=collection_name,
            field_name="metadata.tenant_id",
            field_schema=PayloadSchemaType.KEYWORD
        )
        logger.info("Created keyword payload index for metadata.tenant_id on collection '%s'", collection_name)
    except Exception as e:
        logger.warning("Could not create payload index on collection '%s': %s", collection_name, e)
    
    logger.info("Qdrant hybrid collection '%s' created in %.2fs (%d docs)", 
                collection_name, time.time() - t0, len(docs))
    
    return vectorstore


@traceable(name="hybrid_search")
def hybrid_search(
    query: str,
    vectorstore: QdrantVectorStore,
    top_k: int = 20,
    tenant_id: str = "default"
) -> List[Document]:
    """
    Qdrant built-in hybrid search.
    
    THIS REPLACES ~100 LINES OF MANUAL CODE:
    - bm25_search()
    - vector_search_faiss()
    - rrf_fusion()
    - hybrid_search_manual()
    
    Qdrant internally:
    1. Searches dense vectors (semantic similarity)
    2. Searches sparse vectors (BM25-like keyword matching)
    3. Combines using RRF fusion
    4. Returns unified results
    
    All in ONE function call!
    """
    from qdrant_client.models import Filter, FieldCondition, MatchValue

    tenant_filter = Filter(
        must=[
            FieldCondition(
                key="metadata.tenant_id",
                match=MatchValue(value=tenant_id)
            )
        ]
    ) if tenant_id else None

    results = vectorstore.similarity_search(query, k=top_k, filter=tenant_filter)
    logger.info("Hybrid search retrieved %d docs for tenant '%s' and query '%s...'", len(results), tenant_id, query[:30])
    return results


@traceable(name="rerank_documents")
def rerank_documents(query: str, docs: List[Document], top_k: int = 5) -> List[Document]:
    """
    Re-rank documents using Cohere Rerank API.
    
    Pipeline: All chunks → Hybrid (fast, top 20) → Rerank (accurate, top 5) → LLM
    """
    if not docs:
        return docs
    
    cohere_api_key = os.getenv("COHERE_API_KEY")
    if not cohere_api_key:
        logger.warning("COHERE_API_KEY is not set. Skipping reranking.")
        return docs[:top_k]
        
    import httpx
    try:
        headers = {
            "Authorization": f"Bearer {cohere_api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": "rerank-english-v3.0",
            "query": query,
            "documents": [doc.page_content for doc in docs],
            "top_n": top_k
        }
        
        with httpx.Client(timeout=10.0) as client:
            resp = client.post("https://api.cohere.com/v1/rerank", headers=headers, json=payload)
            resp.raise_for_status()
            data = resp.json()
            
        reranked_docs = []
        for result in data.get("results", []):
            idx = result["index"]
            reranked_docs.append(docs[idx])
            
        # OLD AND LOCAL:
        # query_doc_pairs = [(query, doc.page_content) for doc in docs]
        # scores = get_reranker().predict(query_doc_pairs)
        # scored_docs = list(zip(docs, scores))
        # scored_docs.sort(key=lambda x: x[1], reverse=True)
        # logger.info("Reranking: %d docs → top %d", len(docs), top_k)
        # reranked_docs = [doc for doc, _ in scored_docs[:top_k]]
            
        logger.info("Cohere Rerank successful: %d docs → top %d", len(docs), len(reranked_docs))
        return reranked_docs
    except Exception as e:
        logger.error("Cohere Rerank failed: %s. Returning top %d un-reranked documents.", e, top_k)
        return docs[:top_k]


# ============================================
# DOCUMENT LOADING
# ============================================

@traceable(name="load_pdf")
def load_pdf_from_bytes(content: bytes, filename: str) -> List[Document]:
    """Load PDF from bytes using a temp file."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(content)
        tmp_path = tmp.name
    try:
        from langchain_community.document_loaders import PyPDFLoader
        loader = PyPDFLoader(tmp_path)
        pages = loader.load()
        for page in pages:
            page.metadata["source"] = filename
        docs = get_splitter().split_documents(pages)
        logger.info("Loaded %d chunks from %s", len(docs), filename)
        return docs
    finally:
        Path(tmp_path).unlink(missing_ok=True)


@traceable(name="load_txt")
def load_text_chunks(text: str, filename: str) -> List[Document]:
    """Load text content directly."""
    if not text.strip():
        raise ValueError('Text content is empty')
    docs = get_splitter().create_documents([text], metadatas=[{"source": filename}])
    logger.info("Loaded %d chunks from %s", len(docs), filename)
    return docs


@traceable(name="plain_text_chunks")
def plain_text_chunks(raw_text: str, *, source: str = "user_input") -> List[Document]:
    if not raw_text.strip():
        raise ValueError("Input text is empty")
    docs = get_splitter().create_documents([raw_text], metadatas=[{"source": source}])
    logger.info("Created %d chunks from %s", len(docs), source)
    return docs


# ============================================
# INDEXING
# ============================================

def store_document_cache(docs: List[Document], vectorstore: QdrantVectorStore, collection_name: str, document_id: Optional[str] = None) -> str:
    """
    Store vectorstore reference for retrieval.
    Registers under both collection_name and document_id so lookups by Postgres UUID succeed.
    """
    global current_document_id, document_cache
    
    key_id = document_id or collection_name
    data = {
        "vectorstore": vectorstore,
        "collection_name": collection_name,
        "docs": docs,
    }
    
    document_cache[key_id] = data
    document_cache[collection_name] = data
    
    current_document_id = key_id
    logger.info("Cached document: key=%s (collection=%s)", key_id, collection_name)
    return key_id


@traceable(name="index_get_pdf", tags=["indexing"])
def index_get_pdf(content: bytes, filename: str, tenant_id: str = "default", document_id: Optional[str] = None) -> str:
    """Index PDF from bytes content."""
    t0 = time.time()
    docs = load_pdf_from_bytes(content, filename)
    for doc in docs:
        doc.metadata["tenant_id"] = tenant_id
    base_name = re.sub(r'[^a-zA-Z0-9_]', '_', filename.rsplit('.', 1)[0])[:30]
    coll_suffix = document_id.replace('-', '_')[:16] if document_id else str(uuid4())[:8]
    collection_name = f"doc_{base_name}_{coll_suffix}"
    vectorstore = create_vector_store(docs, collection_name)
    doc_id = store_document_cache(docs, vectorstore, collection_name, document_id=document_id)
    logger.info("Total PDF indexing: %.2fs", time.time() - t0)
    return doc_id


@traceable(name="index_get_txt", tags=["indexing"])
def index_get_txt(text: str, filename: str, tenant_id: str = "default", document_id: Optional[str] = None) -> str:
    """Index text content directly."""
    docs = load_text_chunks(text, filename)
    for doc in docs:
        doc.metadata["tenant_id"] = tenant_id
    base_name = re.sub(r'[^a-zA-Z0-9_]', '_', filename.rsplit('.', 1)[0])[:30]
    coll_suffix = document_id.replace('-', '_')[:16] if document_id else str(uuid4())[:8]
    collection_name = f"doc_{base_name}_{coll_suffix}"
    vectorstore = create_vector_store(docs, collection_name)
    return store_document_cache(docs, vectorstore, collection_name, document_id=document_id)


@traceable(name="index_get_plain_text", tags=["indexing"])
def index_get_plain_text(raw_text: str, tenant_id: str = "default", document_id: Optional[str] = None) -> str:
    docs = plain_text_chunks(raw_text)
    for doc in docs:
        doc.metadata["tenant_id"] = tenant_id
    coll_suffix = document_id.replace('-', '_')[:16] if document_id else str(uuid4())[:8]
    collection_name = f"plain_text_{coll_suffix}"
    vectorstore = create_vector_store(docs, collection_name)
    return store_document_cache(docs, vectorstore, collection_name, document_id=document_id)


def get_document_data(document_id: Optional[str] = None) -> dict:
    """Get all document data from cache. Reconstructs if missing but exists in database."""
    target_id = document_id or current_document_id
    if not target_id:
        raise ValueError("No document found for the given ID")
        
    if target_id in document_cache:
        return document_cache[target_id]

    # Candidate collection names in Qdrant
    sanitized_id = target_id.replace("-", "_")
    candidates = [
        target_id,
        sanitized_id,
    ]

    # Look up active qdrant_collection from Postgres DocumentVersion if target_id is a UUID
    try:
        import uuid
        from sqlalchemy import select
        from backend.db.session import AsyncSessionLocal
        from backend.db.models import DocumentVersion
        import asyncio

        async def _lookup_db():
            async with AsyncSessionLocal() as db:
                res = await db.execute(
                    select(DocumentVersion.qdrant_collection)
                    .where(DocumentVersion.document_id == uuid.UUID(target_id))
                    .where(DocumentVersion.is_active == True)
                )
                return res.scalar_one_or_none()

        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as pool:
                    db_coll = pool.submit(lambda: asyncio.run(_lookup_db())).result()
            else:
                db_coll = loop.run_until_complete(_lookup_db())

            if db_coll and db_coll not in candidates:
                candidates.insert(0, db_coll)
        except Exception:
            pass
    except Exception:
        pass

    # Search Qdrant for matching collection
    for coll in candidates:
        if collection_exists(coll):
            logger.info("Reconstructing vectorstore for existing collection: %s (target_id=%s)", coll, target_id)
            from langchain_qdrant import QdrantVectorStore, RetrievalMode
            client = _get_qdrant_client()
            vectorstore = QdrantVectorStore(
                client=client,
                collection_name=coll,
                embedding=get_embeddings(),
                sparse_embedding=get_sparse_embeddings(),
                retrieval_mode=RetrievalMode.HYBRID,
            )
            data = {
                "vectorstore": vectorstore,
                "collection_name": coll,
                "docs": [],
            }
            document_cache[target_id] = data
            document_cache[coll] = data
            return data

    # Also list all Qdrant collections to check if any collection contains matching prefix
    try:
        client = _get_qdrant_client()
        all_colls = [c.name for c in client.get_collections().collections]
        for coll in all_colls:
            if sanitized_id in coll:
                logger.info("Found matching collection by substring: %s for target_id %s", coll, target_id)
                from langchain_qdrant import QdrantVectorStore, RetrievalMode
                vectorstore = QdrantVectorStore(
                    client=client,
                    collection_name=coll,
                    embedding=get_embeddings(),
                    sparse_embedding=get_sparse_embeddings(),
                    retrieval_mode=RetrievalMode.HYBRID,
                )
                data = {
                    "vectorstore": vectorstore,
                    "collection_name": coll,
                    "docs": [],
                }
                document_cache[target_id] = data
                document_cache[coll] = data
                return data
    except Exception:
        pass

    raise ValueError(f"No document/collection found for the given ID: {target_id}")



# ============================================
# MAIN QA FUNCTION
# ============================================

@traceable(
    name="ask_question",
    metadata={
        "version": "2.0",
        "model": "llama-4-scout",
        "vector_db": "qdrant",
        "hybrid": "qdrant_built_in",
        "rerank": RERANK_ENABLED
    },
    tags=["rag", "qa", "hybrid", "qdrant"]
)
def ask_question(question: str, *, document_id: Optional[str] = None, k: int = 5, tenant_id: str = "default") -> tuple[str, list[str]]:
    """
    Main RAG question-answering function.
    
    Pipeline:
    1. Hybrid retrieval (Qdrant handles BM25 + Vector + RRF internally)
    2. Rerank with CrossEncoder
    3. Build context and generate answer with LLM
    """
    doc_data = get_document_data(document_id)
    vectorstore = doc_data["vectorstore"]
    
    # Step 1: Retrieval (Qdrant hybrid search)
    retrieve_k = INITIAL_K if RERANK_ENABLED else k
    retrieved_docs = hybrid_search(question, vectorstore, top_k=retrieve_k, tenant_id=tenant_id)
    
    logger.info("Retrieved %d documents for question: '%s...'", len(retrieved_docs), question[:50])
    
    # Step 2: Rerank if enabled
    if RERANK_ENABLED and len(retrieved_docs) > k:
        retrieved_docs = rerank_documents(question, retrieved_docs, top_k=k)
    
    # Step 3: Build context
    context = "\n\n".join(
        f"[c{i+1}] {d.page_content}\nMETADATA: {d.metadata}"
        for i, d in enumerate(retrieved_docs)
    )

    RAG_TEMPLATE = """Role: You are a copilot-style enterprise assistant.

Rules:
- Use ONLY information supported by <context>.
- If missing, say "I don't know based on the provided context." and ask 1 clarifying question.
- Add citations like [c1] after every factual sentence.

<context>
{context}
</context>

Question: {question}
Answer (bullets):
"""

    final_prompt = RAG_TEMPLATE.format(context=context, question=question)
    groq_api_key = os.getenv("GROQ_API_KEY")
    if not groq_api_key:
        raise ValueError("GROQ_API_KEY is not set. Please add it to your .env file.")
    from langchain_groq import ChatGroq
    model_name = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
    llm = ChatGroq(model=model_name, temperature=0, api_key=groq_api_key)
    response = llm.invoke(final_prompt)


    answer = getattr(response, "text", None) or getattr(response, "content", str(response))
    sources = list(set(doc.metadata.get("source", "") for doc in retrieved_docs))

    return answer, sources

# ============================================
async def query_document(question, document_id=None, tenant_id="default", request_id=None) -> dict:
    """Query a document with guardrails checks and observability timing."""
    # Input guardrail check
    is_safe, message = RagGuardrails.check_input(question)
    if not is_safe:
        return {"answer": message, "blocked": True, "sources": []}

    try:
        async with timed_stage("rag_pipeline", request_id=request_id, tenant_id=tenant_id):
            answer, sources = ask_question(question, document_id=document_id, tenant_id=tenant_id)

        _, cleaned_answer = RagGuardrails.check_output(answer, sources)
        
        # Log estimated token cost for observability
        input_tokens = len(question.split()) * 4
        output_tokens = len(cleaned_answer.split()) * 4
        cost = estimate_token_cost(input_tokens, output_tokens)
        logger.info(
            "Query cost estimate",
            extra={
                "request_id": request_id,
                "tenant_id": tenant_id,
                "estimated_cost_usd": cost,
                "token_count": input_tokens + output_tokens,
            }
        )


        return {"answer": cleaned_answer, "blocked": False, "sources": sources}  # ✅ real sources

    except Exception as e:
        return {"answer": f"Error: {str(e)}", "blocked": False, "sources": []}

# ============================================
# STREAMING QA FUNCTION
# ============================================

async def stream_answer(
    question: str,
    *,
    document_id: Optional[str] = None,
    k: int = 5,
    tenant_id: str = "default",
) -> AsyncIterator[str]:
    """
    Stream the LLM answer token-by-token as SSE events.

    Pipeline is identical to ask_question(), but uses llm.astream()
    so each token chunk is yielded immediately.

    SSE event format
    ----------------
    Token chunk:  data: {"token": "..."}
    Final event:  data: {"done": true, "sources": [...], "answer": "<full cleaned answer>"}
    Error event:  data: {"error": "<message>"}
    """
    try:
        doc_data = get_document_data(document_id)
    except ValueError as exc:
        yield f"data: {json.dumps({'error': str(exc)})}\n\n"
        return

    vectorstore = doc_data["vectorstore"]

    # Step 1: Hybrid retrieval
    retrieve_k = INITIAL_K if RERANK_ENABLED else k
    retrieved_docs = hybrid_search(question, vectorstore, top_k=retrieve_k, tenant_id=tenant_id)

    # Step 2: Rerank
    if RERANK_ENABLED and len(retrieved_docs) > k:
        retrieved_docs = rerank_documents(question, retrieved_docs, top_k=k)

    # Step 3: Build context
    context = "\n\n".join(
        f"[c{i+1}] {d.page_content}\nMETADATA: {d.metadata}"
        for i, d in enumerate(retrieved_docs)
    )

    RAG_TEMPLATE = """Role: You are a copilot-style enterprise assistant.

Rules:
- Use ONLY information supported by <context>.
- If missing, say "I don't know based on the provided context." and ask 1 clarifying question.
- Add citations like [c1] after every factual sentence.

<context>
{context}
</context>

Question: {question}
Answer (bullets):
"""
    final_prompt = RAG_TEMPLATE.format(context=context, question=question)

    groq_api_key = os.getenv("GROQ_API_KEY")
    if not groq_api_key:
        yield f"data: {json.dumps({'error': 'GROQ_API_KEY is not set.'})}\n\n"
        return

    from langchain_groq import ChatGroq
    model_name = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
    llm = ChatGroq(model=model_name, temperature=0, api_key=groq_api_key)

    sources = list(set(doc.metadata.get("source", "") for doc in retrieved_docs))

    # Step 4: Stream tokens
    full_answer = ""
    try:
        async for chunk in llm.astream(final_prompt):
            token = getattr(chunk, "content", "") or ""
            if token:
                full_answer += token
                yield f"data: {json.dumps({'token': token})}\n\n"
    except Exception as exc:
        logger.error("Streaming error: %s", exc)
        yield f"data: {json.dumps({'error': str(exc)})}\n\n"
        return

    # Step 5: Output guardrails on full answer, send DONE
    _, cleaned = RagGuardrails.check_output(full_answer, sources)
    yield f"data: {json.dumps({'done': True, 'sources': sources, 'answer': cleaned})}\n\n"


# ============================================

def list_collections() -> List[str]:
    """List all Qdrant collections."""
    try:
        client = _get_qdrant_client()
        collections = client.get_collections()
        return [c.name for c in collections.collections]
    except Exception as e:
        logger.error("Failed to list collections: %s", e)
        return []


def get_collection_info(collection_name: str) -> dict:
    """Get detailed info about a Qdrant collection."""
    try:
        client = _get_qdrant_client()
        info = client.get_collection(collection_name)
        return {
            "name": collection_name,
            "vectors_count": info.vectors_count,
            "points_count": info.points_count,
        }
    except Exception as e:
        return {"error": str(e)}


def delete_collection(collection_name: str) -> bool:
    """Delete a Qdrant collection."""
    try:
        client = _get_qdrant_client()
        client.delete_collection(collection_name)
        logger.info("Deleted collection: %s", collection_name)
        return True
    except Exception as e:
        logger.error("Failed to delete collection: %s", e)
        return False


def collection_exists(collection_name: str) -> bool:
    """Check if a collection exists."""
    try:
        client = _get_qdrant_client()
        client.get_collection(collection_name)
        return True
    except Exception:
        return False