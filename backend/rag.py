##Comments maine hi likha hai for clarity and learning purpose
import os
import tempfile
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from uuid import uuid4
import logging
import re

import dotenv
dotenv.load_dotenv()

from langsmith import Client
from langsmith.run_helpers import traceable

from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_groq import ChatGroq

# Reranking model
from sentence_transformers import CrossEncoder

## Qdrant - in place of faiss bhai shab 100 lines of bm25 and hybrid search ko ye ik line replace kar dega
from langchain_qdrant import QdrantVectorStore, RetrievalMode, FastEmbedSparse
from qdrant_client import QdrantClient

###Basic guadrail functions 
from backend.ragguardrails import RagGuardrails

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

langsmith_client: Optional[Client] = None
if os.getenv("LANGCHAIN_API_KEY"):
    langsmith_client = Client()
    logger.info("Langsmith client initialized")

# Chunking config
chunk_size = 2000
chunk_overlap = 400

splitter = RecursiveCharacterTextSplitter(
    chunk_size=chunk_size,
    chunk_overlap=chunk_overlap
)

# Document cache - stores vectorstore and metadata
document_cache: Dict[str, dict] = {}
current_document_id: Optional[str] = None

# Qdrant path config (client created only when needed)
QDRANT_PATH = "./qdrant_data"
logger.info("Qdrant storage path: %s", QDRANT_PATH)

# Global client reference (created once, reused)
_qdrant_client: Optional[QdrantClient] = None


def _get_qdrant_client() -> QdrantClient:
    """Get or create the singleton Qdrant client."""
    global _qdrant_client
    if _qdrant_client is None:
        _qdrant_client = QdrantClient(path=QDRANT_PATH)
        logger.info("Qdrant client created at %s", QDRANT_PATH)
    return _qdrant_client


# ============================================
# PRELOAD EMBEDDINGS AT STARTUP
# ============================================
logger.info("Preloading embedding model (one-time)...")
_load_start = time.time()
_embeddings = HuggingFaceEmbeddings(
    model_name='sentence-transformers/all-MiniLM-L6-v2',
    model_kwargs={'device': 'cpu'},
    encode_kwargs={
        'normalize_embeddings': True,
        'batch_size': 32
    }
)
_embeddings.embed_query("warmup")
logger.info("Embedding model ready in %.2fs", time.time() - _load_start)

# ============================================
# Preload Sparse Embeddings for Qdrant -> BM25 jaise
# ============================================
logger.info("Preloading sparse embedding model for Qdrant...")
_sparse_embed_start = time.time()
_sparse_embeddings = FastEmbedSparse(model_name="Qdrant/bm25")
logger.info("Sparse embedding model ready in %.2fs", time.time() - _sparse_embed_start)


# ============================================
# PRELOAD RERANKER MODEL AT STARTUP
# ============================================
logger.info("Preloading re-ranking model...")
_rerank_start = time.time()
_reranker = CrossEncoder(
    'cross-encoder/ms-marco-MiniLM-L-6-v2',
    device='cpu',
    max_length=512,
)
_reranker.predict([("warmup question", "warmup passage")])
logger.info("Re-ranking model ready in %.2fs", time.time() - _rerank_start)

# ============================================
# RETRIEVAL CONFIG
# ============================================
HYBRID_ENABLED = True      # Qdrant built-in hybrid (BM25 + Vector)
RERANK_ENABLED = True      # Rerank after retrieval
INITIAL_K = 20             # Retrieve this many docs
FINAL_K = 5                # Final docs after reranking


def _get_embeddings() -> HuggingFaceEmbeddings:
    return _embeddings


@traceable(name="create_vector_store")
def create_vector_store(docs: List[Document], collection_name: str = "documents") -> QdrantVectorStore:
    """
    Create Qdrant vector store with BUILT-IN hybrid search.
    """
    if not docs:
        raise ValueError("No documents provided")
    
    t0 = time.time()
    
    # Use in-memory Qdrant (no persistence, but avoids file lock issues)
    vectorstore = QdrantVectorStore.from_documents(
        docs,
        embedding=_embeddings,
        sparse_embedding=_sparse_embeddings,
        location=":memory:",                  # In-memory mode
        collection_name=collection_name,
        retrieval_mode=RetrievalMode.HYBRID,
    )
    
    logger.info("Qdrant hybrid collection '%s' created in %.2fs (%d docs)", 
                collection_name, time.time() - t0, len(docs))
    
    return vectorstore


@traceable(name="hybrid_search")
def hybrid_search(
    query: str,
    vectorstore: QdrantVectorStore,
    top_k: int = 20
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
    results = vectorstore.similarity_search(query, k=top_k)
    logger.info("Hybrid search retrieved %d docs for '%s...'", len(results), query[:30])
    return results


@traceable(name="rerank_documents")
def rerank_documents(query: str, docs: List[Document], top_k: int = 5) -> List[Document]:
    """
    Re-rank documents using CrossEncoder.
    
    Why rerank after retrieval?
    - Bi-encoder (embedding): Fast but approximate
    - Cross-encoder (reranker): Slow but accurate
    
    Pipeline: All chunks → Hybrid (fast, top 20) → Rerank (accurate, top 5) → LLM
    """
    if not docs:
        return docs
    
    query_doc_pairs = [(query, doc.page_content) for doc in docs]
    scores = _reranker.predict(query_doc_pairs)
    
    scored_docs = list(zip(docs, scores))
    scored_docs.sort(key=lambda x: x[1], reverse=True)
    
    logger.info("Reranking: %d docs → top %d", len(docs), top_k)
    
    return [doc for doc, _ in scored_docs[:top_k]]


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
        loader = PyPDFLoader(tmp_path)
        pages = loader.load()
        for page in pages:
            page.metadata["source"] = filename
        docs = splitter.split_documents(pages)
        logger.info("Loaded %d chunks from %s", len(docs), filename)
        return docs
    finally:
        Path(tmp_path).unlink(missing_ok=True)


@traceable(name="load_txt")
def load_text_chunks(text: str, filename: str) -> List[Document]:
    """Load text content directly."""
    if not text.strip():
        raise ValueError('Text content is empty')
    docs = splitter.create_documents([text], metadatas=[{"source": filename}])
    logger.info("Loaded %d chunks from %s", len(docs), filename)
    return docs


@traceable(name="plain_text_chunks")
def plain_text_chunks(raw_text: str, *, source: str = "user_input") -> List[Document]:
    if not raw_text.strip():
        raise ValueError("Input text is empty")
    docs = splitter.create_documents([raw_text], metadatas=[{"source": source}])
    logger.info("Created %d chunks from %s", len(docs), source)
    return docs


# ============================================
# INDEXING
# ============================================

def store_document_cache(docs: List[Document], vectorstore: QdrantVectorStore, collection_name: str) -> str:
    """
    Store vectorstore reference for retrieval.
    
    With Qdrant:
    - Data is persisted on disk automatically (./qdrant_data/)
    - Survives server restarts
    - No need to store BM25 index separately (built-in!)
    """
    global current_document_id, document_cache
    document_cache.clear()
    
    document_id = str(uuid4())
    
    document_cache[document_id] = {
        "vectorstore": vectorstore,
        "collection_name": collection_name,
        "docs": docs,  # Keep for debugging
    }
    
    current_document_id = document_id
    logger.info("Cached document: %s (collection=%s, hybrid=built-in)", document_id, collection_name)
    return document_id


@traceable(name="index_get_pdf", tags=["indexing"])
def index_get_pdf(content: bytes, filename: str) -> str:
    """Index PDF from bytes content."""
    t0 = time.time()
    docs = load_pdf_from_bytes(content, filename)
    collection_name = re.sub(r'[^a-zA-Z0-9_]', '_', filename.rsplit('.', 1)[0])[:50]
    vectorstore = create_vector_store(docs, collection_name)
    doc_id = store_document_cache(docs, vectorstore, collection_name)
    logger.info("Total PDF indexing: %.2fs", time.time() - t0)
    return doc_id


@traceable(name="index_get_txt", tags=["indexing"])
def index_get_txt(text: str, filename: str) -> str:
    """Index text content directly."""
    docs = load_text_chunks(text, filename)
    collection_name = re.sub(r'[^a-zA-Z0-9_]', '_', filename.rsplit('.', 1)[0])[:50]
    vectorstore = create_vector_store(docs, collection_name)
    return store_document_cache(docs, vectorstore, collection_name)


@traceable(name="index_get_plain_text", tags=["indexing"])
def index_get_plain_text(raw_text: str) -> str:
    docs = plain_text_chunks(raw_text)
    collection_name = "plain_text"
    vectorstore = create_vector_store(docs, collection_name)
    return store_document_cache(docs, vectorstore, collection_name)


def get_document_data(document_id: Optional[str] = None) -> dict:
    """Get all document data from cache."""
    target_id = document_id or current_document_id
    if not target_id or target_id not in document_cache:
        raise ValueError("No document found for the given ID")
    return document_cache[target_id]


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
def ask_question(question: str, *, document_id: Optional[str] = None, k: int = 5) -> tuple[str, list[str]]:
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
    retrieved_docs = hybrid_search(question, vectorstore, top_k=retrieve_k)
    
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

    prompt = PromptTemplate(
        template=RAG_TEMPLATE,
        input_variables=["context", "question"],
    )
    final_prompt = prompt.format(context=context, question=question)
    llm = ChatGroq(model="meta-llama/llama-4-scout-17b-16e-instruct", temperature=0)
    response = llm.invoke(final_prompt)

    answer = getattr(response, "text", None) or getattr(response, "content", str(response))
    sources = list(set(doc.metadata.get("source", "") for doc in retrieved_docs))

    return answer, sources

# ============================================
async def query_document(document_id: str , question :str)->dict:
    """Query a document with guradrails checks
    """
    ##input check
    is_safe,message = RagGuardrails.check_input(question)
    if not is_safe:
        return {"answer": message,
                "blocked":True,
                "sources":[]}

    try : 
        answer,sources = ask_question(question,document_id=document_id)
        _,cleaned_answer = RagGuardrails.check_output(answer,sources)
        
        return {"answer":cleaned_answer,
                "blocked":False,
                "sources":sources}
    except Exception as e:
        logger.error("Query failed: %s", str(e))
        return {
            "answer":  f"Error processing the query: {str(e)}",
            "blocked": False,
            "sources": []
        }
            

# ============================================
# QDRANT UTILITY FUNCTIONS
# ============================================

def list_collections() -> List[str]:
    """List all Qdrant collections."""
    try:
        client = QdrantClient(path=QDRANT_PATH)
        collections = client.get_collections()
        result = [c.name for c in collections.collections]
        client.close()
        return result
    except Exception:
        return []


def get_collection_info(collection_name: str) -> dict:
    """Get detailed info about a Qdrant collection."""
    try:
        client = QdrantClient(path=QDRANT_PATH)
        info = client.get_collection(collection_name)
        result = {
            "name": collection_name,
            "vectors_count": info.vectors_count,
            "points_count": info.points_count,
        }
        client.close()
        return result
    except Exception as e:
        return {"error": str(e)}


def delete_collection(collection_name: str) -> bool:
    """Delete a Qdrant collection."""
    try:
        client = QdrantClient(path=QDRANT_PATH)
        client.delete_collection(collection_name)
        client.close()
        logger.info("Deleted collection: %s", collection_name)
        return True
    except Exception as e:
        logger.error("Failed to delete collection: %s", e)
        return False


def collection_exists(collection_name: str) -> bool:
    """Check if a collection exists."""
    try:
        client = QdrantClient(path=QDRANT_PATH)
        client.get_collection(collection_name)
        client.close()
        return True
    except Exception:
        return False