##Comments maine hi likha hai for clarity and learning purpose
import os
import tempfile
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from uuid import uuid4
import logging

import dotenv
dotenv.load_dotenv()

from langsmith import Client
from langsmith.run_helpers import traceable

from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_groq import ChatGroq
# Reranking model
from sentence_transformers import CrossEncoder

# BM25 for hybrid retrieval
from rank_bm25 import BM25Okapi
import re

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

# Document cache - stores vectorstore, bm25, and original docs
document_cache: Dict[str, dict] = {}
current_document_id: Optional[str] = None

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
HYBRID_ENABLED = True      # BM25 + Vector search
RERANK_ENABLED = True      # Rerank after retrieval
INITIAL_K = 20             # Retrieve from each method
FINAL_K = 5                # Final docs after fusion/reranking
RRF_K = 60                 # RRF constant


def _get_embeddings() -> HuggingFaceEmbeddings:
    return _embeddings


def tokenize(text: str) -> List[str]:
    """Simple tokenizer for BM25."""
    return re.findall(r'\w+', text.lower())


# ============================================
# BM25 FUNCTIONS
# ============================================

@traceable(name="create_bm25_index")
def create_bm25_index(docs: List[Document]) -> BM25Okapi:
    """Create BM25 index from document chunks."""
    tokenized_docs = [tokenize(doc.page_content) for doc in docs]
    bm25 = BM25Okapi(tokenized_docs)
    logger.info("BM25 index created: %d chunks", len(docs))
    return bm25


@traceable(name="bm25_search")
def bm25_search(
    query: str, 
    bm25: BM25Okapi, 
    docs: List[Document], 
    top_k: int = 20
) -> List[Tuple[Document, float]]:
    """
    BM25 keyword search.
    Returns: List of (Document, score) - DIFFERENT chunks than vector!
    """
    query_tokens = tokenize(query)
    scores = bm25.get_scores(query_tokens)
    
    # Pair docs with scores and sort
    scored_docs = list(zip(docs, scores))
    scored_docs.sort(key=lambda x: x[1], reverse=True)
    
    logger.info("BM25: top score=%.2f for '%s...'", 
                scored_docs[0][1] if scored_docs else 0, 
                query[:30])
    
    return scored_docs[:top_k]


@traceable(name="vector_search")
def vector_search(
    query: str, 
    vectorstore: FAISS, 
    top_k: int = 20
) -> List[Tuple[Document, float]]:
    """
    Vector semantic search.
    Returns: List of (Document, score) - DIFFERENT chunks than BM25!
    """
    results = vectorstore.similarity_search_with_score(query, k=top_k)
    # Convert distance to similarity (higher = better)
    scored_docs = [(doc, 1 / (1 + dist)) for doc, dist in results]
    
    logger.info("Vector: top score=%.3f for '%s...'", 
                scored_docs[0][1] if scored_docs else 0, 
                query[:30])
    
    return scored_docs


@traceable(name="rrf_fusion")
def rrf_fusion(
    results_list: List[List[Tuple[Document, float]]], 
    k: int = 60
) -> List[Tuple[Document, float]]:
    """
    Reciprocal Rank Fusion - combines rankings from multiple methods.
    
    Chunks in BOTH lists get higher scores!
    """
    doc_scores: Dict[int, float] = {}
    doc_map: Dict[int, Document] = {}
    
    for results in results_list:
        for rank, (doc, _) in enumerate(results, start=1):
            # Use hash of content as unique ID
            doc_id = hash(doc.page_content)
            
            if doc_id not in doc_map:
                doc_map[doc_id] = doc
                doc_scores[doc_id] = 0.0
            
            # RRF: 1/(k + rank)
            doc_scores[doc_id] += 1.0 / (k + rank)
    
    # Sort by RRF score
    sorted_items = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)
    
    logger.info("RRF: %d unique chunks from %d methods", len(sorted_items), len(results_list))
    
    return [(doc_map[doc_id], score) for doc_id, score in sorted_items]


@traceable(name="hybrid_search")
def hybrid_search(
    query: str,
    vectorstore: FAISS,
    bm25: BM25Okapi,
    docs: List[Document],
    initial_k: int = 20,
    final_k: int = 5
) -> List[Document]:
    """
    Hybrid search: BM25 + Vector + RRF fusion.
    
    Each method finds DIFFERENT chunks, RRF combines them.
    Chunks appearing in BOTH get boosted!
    """
    # BM25 finds chunks by keyword match
    bm25_results = bm25_search(query, bm25, docs, top_k=initial_k)
    
    # Vector finds chunks by semantic similarity
    vector_results = vector_search(query, vectorstore, top_k=initial_k)
    
    # RRF combines both - chunks in both lists score higher
    fused = rrf_fusion([bm25_results, vector_results], k=RRF_K)
    
    return [doc for doc, _ in fused[:final_k]]


@traceable(name="rerank_documents")
def rerank_documents(query: str, docs: List[Document], top_k: int = 5) -> List[Document]:
    """Re-rank documents using CrossEncoder."""
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

@traceable(name="create_vector_store")
def create_vector_store(docs: List[Document]) -> FAISS:
    if not docs:
        raise ValueError("No documents provided")
    t0 = time.time()
    vectorstore = FAISS.from_documents(docs, _embeddings)
    logger.info("Vector store created in %.2fs (%d docs)", time.time() - t0, len(docs))
    return vectorstore


def store_document_cache(docs: List[Document], vectorstore: FAISS) -> str:
    """Store vectorstore + BM25 + original docs for hybrid retrieval."""
    global current_document_id, document_cache
    document_cache.clear()
    
    document_id = str(uuid4())
    
    # Create BM25 index if hybrid enabled
    bm25 = create_bm25_index(docs) if HYBRID_ENABLED else None
    
    # Store all components
    document_cache[document_id] = {
        "vectorstore": vectorstore,
        "bm25": bm25,
        "docs": docs  # BM25 needs this to return documents!
    }
    
    current_document_id = document_id
    logger.info("Cached document: %s (hybrid=%s)", document_id, HYBRID_ENABLED)
    return document_id


@traceable(name="index_get_pdf", tags=["indexing"])
def index_get_pdf(content: bytes, filename: str) -> str:
    """Index PDF from bytes content."""
    t0 = time.time()
    docs = load_pdf_from_bytes(content, filename)
    vectorstore = create_vector_store(docs)
    doc_id = store_document_cache(docs, vectorstore)
    logger.info("Total PDF indexing: %.2fs", time.time() - t0)
    return doc_id


@traceable(name="index_get_txt", tags=["indexing"])
def index_get_txt(text: str, filename: str) -> str:
    """Index text content directly."""
    docs = load_text_chunks(text, filename)
    vectorstore = create_vector_store(docs)
    return store_document_cache(docs, vectorstore)


@traceable(name="index_get_plain_text", tags=["indexing"])
def index_get_plain_text(raw_text: str) -> str:
    docs = plain_text_chunks(raw_text)
    vectorstore = create_vector_store(docs)
    return store_document_cache(docs, vectorstore)


def get_document_data(document_id: Optional[str] = None) -> dict:
    """Get all document data (vectorstore, bm25, docs)."""
    target_id = document_id or current_document_id
    if not target_id or target_id not in document_cache:
        raise ValueError("No document found for the given ID")
    return document_cache[target_id]


# ============================================
# MAIN QA FUNCTION
# ============================================

@traceable(
    name="ask_question",
    metadata={"version": "1.2", "model": "llama-4-scout", "hybrid": HYBRID_ENABLED, "rerank": RERANK_ENABLED},
    tags=["rag", "qa", "hybrid"] if HYBRID_ENABLED else ["rag", "qa"]
)
def ask_question(question: str, *, document_id: Optional[str] = None, k: int = 5) -> tuple[str, list[str]]:
    doc_data = get_document_data(document_id)
    vectorstore = doc_data["vectorstore"]
    bm25 = doc_data["bm25"]
    docs = doc_data["docs"]
    
    # Step 1: Retrieval (Hybrid or Vector-only)
    if HYBRID_ENABLED and bm25 is not None:
        # Hybrid: BM25 finds keyword matches, Vector finds semantic matches
        # RRF combines both - chunks in BOTH lists get boosted!
        retrieve_k = INITIAL_K if RERANK_ENABLED else k
        retrieved_docs = hybrid_search(
            question, vectorstore, bm25, docs,
            initial_k=INITIAL_K,
            final_k=retrieve_k
        )
    else:
        # Vector-only
        retrieve_k = INITIAL_K if RERANK_ENABLED else k
        retriever = vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": retrieve_k}
        )
        retrieved_docs = retriever.invoke(question)
    
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

