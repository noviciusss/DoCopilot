import os
import tempfile
import time
from pathlib import Path
from typing import Dict, List, Optional
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

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

langsmith_client: Optional[Client] = None
if os.getenv("LANGCHAIN_API_KEY"):
    langsmith_client = Client()
    logger.info("Langsmith client initialized")

chunk_size = 2000
chunk_overlap = 400

splitter = RecursiveCharacterTextSplitter(
    chunk_size=chunk_size,
    chunk_overlap=chunk_overlap
)

document_cache: Dict[str, FAISS] = {}
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
_embeddings.embed_query("warmup")  # Warm up
logger.info("Embedding model ready in %.2fs", time.time() - _load_start)


def _get_embedings() -> HuggingFaceEmbeddings:
    return _embeddings


@traceable(name="load_pdf")
def load_pdf_from_bytes(content: bytes, filename: str) -> List[Document]:
    """Load PDF from bytes using a temp file."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(content)
        tmp_path = tmp.name
    try:
        loader = PyPDFLoader(tmp_path)
        pages = loader.load()
        # Set source metadata to the original filename
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

@traceable(name="create_vector_store")
def create_vector_store(docs: List[Document]) -> FAISS:
    if not docs:
        raise ValueError("No documents provided")
    t0 = time.time()
    vectorstore = FAISS.from_documents(docs, _embeddings)
    logger.info("Vector store created in %.2fs (%d docs)", time.time() - t0, len(docs))
    return vectorstore


def store_vector_cache(vectors: FAISS) -> str:
    global current_document_id, document_cache
    document_cache.clear()
    document_id = str(uuid4())
    document_cache[document_id] = vectors
    current_document_id = document_id
    logger.info("Cached document: %s", document_id)
    return document_id


@traceable(name="index_get_pdf", tags=["indexing"])
def index_get_pdf(content: bytes, filename: str) -> str:
    """Index PDF from bytes content."""
    t0 = time.time()
    docs = load_pdf_from_bytes(content, filename)
    vectors = create_vector_store(docs)
    doc_id = store_vector_cache(vectors)
    logger.info("Total PDF indexing: %.2fs", time.time() - t0)
    return doc_id


@traceable(name="index_get_txt", tags=["indexing"])
def index_get_txt(text: str, filename: str) -> str:
    """Index text content directly."""
    docs = load_text_chunks(text, filename)
    vectors = create_vector_store(docs)
    return store_vector_cache(vectors)


@traceable(name="index_get_plain_text", tags=["indexing"])
def index_get_plain_text(raw_text: str) -> str:
    docs = plain_text_chunks(raw_text)
    vectors = create_vector_store(docs)
    return store_vector_cache(vectors)


def get_vector_store(document_id: Optional[str] = None) -> FAISS:
    target_id = document_id or current_document_id
    if not target_id or target_id not in document_cache:
        raise ValueError("No vector store found for the given document ID")
    return document_cache[target_id]


@traceable(
    name="ask_question",
    metadata={"version": "1.0", "model": "gemini-2.5-flash-lite"},
    tags=["rag", "qa"]
)
def ask_question(question: str, *, document_id: Optional[str] = None, k: int = 5) -> tuple[str, list[str]]:
    vectorstore = get_vector_store(document_id)
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": k}
    )
    
    docs = retriever.invoke(question)
    
    context = "\n\n".join(
        f"[c{i+1}] {d.page_content}\nMETADATA: {d.metadata}"
        for i, d in enumerate(docs)
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
    llm = ChatGoogleGenerativeAI(model='gemini-2.5-flash-lite', temperature=0)
    response = llm.invoke(final_prompt)

    answer = getattr(response, "text", None) or getattr(response, "content", str(response))
    sources = list(set(doc.metadata.get("source", "") for doc in docs))

    return answer, sources

