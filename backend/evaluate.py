import os
import json
import httpx
import dotenv

# Load environment variables FIRST
dotenv.load_dotenv(dotenv.find_dotenv())

from langsmith import Client
from langsmith.evaluation import evaluate
from langsmith.schemas import Run, Example

# Verify API key is loaded
api_key = os.getenv("LANGSMITH_API_KEY")
if not api_key:
    raise ValueError("LANGSMITH_API_KEY not found in environment. Check your .env file.")
print(f"Using LangSmith API key: {api_key[:10]}...")

# Initialize LangSmith client
client = Client()

BACKEND_URL = "http://localhost:8000"
DOCUMENT_ID = None

TIMEOUT = httpx.Timeout(120.0, connect=10.0)


def upload_pdf_and_get_id(pdf_path: str) -> str:
    """Upload the PDF and return document_id."""
    with open(pdf_path, "rb") as f:
        response = httpx.post(
            f"{BACKEND_URL}/upload",
            files={"pdf_file": ("aws-overview.pdf", f, "application/pdf")},
            timeout=TIMEOUT
        )
    response.raise_for_status()
    return response.json()["document_id"]


def target(inputs: dict) -> dict:
    """Call your RAG API and return the answer."""
    response = httpx.post(
        f"{BACKEND_URL}/chat",
        json={"document_id": DOCUMENT_ID, "question": inputs["question"]},
        timeout=TIMEOUT
    )
    response.raise_for_status()
    data = response.json()
    return {"answer": data.get("answer", ""), "sources": data.get("sources", [])}


def correctness(run: Run, example: Example) -> dict:
    """Check if expected answer keywords appear in response."""
    predicted = run.outputs.get("answer", "").lower()
    expected = example.outputs.get("expected_answer", "").lower()
    
    expected_words = set(expected.split())
    predicted_words = set(predicted.split())
    
    if not expected_words:
        return {"key": "correctness", "score": 0.0}
    
    overlap = len(expected_words & predicted_words)
    score = overlap / len(expected_words)
    return {"key": "correctness", "score": score}


def answer_relevance(run: Run, example: Example) -> dict:
    """Check if answer addresses the question."""
    answer = run.outputs.get("answer", "")
    question = example.inputs.get("question", "")
    
    if not answer or len(answer) < 10:
        return {"key": "relevance", "score": 0.0}
    
    q_words = set(question.lower().split()) - {"what", "is", "the", "a", "an", "how", "why", "which"}
    a_words = set(answer.lower().split())
    
    overlap = len(q_words & a_words)
    score = min(1.0, overlap / max(len(q_words), 1))
    return {"key": "relevance", "score": score}


def has_sources(run: Run, example: Example) -> dict:
    """Check if sources were returned."""
    sources = run.outputs.get("sources", [])
    return {"key": "has_sources", "score": 1.0 if sources else 0.0}


def create_or_get_dataset(dataset_name: str) -> str:
    """Create dataset with examples, or verify existing one has examples."""
    
    # Try to read existing dataset
    try:
        dataset = client.read_dataset(dataset_name=dataset_name)
        examples = list(client.list_examples(dataset_id=dataset.id))
        
        if len(examples) > 0:
            print(f"Using existing dataset '{dataset_name}' with {len(examples)} examples")
            return dataset_name
        else:
            # Dataset exists but empty - delete and recreate
            print(f"Dataset '{dataset_name}' is empty. Deleting and recreating...")
            client.delete_dataset(dataset_id=dataset.id)
    except Exception as e:
        print(f"Dataset not found, creating new: {e}")
    
    # Create new dataset
    dataset = client.create_dataset(dataset_name=dataset_name)
    print(f"Created dataset: {dataset_name}")
    
    # Add examples from JSONL
    count = 0
    with open("../data.jsonl", "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            client.create_example(
                inputs={"question": item["question"]},
                outputs={"expected_answer": item["expected_answer"]},
                metadata={
                    "id": item.get("id", ""),
                    "type": item.get("type", ""),
                    "tags": item.get("tags", [])
                },
                dataset_id=dataset.id
            )
            count += 1
    
    print(f"Added {count} examples to dataset")
    return dataset_name


def run_evaluation():
    global DOCUMENT_ID
    
    # 1. Upload PDF first
    print("Uploading PDF (this may take a minute)...")
    DOCUMENT_ID = upload_pdf_and_get_id("../aws-overview.pdf")
    print(f"Document ID: {DOCUMENT_ID}")
    
    # 2. Create or get dataset with examples
    dataset_name = create_or_get_dataset("aws-overview-qa")
    
    # 3. Run evaluation
    evaluators = [correctness, answer_relevance, has_sources]
    
    results = evaluate(
        target,
        data=dataset_name,
        evaluators=evaluators,
        experiment_prefix="rag-eval",
        metadata={"model": "gemini", "document": "aws-overview.pdf"}
    )
    
    print("\n=== Evaluation Results ===")
    print(results)


if __name__ == "__main__":
    run_evaluation()
