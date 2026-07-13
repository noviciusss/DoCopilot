import os
import time
import pytest
from backend.rag import index_get_pdf, ask_question
from backend.evaluate_local import llm_correctness, llm_relevance, load_questions

# Baselines from previous manual eval
BASE_CORRECTNESS = 0.892
BASE_RELEVANCE = 0.900

# Tolerate up to 8% drop to account for LLM non-determinism
CORRECTNESS_THRESHOLD = BASE_CORRECTNESS - 0.08
RELEVANCE_THRESHOLD = BASE_RELEVANCE - 0.08

@pytest.fixture(scope="module")
def indexed_document():
    """Index the test PDF file to Qdrant once for the entire module test run."""
    pdf_path = os.path.join(os.path.dirname(__file__), "../aws-overview.pdf")
    if not os.path.exists(pdf_path):
        pytest.fail(f"Test PDF not found at: {pdf_path}")
        
    print(f"\nIndexing PDF {pdf_path} for testing...")
    with open(pdf_path, "rb") as f:
        content = f.read()
    
    # Use a specific test collection
    doc_id = index_get_pdf(content, "aws_overview_test.pdf", tenant_id="test_tenant")
    return doc_id

@pytest.fixture(scope="module")
def questions_list():
    """Load golden questions from data.jsonl."""
    jsonl_path = os.path.join(os.path.dirname(__file__), "../data.jsonl")
    if not os.path.exists(jsonl_path):
        pytest.fail(f"Golden dataset not found at: {jsonl_path}")
    
    questions = load_questions(jsonl_path)
    # For CI efficiency and cost control, default to a representative subset of 5 questions
    # unless RUN_FULL_EVAL is explicitly set to true.
    if os.getenv("RUN_FULL_EVAL", "false").lower() != "true":
        print(f"\nRunning fast CI eval (5 of {len(questions)} questions) to save API credits.")
        return questions[:5]
    else:
        print(f"\nRunning full eval suite ({len(questions)} questions).")
        return questions

def test_rag_regression(indexed_document, questions_list):
    """Run RAG pipeline and assert that correctness and relevance haven't regressed."""
    results = []
    
    for i, q in enumerate(questions_list):
        question_text = q["question"]
        expected = q["expected_answer"]
        print(f"\n[{i+1}/{len(questions_list)}] Querying: {question_text[:50]}...")
        
        # Run RAG
        start = time.time()
        predicted, sources = ask_question(question_text, document_id=indexed_document, tenant_id="test_tenant")
        latency = time.time() - start
        
        # Evaluate
        score_correctness = llm_correctness(question_text, expected, predicted)
        score_relevance = llm_relevance(question_text, predicted)
        
        print(f"  Result: Correctness={score_correctness:.2f}, Relevance={score_relevance:.2f}, Latency={latency:.2f}s")
        
        results.append({
            "correctness": score_correctness,
            "relevance": score_relevance,
        })
        
        # Avoid Groq rate limit issues in sequential calls
        time.sleep(3)
        
    avg_correctness = sum(r["correctness"] for r in results) / len(results)
    avg_relevance = sum(r["relevance"] for r in results) / len(results)
    
    print("\n" + "="*50)
    print("EVALUATION METRICS RESULTS")
    print("="*50)
    print(f"Avg Correctness: {avg_correctness:.1%} (Threshold: {CORRECTNESS_THRESHOLD:.1%})")
    print(f"Avg Relevance:   {avg_relevance:.1%} (Threshold: {RELEVANCE_THRESHOLD:.1%})")
    print("="*50)
    
    assert avg_correctness >= CORRECTNESS_THRESHOLD, (
        f"Correctness {avg_correctness:.1%} fell below threshold {CORRECTNESS_THRESHOLD:.1%}"
    )
    assert avg_relevance >= RELEVANCE_THRESHOLD, (
        f"Relevance {avg_relevance:.1%} fell below threshold {RELEVANCE_THRESHOLD:.1%}"
    )
