import os
import json
import time
import dotenv

dotenv.load_dotenv(dotenv.find_dotenv())

from langchain_groq import ChatGroq
from rag import index_get_pdf, ask_question

# LLM for evaluation ///  We can use embeddings similarity too its faster and no extra calls but less accurate
eval_llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0)


def llm_correctness(question: str, expected: str, predicted: str) -> float:
    """Use LLM to judge if answer is correct."""
    prompt = f"""You are an evaluation judge. Score how correct the predicted answer is compared to the expected answer.

Question: {question}
Expected Answer: {expected}
Predicted Answer: {predicted}

Score from 0.0 to 1.0:
- 1.0 = Completely correct, covers all key points
- 0.7 = Mostly correct, minor omissions
- 0.5 = Partially correct
- 0.3 = Has some relevant info but mostly wrong
- 0.0 = Completely wrong or irrelevant

Return ONLY a number between 0.0 and 1.0, nothing else."""

    try:
        response = eval_llm.invoke(prompt)
        score = float(response.content.strip())
        return min(1.0, max(0.0, score))
    except:
        return 0.5  # fallback


def llm_relevance(question: str, answer: str) -> float:
    """Use LLM to judge if answer is relevant to question."""
    prompt = f"""You are an evaluation judge. Score how relevant the answer is to the question.

Question: {question}
Answer: {answer}

Score from 0.0 to 1.0:
- 1.0 = Directly answers the question
- 0.7 = Mostly relevant with some tangential info
- 0.5 = Partially relevant
- 0.3 = Loosely related
- 0.0 = Completely irrelevant

Return ONLY a number between 0.0 and 1.0, nothing else."""

    try:
        response = eval_llm.invoke(prompt)
        score = float(response.content.strip())
        return min(1.0, max(0.0, score))
    except:
        return 0.5


def llm_faithfulness(answer: str, sources: list[str], context: str) -> float:
    """Check if answer is grounded in retrieved context."""
    prompt = f"""You are an evaluation judge. Score how well the answer is supported by the context.

Context: {context[:2000]}
Answer: {answer}

Score from 0.0 to 1.0:
- 1.0 = Every claim is directly supported by context
- 0.7 = Most claims supported, minor additions
- 0.5 = Some claims supported, some not
- 0.3 = Few claims supported
- 0.0 = Answer contradicts or ignores context

Return ONLY a number between 0.0 and 1.0, nothing else."""

    try:
        response = eval_llm.invoke(prompt)
        score = float(response.content.strip())
        return min(1.0, max(0.0, score))
    except:
        return 0.5


# Keep keyword-based as backup/comparison
def keyword_correctness(predicted: str, expected: str) -> float:
    expected_words = set(expected.lower().split())
    predicted_words = set(predicted.lower().split())
    if not expected_words:
        return 0.0
    overlap = len(expected_words & predicted_words)
    return overlap / len(expected_words)


def keyword_relevance(question: str, answer: str) -> float:
    if not answer or len(answer) < 10:
        return 0.0
    q_words = set(question.lower().split()) - {"what", "is", "the", "a", "an", "how", "why", "which"}
    a_words = set(answer.lower().split())
    overlap = len(q_words & a_words)
    return min(1.0, overlap / max(len(q_words), 1))


def load_questions(jsonl_path: str) -> list[dict]:
    questions = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                questions.append(json.loads(line))
    return questions


def run_local_evaluation():
    # 1. Index PDF
    print("Indexing PDF...")
    with open("../aws-overview.pdf", "rb") as f:
        content = f.read()
    doc_id = index_get_pdf(content, "aws-overview.pdf")
    print(f"Document ID: {doc_id}")
    
    # 2. Load questions
    questions = load_questions("../data.jsonl")
    print(f"Loaded {len(questions)} questions\n")
    
    # 3. Evaluate
    results = []
    totals = {
        "llm_correctness": 0.0,
        "llm_relevance": 0.0,
        "keyword_correctness": 0.0,
        "keyword_relevance": 0.0,
        "has_sources": 0,
        "latency": 0.0
    }
    successful = 0
    
    for i, q in enumerate(questions):
        print(f"[{i+1}/{len(questions)}] {q['question'][:50]}...")
        
        start = time.time()
        try:
            answer, sources = ask_question(q["question"], document_id=doc_id)
            latency = time.time() - start
            
            # LLM-based scores
            llm_corr = llm_correctness(q["question"], q["expected_answer"], answer)
            llm_rel = llm_relevance(q["question"], answer)
            
            # Keyword-based scores (for comparison)
            kw_corr = keyword_correctness(answer, q["expected_answer"])
            kw_rel = keyword_relevance(q["question"], answer)
            
            has_sources = 1.0 if sources else 0.0
            
            results.append({
                "id": q.get("id", ""),
                "question": q["question"],
                "expected": q["expected_answer"],
                "predicted": answer,
                "sources": sources,
                "llm_correctness": llm_corr,
                "llm_relevance": llm_rel,
                "keyword_correctness": kw_corr,
                "keyword_relevance": kw_rel,
                "has_sources": has_sources,
                "latency": latency,
                "success": True
            })
            
            successful += 1
            totals["llm_correctness"] += llm_corr
            totals["llm_relevance"] += llm_rel
            totals["keyword_correctness"] += kw_corr
            totals["keyword_relevance"] += kw_rel
            totals["has_sources"] += has_sources
            totals["latency"] += latency
            
            print(f"  ✅ LLM: {llm_corr:.0%} corr, {llm_rel:.0%} rel | KW: {kw_corr:.0%} corr, {kw_rel:.0%} rel")
            
            # Rate limiting for Groq
            time.sleep(6)
            
        except Exception as e:
            print(f"  ❌ ERROR: {str(e)[:80]}")
            results.append({
                "id": q.get("id", ""),
                "question": q["question"],
                "error": str(e),
                "success": False
            })
    
    # 4. Summary
    n = successful if successful > 0 else 1
    print("\n" + "="*60)
    print("EVALUATION SUMMARY")
    print("="*60)
    print(f"Total Questions:      {len(questions)}")
    print(f"Successful:           {successful}/{len(questions)}")
    print(f"\n--- LLM-Based Scores (Semantic) ---")
    print(f"Avg Correctness:      {totals['llm_correctness']/n:.1%}")
    print(f"Avg Relevance:        {totals['llm_relevance']/n:.1%}")
    print(f"\n--- Keyword-Based Scores (Baseline) ---")
    print(f"Avg Correctness:      {totals['keyword_correctness']/n:.1%}")
    print(f"Avg Relevance:        {totals['keyword_relevance']/n:.1%}")
    print(f"\n--- Other Metrics ---")
    print(f"Has Sources Rate:     {totals['has_sources']/n:.1%}")
    print(f"Avg Latency:          {totals['latency']/n:.2f}s")
    print("="*60)
    
    # 5. Save
    with open("evaluation_results.json", "w", encoding="utf-8") as f:
        json.dump({
            "summary": {
                "total_questions": len(questions),
                "successful": successful,
                "llm_correctness": totals["llm_correctness"] / n,
                "llm_relevance": totals["llm_relevance"] / n,
                "keyword_correctness": totals["keyword_correctness"] / n,
                "keyword_relevance": totals["keyword_relevance"] / n,
                "has_sources_rate": totals["has_sources"] / n,
                "avg_latency": totals["latency"] / n
            },
            "results": results
        }, f, indent=2, ensure_ascii=False)
    
    print("\nResults saved to evaluation_results.json")


if __name__ == "__main__":
    run_local_evaluation()