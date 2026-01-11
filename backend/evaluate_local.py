import os
import json
import time
import dotenv

dotenv.load_dotenv(dotenv.find_dotenv())

from rag import index_get_pdf, ask_question

# Rate limit: wait between requests to avoid 429 errors
RATE_LIMIT_DELAY = 5  # seconds between requests (Gemini free tier: 10/min)


def load_questions(jsonl_path: str) -> list[dict]:
    questions = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                questions.append(json.loads(line))
    return questions


def calculate_correctness(predicted: str, expected: str) -> float:
    expected_words = set(expected.lower().split())
    predicted_words = set(predicted.lower().split())
    
    if not expected_words:
        return 0.0
    
    overlap = len(expected_words & predicted_words)
    return overlap / len(expected_words)


def calculate_relevance(question: str, answer: str) -> float:
    if not answer or len(answer) < 10:
        return 0.0
    
    q_words = set(question.lower().split()) - {"what", "is", "the", "a", "an", "how", "why", "which"}
    a_words = set(answer.lower().split())
    
    overlap = len(q_words & a_words)
    return min(1.0, overlap / max(len(q_words), 1))


def run_local_evaluation():
    # 1. Index PDF
    print("Indexing PDF...")
    with open("../aws-overview.pdf", "rb") as f:
        content = f.read()
    doc_id = index_get_pdf(content, "aws-overview.pdf")
    print(f"Document ID: {doc_id}")
    
    # 2. Load questions
    questions = load_questions("../data.jsonl")
    print(f"Loaded {len(questions)} questions")
    print(f"Rate limit delay: {RATE_LIMIT_DELAY}s between requests")
    print(f"Estimated time: {len(questions) * RATE_LIMIT_DELAY / 60:.1f} minutes\n")
    
    # 3. Evaluate each question
    results = []
    successful = 0
    total_correctness = 0.0
    total_relevance = 0.0
    total_has_sources = 0
    total_latency = 0.0
    
    for i, q in enumerate(questions):
        print(f"[{i+1}/{len(questions)}] {q['question'][:50]}...")
        
        start = time.time()
        try:
            answer, sources = ask_question(q["question"], document_id=doc_id)
            latency = time.time() - start
            
            correctness = calculate_correctness(answer, q["expected_answer"])
            relevance = calculate_relevance(q["question"], answer)
            has_sources = 1.0 if sources else 0.0
            
            results.append({
                "id": q.get("id", ""),
                "question": q["question"],
                "expected": q["expected_answer"],
                "predicted": answer,
                "sources": sources,
                "correctness": correctness,
                "relevance": relevance,
                "has_sources": has_sources,
                "latency": latency,
                "success": True
            })
            
            successful += 1
            total_correctness += correctness
            total_relevance += relevance
            total_has_sources += has_sources
            total_latency += latency
            
            print(f"  ✅ Correctness: {correctness:.2f} | Relevance: {relevance:.2f} | Latency: {latency:.2f}s")
            
        except Exception as e:
            error_msg = str(e)
            print(f"ERROR: {error_msg[:80]}")
            results.append({
                "id": q.get("id", ""),
                "question": q["question"],
                "error": error_msg,
                "success": False
            })
        
        # Rate limiting - wait before next request
        if i < len(questions) - 1:
            print(f"  ⏳ Waiting {RATE_LIMIT_DELAY}s...")
            time.sleep(RATE_LIMIT_DELAY)
    
    # 4. Print summary (only for successful questions)
    print("\n" + "="*60)
    print("EVALUATION SUMMARY")
    print("="*60)
    print(f"Total Questions:     {len(questions)}")
    print(f"Successful:          {successful}/{len(questions)} ({successful/len(questions):.0%})")
    
    if successful > 0:
        print(f"Avg Correctness:     {total_correctness/successful:.2%}")
        print(f"Avg Relevance:       {total_relevance/successful:.2%}")
        print(f"Has Sources Rate:    {total_has_sources/successful:.2%}")
        print(f"Avg Latency:         {total_latency/successful:.2f}s")
    print("="*60)
    
    # 5. Save results
    with open("evaluation_results.json", "w", encoding="utf-8") as f:
        json.dump({
            "summary": {
                "total_questions": len(questions),
                "successful": successful,
                "success_rate": successful / len(questions),
                "avg_correctness": total_correctness / successful if successful else 0,
                "avg_relevance": total_relevance / successful if successful else 0,
                "has_sources_rate": total_has_sources / successful if successful else 0,
                "avg_latency": total_latency / successful if successful else 0
            },
            "results": results
        }, f, indent=2, ensure_ascii=False)
    
    print("\nResults saved to evaluation_results.json")


if __name__ == "__main__":
    run_local_evaluation()