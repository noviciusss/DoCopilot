import time
import re
from typing import List, Dict, Any

def calculate_recall_at_k(retrieved_chunks: List[str], ground_truth_chunks: List[str], k: int = 5) -> float:
    """Calculates Recall@k: fraction of relevant ground-truth chunks retrieved in top k."""
    if not ground_truth_chunks:
        return 1.0  # Not applicable / empty truth set
    
    top_k = retrieved_chunks[:k]
    hits = 0
    for gt in ground_truth_chunks:
        if any(gt.lower() in chunk.lower() for chunk in top_k):
            hits += 1
    return hits / len(ground_truth_chunks)

def calculate_mrr(retrieved_chunks: List[str], ground_truth_chunks: List[str]) -> float:
    """Calculates Mean Reciprocal Rank (MRR) based on first relevant chunk rank."""
    if not ground_truth_chunks:
        return 1.0
        
    for rank_idx, chunk in enumerate(retrieved_chunks, start=1):
        if any(gt.lower() in chunk.lower() for gt in ground_truth_chunks):
            return 1.0 / rank_idx
    return 0.0

def calculate_citation_precision(answer: str, retrieved_sources: List[str]) -> float:
    """Checks if citation tags (e.g. [c1], [c2]) in generated answer refer to valid sources."""
    citations = re.findall(r"\[c(\d+)\]", answer)
    if not citations:
        return 1.0  # No citations claimed
        
    valid_citations = 0
    for c_idx in citations:
        idx = int(c_idx) - 1
        if 0 <= idx < len(retrieved_sources):
            valid_citations += 1
            
    return valid_citations / len(citations)

def evaluate_groundedness_heuristic(answer: str, retrieved_context: str) -> float:
    """
    Heuristic groundedness check: percentage of key terms in answer present in context.
    (Can be upgraded to LLM-as-judge call if GROQ_API_KEY is available).
    """
    words = [w.lower() for w in re.findall(r"\b\w{4,}\b", answer) if w.lower() not in {"this", "that", "with", "from", "have", "were"}]
    if not words:
        return 1.0
    matches = sum(1 for w in words if w in retrieved_context.lower())
    return round(matches / len(words), 2)
