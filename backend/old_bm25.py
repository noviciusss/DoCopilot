# ============================================
# OLD BM25 CODE (COMMENTED OUT)
# Replaced by Qdrant's built-in sparse vectors
# Kept for reference to show previous work
# ============================================

# from rank_bm25 import BM25Okapi
# from langchain_community.vectorstores import FAISS

# def tokenize(text: str) -> List[str]:
#     """
#     Simple tokenizer for BM25.
#     
#     How it works:
#     - Converts text to lowercase
#     - Extracts all word characters using regex
#     - Returns list of tokens
#     
#     Example:
#         "What is EC2?" → ["what", "is", "ec2"]
#     """
#     return re.findall(r'\w+', text.lower())


# @traceable(name="create_bm25_index")
# def create_bm25_index(docs: List[Document]) -> BM25Okapi:
#     """
#     Create BM25 index from document chunks.
#     
#     BM25 (Best Matching 25) is a keyword-based ranking function.
#     It scores documents based on:
#     - Term Frequency (TF): How often query terms appear in a doc
#     - Inverse Document Frequency (IDF): Rare terms get higher weight
#     - Document length normalization
#     
#     Formula (simplified):
#         score(D,Q) = Σ IDF(qi) * (f(qi,D) * (k1+1)) / (f(qi,D) + k1*(1-b+b*|D|/avgdl))
#     
#     Where:
#         - qi = query term
#         - f(qi,D) = term frequency in document D
#         - |D| = document length
#         - avgdl = average document length
#         - k1, b = tuning parameters (typically k1=1.5, b=0.75)
#     
#     Unlike vector search, BM25 does EXACT keyword matching.
#     Good for: Acronyms (EC2, S3), specific terms, exact phrases
#     Bad for: Synonyms, paraphrases, semantic similarity
#     """
#     tokenized_docs = [tokenize(doc.page_content) for doc in docs]
#     bm25 = BM25Okapi(tokenized_docs)
#     logger.info("BM25 index created: %d chunks", len(docs))
#     return bm25


# @traceable(name="bm25_search")
# def bm25_search(
#     query: str, 
#     bm25: BM25Okapi, 
#     docs: List[Document], 
#     top_k: int = 20
# ) -> List[Tuple[Document, float]]:
#     """
#     BM25 keyword search.
#     
#     How it works:
#     1. Tokenize query: "What is EC2?" → ["what", "is", "ec2"]
#     2. For each chunk, calculate BM25 score based on keyword matches
#     3. Return chunks sorted by score (highest first)
#     
#     Returns: List of (Document, score) - finds DIFFERENT chunks than vector!
#     
#     Example:
#         Query: "EC2 pricing"
#         BM25 finds: Chunks containing exact words "EC2" and "pricing"
#         Vector finds: Chunks about "compute costs" (semantic meaning)
#     
#     Why BM25 + Vector together?
#         - BM25 catches: "EC2", "S3", "IAM" (exact acronyms)
#         - Vector catches: "virtual machines" when asking about "EC2"
#         - Combined = best of both worlds
#     """
#     query_tokens = tokenize(query)
#     scores = bm25.get_scores(query_tokens)
#     
#     # Pair docs with scores and sort
#     scored_docs = list(zip(docs, scores))
#     scored_docs.sort(key=lambda x: x[1], reverse=True)
#     
#     logger.info("BM25: top score=%.2f for '%s...'", 
#                 scored_docs[0][1] if scored_docs else 0, 
#                 query[:30])
#     
#     return scored_docs[:top_k]


# @traceable(name="vector_search_faiss")
# def vector_search_faiss(
#     query: str, 
#     vectorstore: FAISS, 
#     top_k: int = 20
# ) -> List[Tuple[Document, float]]:
#     """
#     Vector semantic search using FAISS.
#     
#     How it works:
#     1. Embed query: "What is EC2?" → [0.12, 0.85, ...] (384-dim vector)
#     2. Compare to all chunk embeddings using L2 distance
#     3. Return most similar chunks (lowest distance = most similar)
#     
#     Returns: List of (Document, score)
#     
#     Note: FAISS returns DISTANCE (lower = better)
#           We convert to similarity: 1/(1+distance)
#     """
#     results = vectorstore.similarity_search_with_score(query, k=top_k)
#     # Convert distance to similarity (higher = better)
#     scored_docs = [(doc, 1 / (1 + dist)) for doc, dist in results]
#     
#     logger.info("Vector: top score=%.3f for '%s...'", 
#                 scored_docs[0][1] if scored_docs else 0, 
#                 query[:30])
#     
#     return scored_docs


# @traceable(name="rrf_fusion")
# def rrf_fusion(
#     results_list: List[List[Tuple[Document, float]]], 
#     k: int = 60
# ) -> List[Tuple[Document, float]]:
#     """
#     Reciprocal Rank Fusion - combines rankings from multiple methods.
#     
#     Why RRF instead of score averaging?
#     - BM25 scores: unbounded (can be 0-100+)
#     - Vector scores: 0-1 range
#     - Can't just add them! BM25 would dominate.
#     
#     RRF uses RANKS instead of scores:
#     - RRF_score(doc) = Σ 1/(k + rank) for each method
#     - k=60 (constant) prevents rank 1 from dominating too much
#     
#     Formula:
#         RRF(d) = Σ 1/(k + rank_i(d))
#         where rank_i(d) is document d's rank in result list i
#     
#     Example with k=60:
#         Doc A: BM25 rank=1, Vector rank=2
#         RRF(A) = 1/(60+1) + 1/(60+2) = 0.0164 + 0.0161 = 0.0325
#         
#         Doc B: BM25 rank=5, Vector rank=1
#         RRF(B) = 1/(60+5) + 1/(60+1) = 0.0154 + 0.0164 = 0.0318
#         
#         Doc C: BM25 rank=3, Vector rank=NOT_FOUND
#         RRF(C) = 1/(60+3) + 0 = 0.0159
#     
#     Key insight: Docs in BOTH lists get higher scores!
#     This is why hybrid retrieval works - it boosts docs that are
#     relevant by BOTH keyword match AND semantic similarity.
#     """
#     doc_scores: Dict[int, float] = {}
#     doc_map: Dict[int, Document] = {}
#     
#     for results in results_list:
#         for rank, (doc, _) in enumerate(results, start=1):
#             # Use hash of content as unique ID
#             doc_id = hash(doc.page_content)
#             
#             if doc_id not in doc_map:
#                 doc_map[doc_id] = doc
#                 doc_scores[doc_id] = 0.0
#             
#             # RRF formula: 1/(k + rank)
#             doc_scores[doc_id] += 1.0 / (k + rank)
#     
#     # Sort by RRF score (highest first)
#     sorted_items = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)
#     
#     logger.info("RRF: %d unique chunks from %d methods", len(sorted_items), len(results_list))
#     
#     return [(doc_map[doc_id], score) for doc_id, score in sorted_items]


# @traceable(name="hybrid_search_manual")
# def hybrid_search_manual(
#     query: str,
#     vectorstore: FAISS,
#     bm25: BM25Okapi,
#     docs: List[Document],
#     initial_k: int = 20,
#     final_k: int = 5
# ) -> List[Document]:
#     """
#     Manual Hybrid search: BM25 + Vector + RRF fusion.
#     
#     THIS IS NOW REPLACED BY QDRANT'S BUILT-IN HYBRID SEARCH!
#     Kept for reference to show the manual implementation.
#     
#     Pipeline:
#     1. BM25 search → Top 20 by keyword match
#     2. Vector search → Top 20 by semantic similarity  
#     3. RRF fusion → Combine both, boost docs in both lists
#     4. Return top K
#     """
#     # BM25 finds chunks by keyword match
#     bm25_results = bm25_search(query, bm25, docs, top_k=initial_k)
#     
#     # Vector finds chunks by semantic similarity
#     vector_results = vector_search_faiss(query, vectorstore, top_k=initial_k)
#     
#     # RRF combines both - chunks in both lists score higher
#     fused = rrf_fusion([bm25_results, vector_results], k=60)
#     
#     return [doc for doc, _ in fused[:final_k]]


# def create_vector_store_faiss(docs: List[Document]) -> FAISS:
#     """
#     Create FAISS vector store.
#     
#     FAISS (Facebook AI Similarity Search) is an in-memory vector index.
#     
#     Limitations (why we switched to Qdrant):
#     - In-memory only (data lost on restart)
#     - No built-in hybrid search (need separate BM25)
#     - No metadata filtering
#     - Single machine only
#     """
#     if not docs:
#         raise ValueError("No documents provided")
#     t0 = time.time()
#     vectorstore = FAISS.from_documents(docs, _embeddings)
#     logger.info("FAISS store created in %.2fs (%d docs)", time.time() - t0, len(docs))
#     return vectorstore


# ============================================
# END OF OLD BM25 CODE
# ============================================


# ============================================
# QDRANT HYBRID SEARCH (REPLACES ALL BM25 CODE ABOVE)
# ============================================