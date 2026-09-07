import re
import logging
from functools import lru_cache
from typing import List, Tuple, Optional, Dict
from sentence_transformers import CrossEncoder

from config.settings import (
    DEFAULT_K_RESULTS,
    UNIFIED_COLLECTION_NAME,
    RERANK_RELEVANCE_THRESHOLD
)
from services.vector_service import get_chroma_client, get_embedding_model

logger = logging.getLogger(__name__)


def is_document_query(query: str) -> bool:
    """Keyword-based check: is this query about a specific document?"""
    if not isinstance(query, str):
        logger.warning(f"is_document_query received non-string: {type(query)}")
        return False

    doc_keywords = [
        'pdf', 'document', 'file', 'chapter', 'page', 'section',
        'this doc', 'the doc', 'this pdf', 'the pdf', 'this file'
    ]
    is_doc = any(kw in query.lower() for kw in doc_keywords)
    logger.info("📄 Document-specific query" if is_doc else "💬 General query")
    return is_doc


def reciprocal_rank_fusion(dense_results: List[dict], sparse_results: List[dict], k: int = 60) -> List[Tuple[dict, float]]:
    """
    Merge dense and sparse search results using Reciprocal Rank Fusion (RRF).
    """
    scores = {}
    
    for rank, item in enumerate(dense_results, 1):
        item_id = item['id']
        scores[item_id] = scores.get(item_id, 0.0) + 1.0 / (k + rank)
        
    for rank, item in enumerate(sparse_results, 1):
        item_id = item['id']
        scores[item_id] = scores.get(item_id, 0.0) + 1.0 / (k + rank)
        
    item_map = {item['id']: item for item in dense_results + sparse_results}
    sorted_ids = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)
    
    merged_items = []
    for item_id in sorted_ids:
        merged_items.append((item_map[item_id], scores[item_id]))
        
    return merged_items


@lru_cache(maxsize=1)
def get_reranker_model():
    """Lazy load cross-encoder reranker with caching."""
    logger.info("Loading cross-encoder reranker model: cross-encoder/ms-marco-MiniLM-L-6-v2")
    return CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")


def retrieve_context(query: str, doc_name: Optional[str] = None, k: int = DEFAULT_K_RESULTS) -> Tuple[List[str], List[float]]:
    """
    Hybrid retrieval pipeline (Dense + BM25) with CrossEncoder reranking.
    """
    try:
        client = get_chroma_client()
        collection = client.get_or_create_collection(name=UNIFIED_COLLECTION_NAME)
    except Exception as e:
        logger.error(f"Error accessing database: {str(e)}")
        return [], []

    try:
        # 1. Dense Retrieval (ChromaDB)
        embed_model = get_embedding_model()
        query_emb = embed_model.encode([f"query: {query}"]).tolist()
        
        top_n = max(k * 4, 20)
        
        if doc_name and doc_name != "-- Search All Documents --":
            results = collection.query(
                query_embeddings=query_emb,
                n_results=top_n,
                where={"source": doc_name},
                include=['documents', 'distances', 'metadatas']
            )
        else:
            results = collection.query(
                query_embeddings=query_emb,
                n_results=top_n,
                include=['documents', 'distances', 'metadatas']
            )
            
        dense_results = []
        if results.get('documents') and results['documents'][0]:
            for i in range(len(results['documents'][0])):
                dense_results.append({
                    'id': results['ids'][0][i],
                    'document': results['documents'][0][i],
                    'metadata': results['metadatas'][0][i],
                    'score': results['distances'][0][i]
                })

        # 2. Sparse Retrieval (BM25)
        if doc_name and doc_name != "-- Search All Documents --":
            candidates = collection.get(where={"source": doc_name}, include=['documents', 'metadatas'])
        else:
            candidates = collection.get(include=['documents', 'metadatas'])
            
        sparse_results = []
        if candidates and candidates.get('documents'):
            from rank_bm25 import BM25Okapi
            
            def tokenize(text):
                return re.findall(r'\w+', text.lower())
                
            tokenized_corpus = [tokenize(doc) for doc in candidates['documents']]
            if tokenized_corpus:
                bm25 = BM25Okapi(tokenized_corpus)
                tokenized_query = tokenize(query)
                bm25_scores = bm25.get_scores(tokenized_query)
                
                for idx, score in enumerate(bm25_scores):
                    if score > 0:
                        sparse_results.append({
                            'id': candidates['ids'][idx],
                            'document': candidates['documents'][idx],
                            'metadata': candidates['metadatas'][idx],
                            'score': float(score)
                        })
                sparse_results.sort(key=lambda x: x['score'], reverse=True)

        # 3. Reciprocal Rank Fusion (RRF)
        merged = reciprocal_rank_fusion(dense_results, sparse_results[:20])
        
        if not merged:
            return [], []

        # 4. CrossEncoder Reranking
        merged_top = merged[:20]
        try:
            reranker = get_reranker_model()
            pairs = [(query, item['document']) for item, _ in merged_top]
            rerank_scores = reranker.predict(pairs)
            
            reranked = []
            for idx, score in enumerate(rerank_scores):
                item = merged_top[idx][0]
                reranked.append((item, float(score)))
                
            reranked.sort(key=lambda x: x[1], reverse=True)
        except Exception as e:
            logger.error(f"Error during reranking: {e}. Falling back to RRF ordering.")
            reranked = [(item, rrf_score) for item, rrf_score in merged_top]

        # 5. Relevance Filtering & Top-K Extraction
        filtered_chunks = []
        filtered_scores = []
        
        for item, score in reranked:
            if score >= RERANK_RELEVANCE_THRESHOLD:
                filtered_chunks.append(item['document'])
                filtered_scores.append(score)
                
        if not filtered_chunks and reranked:
            best_item, best_score = reranked[0]
            logger.warning(f"All chunks below rerank threshold ({RERANK_RELEVANCE_THRESHOLD}). Returning best match (score={best_score:.3f}).")
            filtered_chunks = [best_item['document']]
            filtered_scores = [best_score]

        return filtered_chunks[:k], filtered_scores[:k]

    except Exception as e:
        logger.error(f"Error in retrieve_context: {str(e)}")
        # Dense fallback
        try:
            embed_model = get_embedding_model()
            query_emb = embed_model.encode([f"query: {query}"]).tolist()
            if doc_name and doc_name != "-- Search All Documents --":
                results = collection.query(query_embeddings=query_emb, n_results=k, where={"source": doc_name})
            else:
                results = collection.query(query_embeddings=query_emb, n_results=k)
            
            if results.get('documents') and results['documents'][0]:
                return results['documents'][0], [0.0] * len(results['documents'][0])
        except Exception as fb_err:
            logger.error(f"Fallback dense query failed: {fb_err}")
        return [], []
