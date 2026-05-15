"""
Hybrid retriever: fuses dense (ChromaDB) and sparse (BM25) results.
Uses Reciprocal Rank Fusion (RRF) for score merging.
"""
from typing import List, Optional, Dict
from loguru import logger

from retrieval.models import TextChunk, RetrievedChunk
from retrieval.bm25_index import BM25Index
from retrieval import vector_store, embedder
import config


# RRF constant (standard value is 60)
RRF_K = 60


class HybridRetriever:
    """
    Combines dense vector search (ChromaDB) with BM25 keyword search.
    Chunks must be indexed via index_chunks() before querying.
    """

    def __init__(self):
        self._bm25 = BM25Index()
        self._indexed_doc_ids: set = set()

    def index_chunks(self, chunks: List[TextChunk]) -> None:
        """
        Add chunks to both the vector store and the BM25 index.
        Embeddings are computed and stored in ChromaDB.
        """
        if not chunks:
            return

        logger.info(f"Indexing {len(chunks)} chunks...")

        # Dense: embed and store in ChromaDB
        texts = [c.text for c in chunks]
        embeddings = embedder.embed_texts(texts, show_progress=len(texts) > 20)
        vector_store.add_chunks(chunks, embeddings.tolist())

        # Sparse: add to BM25
        self._bm25.add_chunks(chunks)

        doc_ids = {c.doc_id for c in chunks}
        self._indexed_doc_ids.update(doc_ids)
        logger.info(f"Indexed {len(chunks)} chunks for {len(doc_ids)} document(s)")

    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        doc_id_filter: Optional[str] = None,
    ) -> List[RetrievedChunk]:
        """
        Retrieve relevant chunks for a query using hybrid search.

        Args:
            query:          Natural language query or summary for retrieval.
            top_k:          Number of results to return.
            doc_id_filter:  Restrict to a specific document.

        Returns:
            List of RetrievedChunk sorted by hybrid score (descending).
        """
        top_k = top_k or config.RETRIEVAL_TOP_K
        fetch_k = top_k * 2  # fetch more from each source, then merge

        query_embedding = embedder.embed_single(query).tolist()
        dense_results = vector_store.query(
            query_embedding=query_embedding,
            top_k=fetch_k,
            doc_id_filter=doc_id_filter,
        )

        bm25_results = self._bm25.query(query, top_k=fetch_k)

        merged = _reciprocal_rank_fusion(dense_results, bm25_results, k=RRF_K)

        # Annotate with component scores
        dense_map: Dict[str, float] = {r.chunk_id: r.dense_score for r in dense_results}
        bm25_map: Dict[str, float] = {r.chunk_id: r.bm25_score for r in bm25_results}

        for r in merged:
            r.dense_score = dense_map.get(r.chunk_id, 0.0)
            r.bm25_score = bm25_map.get(r.chunk_id, 0.0)
            r.retrieval_method = "hybrid"

        logger.debug(
            f"Retrieved {len(merged[:top_k])} hybrid chunks "
            f"(dense={len(dense_results)}, bm25={len(bm25_results)})"
        )
        return merged[:top_k]

    def is_indexed(self, doc_id: str) -> bool:
        return doc_id in self._indexed_doc_ids


def _reciprocal_rank_fusion(
    dense: List[RetrievedChunk],
    sparse: List[RetrievedChunk],
    k: int = 60,
) -> List[RetrievedChunk]:
    """
    Merge two ranked lists using Reciprocal Rank Fusion.
    RRF score = sum(1 / (k + rank_i)) across all lists.
    """
    scores: Dict[str, float] = {}
    chunk_map: Dict[str, RetrievedChunk] = {}

    for rank, result in enumerate(dense):
        cid = result.chunk_id
        scores[cid] = scores.get(cid, 0.0) + config.DENSE_WEIGHT / (k + rank + 1)
        chunk_map[cid] = result

    for rank, result in enumerate(sparse):
        cid = result.chunk_id
        scores[cid] = scores.get(cid, 0.0) + config.BM25_WEIGHT / (k + rank + 1)
        chunk_map[cid] = result

    # Sort by RRF score
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)

    merged = []
    for cid, rrf_score in ranked:
        result = chunk_map[cid]
        result.score = rrf_score
        merged.append(result)

    return merged


# Module-level singleton retriever
_retriever: Optional[HybridRetriever] = None


def get_retriever() -> HybridRetriever:
    """Return the shared retriever instance (create if needed)."""
    global _retriever
    if _retriever is None:
        _retriever = HybridRetriever()
    return _retriever
