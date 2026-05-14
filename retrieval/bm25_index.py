"""
BM25 keyword index for sparse retrieval.
Complements dense vector search with exact keyword matching.
"""
from typing import List, Tuple, Dict
from loguru import logger
import math
import re

from rank_bm25 import BM25Okapi

from retrieval.models import TextChunk, RetrievedChunk


class BM25Index:
    """
    In-memory BM25 index over a set of TextChunks.
    Rebuilt when new chunks are added. Suitable for per-session or
    per-document use; for large corpora, persist this.
    """

    def __init__(self):
        self._chunks: List[TextChunk] = []
        self._bm25: BM25Okapi = None
        self._tokenized_corpus: List[List[str]] = []

    def add_chunks(self, chunks: List[TextChunk]) -> None:
        """Add chunks to the index and rebuild BM25."""
        self._chunks.extend(chunks)
        self._tokenized_corpus = [
            _tokenize(c.text) for c in self._chunks
        ]
        if self._tokenized_corpus:
            self._bm25 = BM25Okapi(self._tokenized_corpus)
        logger.debug(f"BM25 index built with {len(self._chunks)} chunks")

    def query(self, query_text: str, top_k: int = 10) -> List[RetrievedChunk]:
        """
        Retrieve top-k chunks by BM25 score.

        Returns:
            List of RetrievedChunk sorted by descending BM25 score.
        """
        if not self._bm25 or not self._chunks:
            return []

        tokens = _tokenize(query_text)
        scores = self._bm25.get_scores(tokens)

        # Normalize BM25 scores to 0–1 range
        max_score = max(scores) if max(scores) > 0 else 1.0
        normalized = scores / max_score

        # Get top-k indices
        top_indices = sorted(
            range(len(normalized)), key=lambda i: normalized[i], reverse=True
        )[:top_k]

        results = []
        for idx in top_indices:
            if normalized[idx] > 0.001:  # filter out near-zero matches
                results.append(RetrievedChunk(
                    chunk=self._chunks[idx],
                    score=float(normalized[idx]),
                    bm25_score=float(normalized[idx]),
                    retrieval_method="bm25",
                ))
        return results

    def size(self) -> int:
        return len(self._chunks)


def _tokenize(text: str) -> List[str]:
    """
    Legal-aware tokenizer: lowercase, split on non-alphanumeric,
    preserve hyphenated legal terms.
    """
    text = text.lower()
    # Preserve common legal abbreviations and hyphenated terms
    tokens = re.findall(r"[a-z0-9]+(?:[-'][a-z0-9]+)*", text)
    # Remove stopwords (lightweight list)
    stopwords = {
        "the", "a", "an", "and", "or", "but", "in", "on", "at", "to",
        "for", "of", "with", "by", "from", "is", "are", "was", "were",
        "be", "been", "has", "have", "had", "do", "does", "did", "this",
        "that", "these", "those", "it", "its", "as",
    }
    return [t for t in tokens if t not in stopwords and len(t) > 1]
