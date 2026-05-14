"""
Sentence-transformer embedding wrapper.
Produces dense vectors for semantic similarity search.
"""
from typing import List
from loguru import logger

import numpy as np
from sentence_transformers import SentenceTransformer

import config

_model: SentenceTransformer = None


def get_embedding_model() -> SentenceTransformer:
    """Lazy-load and cache the embedding model."""
    global _model
    if _model is None:
        logger.info(f"Loading embedding model: {config.EMBEDDING_MODEL}")
        _model = SentenceTransformer(config.EMBEDDING_MODEL)
        logger.info("Embedding model loaded")
    return _model


def embed_texts(texts: List[str], batch_size: int = 64, show_progress: bool = False) -> np.ndarray:
    """
    Embed a list of strings into dense vectors.

    Returns:
        numpy array of shape (len(texts), embedding_dim)
    """
    model = get_embedding_model()
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=show_progress,
        normalize_embeddings=True,   # L2-normalize for cosine similarity
        convert_to_numpy=True,
    )
    return embeddings


def embed_single(text: str) -> np.ndarray:
    """Embed a single string. Convenience wrapper."""
    return embed_texts([text])[0]


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two normalized vectors."""
    return float(np.dot(a, b))
