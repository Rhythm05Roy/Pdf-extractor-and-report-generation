"""
ChromaDB vector store interface.
Handles persistent storage and retrieval of document chunk embeddings.
"""
import json
from typing import List, Optional, Dict, Any
from pathlib import Path
from loguru import logger

import chromadb
from chromadb.config import Settings

from retrieval.models import TextChunk, RetrievedChunk
import config


_client: chromadb.PersistentClient = None
_collection: chromadb.Collection = None

COLLECTION_NAME = "legal_documents"


def _get_collection() -> chromadb.Collection:
    """Lazy-init ChromaDB client and collection."""
    global _client, _collection
    if _collection is not None:
        return _collection

    Path(config.CHROMA_DB_PATH).mkdir(parents=True, exist_ok=True)
    _client = chromadb.PersistentClient(
        path=config.CHROMA_DB_PATH,
        settings=Settings(anonymized_telemetry=False),
    )
    _collection = _client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )
    logger.info(
        f"ChromaDB collection '{COLLECTION_NAME}' ready "
        f"({_collection.count()} existing chunks)"
    )
    return _collection


def add_chunks(chunks: List[TextChunk], embeddings: List[List[float]]) -> None:
    """
    Add a batch of chunks with their precomputed embeddings to ChromaDB.
    Skips chunks that are already stored (idempotent upsert).
    """
    if not chunks:
        return

    collection = _get_collection()
    ids = [c.chunk_id for c in chunks]
    documents = [c.text for c in chunks]
    metadatas = [
        {
            "doc_id": c.doc_id,
            "source_path": c.source_path,
            "page_number": c.page_number,
            "chunk_index": c.chunk_index,
        }
        for c in chunks
    ]

    # ChromaDB upsert is idempotent
    collection.upsert(
        ids=ids,
        embeddings=embeddings,
        documents=documents,
        metadatas=metadatas,
    )
    logger.debug(f"Upserted {len(chunks)} chunks into ChromaDB")


def query(
    query_embedding: List[float],
    top_k: int = None,
    doc_id_filter: Optional[str] = None,
) -> List[RetrievedChunk]:
    """
    Retrieve top-k chunks by vector similarity.

    Args:
        query_embedding: Dense vector for the query.
        top_k:           Number of results (defaults to config.RETRIEVAL_TOP_K).
        doc_id_filter:   If set, restrict results to a specific document.

    Returns:
        List of RetrievedChunk ordered by descending similarity.
    """
    collection = _get_collection()
    top_k = top_k or config.RETRIEVAL_TOP_K

    where = {"doc_id": doc_id_filter} if doc_id_filter else None

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=min(top_k, max(collection.count(), 1)),
        where=where,
        include=["documents", "metadatas", "distances"],
    )

    retrieved: List[RetrievedChunk] = []
    if not results["ids"] or not results["ids"][0]:
        return retrieved

    for i, chunk_id in enumerate(results["ids"][0]):
        doc_text = results["documents"][0][i]
        meta = results["metadatas"][0][i]
        # ChromaDB returns L2 distance; convert to similarity (cosine space → 1 - dist/2)
        distance = results["distances"][0][i]
        similarity = max(0.0, 1.0 - distance / 2.0)

        chunk = TextChunk(
            chunk_id=chunk_id,
            doc_id=meta["doc_id"],
            source_path=meta["source_path"],
            page_number=int(meta["page_number"]),
            chunk_index=int(meta["chunk_index"]),
            text=doc_text,
        )
        retrieved.append(RetrievedChunk(
            chunk=chunk,
            score=similarity,
            dense_score=similarity,
            retrieval_method="dense",
        ))

    return retrieved


def get_all_chunks_for_doc(doc_id: str) -> List[TextChunk]:
    """Return every stored chunk for a given document ID."""
    collection = _get_collection()
    results = collection.get(
        where={"doc_id": doc_id},
        include=["documents", "metadatas"],
    )
    chunks = []
    for i, chunk_id in enumerate(results["ids"]):
        meta = results["metadatas"][i]
        chunks.append(TextChunk(
            chunk_id=chunk_id,
            doc_id=meta["doc_id"],
            source_path=meta["source_path"],
            page_number=int(meta["page_number"]),
            chunk_index=int(meta["chunk_index"]),
            text=results["documents"][i],
        ))
    return chunks


def delete_document(doc_id: str) -> int:
    """Remove all chunks for a document. Returns count of deleted chunks."""
    collection = _get_collection()
    existing = collection.get(where={"doc_id": doc_id})
    ids_to_delete = existing["ids"]
    if ids_to_delete:
        collection.delete(ids=ids_to_delete)
    logger.info(f"Deleted {len(ids_to_delete)} chunks for doc {doc_id[:8]}")
    return len(ids_to_delete)


def collection_size() -> int:
    """Return total number of stored chunks."""
    return _get_collection().count()
