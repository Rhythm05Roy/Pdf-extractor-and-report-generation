"""
Data models for the retrieval stage.
"""
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class TextChunk:
    """A single chunk of text from a document, ready for embedding."""
    chunk_id: str           # "{doc_id}::chunk::{index}"
    doc_id: str
    source_path: str
    page_number: int
    chunk_index: int
    text: str
    char_start: int = 0
    char_end: int = 0
    metadata: dict = field(default_factory=dict)


@dataclass
class RetrievedChunk:
    """A chunk returned by the retrieval layer, with scoring info."""
    chunk: TextChunk
    score: float                    # hybrid score 0.0–1.0
    dense_score: float = 0.0        # vector similarity
    bm25_score: float = 0.0         # keyword relevance
    retrieval_method: str = "hybrid"

    @property
    def chunk_id(self) -> str:
        return self.chunk.chunk_id

    @property
    def text(self) -> str:
        return self.chunk.text

    @property
    def source_label(self) -> str:
        """Human-readable source label for citations."""
        name = self.chunk.source_path.split("/")[-1]
        return f"{name} (p.{self.chunk.page_number})"
