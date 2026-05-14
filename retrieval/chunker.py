"""
Sliding-window text chunker.
Splits document text into overlapping chunks while preserving page metadata.
"""
import re
from typing import List
from loguru import logger

from retrieval.models import TextChunk
from ingestion.models import ExtractedDocument
import config


def chunk_document(extracted_doc: ExtractedDocument) -> List[TextChunk]:
    """
    Chunk an ExtractedDocument into TextChunks ready for embedding.

    Strategy:
    - Split by sentences first (respects boundaries)
    - Group sentences into chunks of ~CHUNK_SIZE chars
    - Overlap last CHUNK_OVERLAP chars of previous chunk into next

    Returns:
        List of TextChunk objects with page number, position, and text.
    """
    chunks: List[TextChunk] = []
    chunk_index = 0

    for page in extracted_doc.pages:
        if not page.raw_text.strip():
            continue

        page_chunks = _chunk_text(
            text=page.raw_text,
            doc_id=extracted_doc.doc_id,
            source_path=extracted_doc.source_path,
            page_number=page.page_number,
            start_index=chunk_index,
        )
        chunks.extend(page_chunks)
        chunk_index += len(page_chunks)

    logger.debug(
        f"Chunked [{extracted_doc.doc_id[:8]}] into {len(chunks)} chunks "
        f"from {len(extracted_doc.pages)} page(s)"
    )
    return chunks


def _chunk_text(
    text: str,
    doc_id: str,
    source_path: str,
    page_number: int,
    start_index: int,
) -> List[TextChunk]:
    """Internal: split one page's text into overlapping chunks."""
    chunks: List[TextChunk] = []
    chunk_size = config.CHUNK_SIZE
    overlap = config.CHUNK_OVERLAP

    # Split into sentences (simple but effective for legal text)
    sentences = _split_sentences(text)

    current_chars = []
    current_len = 0
    char_offset = 0

    for sentence in sentences:
        sent_len = len(sentence)

        if current_len + sent_len > chunk_size and current_chars:
            # Emit current chunk
            chunk_text = " ".join(current_chars).strip()
            if chunk_text:
                idx = start_index + len(chunks)
                chunks.append(TextChunk(
                    chunk_id=f"{doc_id}::chunk::{idx}",
                    doc_id=doc_id,
                    source_path=source_path,
                    page_number=page_number,
                    chunk_index=idx,
                    text=chunk_text,
                    char_start=char_offset,
                    char_end=char_offset + len(chunk_text),
                ))
                char_offset += len(chunk_text)

            # Build overlap from tail of current chunk
            overlap_text = chunk_text[-overlap:] if overlap > 0 else ""
            current_chars = [overlap_text] if overlap_text else []
            current_len = len(overlap_text)

        current_chars.append(sentence)
        current_len += sent_len + 1  # +1 for space

    # Emit remaining text
    if current_chars:
        chunk_text = " ".join(current_chars).strip()
        if chunk_text:
            idx = start_index + len(chunks)
            chunks.append(TextChunk(
                chunk_id=f"{doc_id}::chunk::{idx}",
                doc_id=doc_id,
                source_path=source_path,
                page_number=page_number,
                chunk_index=idx,
                text=chunk_text,
                char_start=char_offset,
                char_end=char_offset + len(chunk_text),
            ))

    return chunks


def _split_sentences(text: str) -> List[str]:
    """
    Split text into sentences. Legal text uses periods, semicolons,
    and section markers as natural boundaries.
    """
    # Split on sentence-ending punctuation followed by whitespace+capital
    parts = re.split(r'(?<=[.!?;])\s+(?=[A-Z0-9"\(\[])', text)
    # Further split very long parts at paragraph breaks
    result = []
    for part in parts:
        if len(part) > config.CHUNK_SIZE * 1.5:
            # Split on newlines if sentence is too long
            sub_parts = part.split("\n")
            result.extend(p.strip() for p in sub_parts if p.strip())
        else:
            stripped = part.strip()
            if stripped:
                result.append(stripped)
    return result
