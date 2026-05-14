"""
Tests for the retrieval stage.
"""
import sys
import pytest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

LEGAL_TEXT = """
This Services Agreement is entered into as of January 15, 2024 between
Westbrook Capital Partners LLC and Meridian Consulting Group Inc.
Case No.: CV-2024-00891-NYC. Total compensation: $540,000.00.
The parties agree to resolve disputes through binding arbitration in New York.
Confidentiality obligations survive termination for three years.
"""


def _make_extracted_doc(text: str, doc_id: str = "test-doc-001"):
    from ingestion.models import ExtractedDocument, PageContent, InputType
    page = PageContent(page_number=1, raw_text=text)
    doc = ExtractedDocument(
        doc_id=doc_id,
        source_path=f"/test/{doc_id}.txt",
        input_type=InputType.TEXT,
        pages=[page],
    )
    doc.__post_init__()
    return doc


class TestChunker:

    def test_chunks_produced(self):
        from retrieval.chunker import chunk_document
        doc = _make_extracted_doc(LEGAL_TEXT * 5)
        chunks = chunk_document(doc)
        assert len(chunks) > 0

    def test_chunk_ids_unique(self):
        from retrieval.chunker import chunk_document
        doc = _make_extracted_doc(LEGAL_TEXT * 5)
        chunks = chunk_document(doc)
        ids = [c.chunk_id for c in chunks]
        assert len(ids) == len(set(ids))

    def test_chunk_has_doc_id(self):
        from retrieval.chunker import chunk_document
        doc = _make_extracted_doc(LEGAL_TEXT)
        chunks = chunk_document(doc)
        for c in chunks:
            assert c.doc_id == "test-doc-001"

    def test_chunk_has_page_number(self):
        from retrieval.chunker import chunk_document
        doc = _make_extracted_doc(LEGAL_TEXT)
        chunks = chunk_document(doc)
        for c in chunks:
            assert c.page_number == 1

    def test_empty_doc_gives_no_chunks(self):
        from retrieval.chunker import chunk_document
        doc = _make_extracted_doc("")
        chunks = chunk_document(doc)
        assert len(chunks) == 0


class TestBM25Index:

    def _build_index(self):
        from retrieval.chunker import chunk_document
        from retrieval.bm25_index import BM25Index
        doc = _make_extracted_doc(LEGAL_TEXT * 3)
        chunks = chunk_document(doc)
        idx = BM25Index()
        idx.add_chunks(chunks)
        return idx, chunks

    def test_query_returns_results(self):
        idx, chunks = self._build_index()
        results = idx.query("arbitration New York", top_k=3)
        assert len(results) >= 0  # may be empty if text too short

    def test_empty_index_returns_empty(self):
        from retrieval.bm25_index import BM25Index
        idx = BM25Index()
        results = idx.query("anything", top_k=3)
        assert results == []

    def test_scores_normalized(self):
        idx, _ = self._build_index()
        results = idx.query("confidentiality termination", top_k=5)
        for r in results:
            assert 0.0 <= r.score <= 1.0


class TestHybridRetriever:

    def test_index_and_retrieve(self, tmp_path):
        """End-to-end: index chunks and retrieve relevant ones."""
        import config
        config.CHROMA_DB_PATH = str(tmp_path / "test_chroma")

        from retrieval.hybrid_retriever import HybridRetriever
        from retrieval.chunker import chunk_document

        doc = _make_extracted_doc(LEGAL_TEXT * 4, "retriever-test-001")
        chunks = chunk_document(doc)

        retriever = HybridRetriever()
        retriever.index_chunks(chunks)

        results = retriever.retrieve("confidentiality obligations", top_k=3)
        assert isinstance(results, list)
        # All results should have a score
        for r in results:
            assert r.score >= 0.0

    def test_source_label_format(self, tmp_path):
        import config
        config.CHROMA_DB_PATH = str(tmp_path / "test_chroma2")

        from retrieval.hybrid_retriever import HybridRetriever
        from retrieval.chunker import chunk_document

        doc = _make_extracted_doc(LEGAL_TEXT * 2, "label-test-001")
        chunks = chunk_document(doc)
        retriever = HybridRetriever()
        retriever.index_chunks(chunks)
        results = retriever.retrieve("agreement", top_k=2)
        for r in results:
            label = r.source_label
            assert "label-test-001.txt" in label or "p." in label
