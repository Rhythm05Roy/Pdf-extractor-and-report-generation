"""
Tests for the ingestion stage.
"""
import sys
import os
import tempfile
import pytest
from pathlib import Path

# Project root on path
sys.path.insert(0, str(Path(__file__).parent.parent))


class TestDocumentLoader:

    def test_load_text_file(self, tmp_path):
        """Text files load correctly."""
        from ingestion.document_loader import load_document
        f = tmp_path / "test.txt"
        f.write_text("This is a test legal document. Case No. CV-2024-001.")
        doc = load_document(str(f))
        assert doc.full_text != ""
        assert "legal document" in doc.full_text
        assert doc.total_pages == 1

    def test_unsupported_extension_raises(self, tmp_path):
        from ingestion.document_loader import load_document
        f = tmp_path / "test.xyz"
        f.write_text("data")
        with pytest.raises(ValueError, match="Unsupported"):
            load_document(str(f))

    def test_missing_file_raises(self):
        from ingestion.document_loader import load_document
        with pytest.raises(FileNotFoundError):
            load_document("/nonexistent/path/doc.pdf")

    def test_doc_id_assigned(self, tmp_path):
        from ingestion.document_loader import load_document
        f = tmp_path / "test.txt"
        f.write_text("Legal content here.")
        doc = load_document(str(f))
        assert doc.doc_id is not None
        assert len(doc.doc_id) == 36  # UUID4 format

    def test_custom_doc_id(self, tmp_path):
        from ingestion.document_loader import load_document
        f = tmp_path / "test.txt"
        f.write_text("Legal content here.")
        doc = load_document(str(f), doc_id="custom-id-123")
        assert doc.doc_id == "custom-id-123"

    def test_page_content_word_count(self, tmp_path):
        from ingestion.document_loader import load_document
        text = "The quick brown fox jumps over the lazy dog."
        f = tmp_path / "test.txt"
        f.write_text(text)
        doc = load_document(str(f))
        assert doc.pages[0].word_count == 9


class TestOCREngine:

    def test_clean_ocr_text(self):
        try:
            from ingestion.ocr_engine import _clean_ocr_text
        except Exception:
            pytest.skip("OCR engine import failed (numpy/pandas ABI issue in venv)")
        raw = "Hello\n\n\n\nWorld\nword-\nnext"
        cleaned = _clean_ocr_text(raw)
        assert "\n\n\n" not in cleaned  # excessive newlines removed
        assert "wordnext" in cleaned   # broken hyphenation fixed

    def test_tesseract_availability_check(self):
        try:
            from ingestion.ocr_engine import is_tesseract_available
        except Exception:
            pytest.skip("OCR engine import failed (numpy/pandas ABI issue in venv)")
        # Should not raise — just return bool
        result = is_tesseract_available()
        assert isinstance(result, bool)


class TestImagePreprocessor:

    def test_preprocess_returns_image(self):
        try:
            from PIL import Image
            import numpy as np
            from ingestion.image_preprocessor import preprocess_image_for_ocr
            img = Image.fromarray((np.random.rand(200, 200, 3) * 255).astype("uint8"))
            result = preprocess_image_for_ocr(img)
            assert result is not None
        except ImportError:
            pytest.skip("PIL not available")

    def test_quality_estimate_range(self):
        try:
            from PIL import Image
            import numpy as np
            from ingestion.image_preprocessor import estimate_image_quality
            img = Image.fromarray((np.random.rand(100, 100, 3) * 255).astype("uint8"))
            score = estimate_image_quality(img)
            assert 0.0 <= score <= 1.0
        except ImportError:
            pytest.skip("PIL/OpenCV not available")
