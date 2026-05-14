"""
Tests for extraction stage.
"""
import sys
import pytest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


CONTRACT_TEXT = """
SERVICES AGREEMENT Case No.: CV-2024-00891-NYC

This Agreement is entered into as of January 15, 2024 by and between
Westbrook Capital Partners LLC ("Client") and Meridian Consulting Group Inc.
("Service Provider"). Plaintiff: Westbrook Capital Partners LLC.
Defendant: Meridian Consulting Group Inc.
Court: United States District Court Southern District of New York.
Total compensation: $540,000.00. Monthly retainer: $45,000.00.
Attorney: Michael D. Harrison, Esq. Bar No. 4521890.
Hearing scheduled for March 15, 2024.
CONFIDENTIALITY: All information shall remain confidential for 3 years.
GOVERNING LAW: Governed by the laws of the State of New York.
"""

NOTICE_TEXT = """
NOTICE OF DEFAULT
You are hereby notified that Redstone Development Corp. is in default.
Date: March 8, 2024. Docket Reference: PSL-2024-0308.
Outstanding rent: $375,000.00.
"""


class TestDocumentClassifier:

    def test_contract_classification(self):
        from extraction.document_classifier import classify_document
        doc_type, conf = classify_document(CONTRACT_TEXT)
        assert doc_type.value == "contract"
        assert conf > 0.5

    def test_notice_classification(self):
        from extraction.document_classifier import classify_document
        doc_type, conf = classify_document(NOTICE_TEXT)
        assert doc_type.value == "notice"
        assert conf > 0.0

    def test_empty_text_returns_unknown(self):
        from extraction.document_classifier import classify_document
        from extraction.models import DocumentType
        doc_type, conf = classify_document("")
        assert doc_type == DocumentType.UNKNOWN
        assert conf == 0.0

    def test_short_text_returns_unknown(self):
        from extraction.document_classifier import classify_document
        from extraction.models import DocumentType
        doc_type, conf = classify_document("Hello")
        assert doc_type == DocumentType.UNKNOWN


class TestFieldExtractor:

    def _make_extracted_doc(self, text: str):
        """Helper: create a minimal ExtractedDocument for testing."""
        from ingestion.models import ExtractedDocument, PageContent, InputType
        page = PageContent(page_number=1, raw_text=text)
        doc = ExtractedDocument(
            doc_id="test-001",
            source_path="/test/doc.txt",
            input_type=InputType.TEXT,
            pages=[page],
        )
        doc.__post_init__()
        return doc

    def test_case_number_extracted(self):
        from extraction.field_extractor import extract_fields
        doc = self._make_extracted_doc(CONTRACT_TEXT)
        struct = extract_fields(doc)
        assert struct.case_number is not None
        assert "2024" in struct.case_number

    def test_dates_extracted(self):
        from extraction.field_extractor import extract_fields
        doc = self._make_extracted_doc(CONTRACT_TEXT)
        struct = extract_fields(doc)
        assert len(struct.all_dates) > 0

    def test_monetary_amounts_extracted(self):
        from extraction.field_extractor import extract_fields
        doc = self._make_extracted_doc(CONTRACT_TEXT)
        struct = extract_fields(doc)
        assert len(struct.monetary_amounts) > 0
        # Should find $540,000 or $45,000
        amounts_str = " ".join(struct.monetary_amounts)
        assert "$" in amounts_str

    def test_plaintiff_extracted(self):
        from extraction.field_extractor import extract_fields
        doc = self._make_extracted_doc(CONTRACT_TEXT)
        struct = extract_fields(doc)
        assert struct.plaintiff is not None

    def test_clauses_extracted(self):
        from extraction.field_extractor import extract_fields
        doc = self._make_extracted_doc(CONTRACT_TEXT)
        struct = extract_fields(doc)
        assert len(struct.key_clauses) > 0

    def test_structured_doc_has_doc_id(self):
        from extraction.field_extractor import extract_fields
        doc = self._make_extracted_doc(CONTRACT_TEXT)
        struct = extract_fields(doc)
        assert struct.doc_id == "test-001"

    def test_to_dict_is_serializable(self):
        import json
        from extraction.field_extractor import extract_fields
        doc = self._make_extracted_doc(NOTICE_TEXT)
        struct = extract_fields(doc)
        d = struct.to_dict()
        # Should be JSON serializable
        json_str = json.dumps(d)
        assert json_str is not None
