"""
Data models for the extraction stage.
"""
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from enum import Enum


class DocumentType(str, Enum):
    CONTRACT = "contract"
    AFFIDAVIT = "affidavit"
    MOTION = "motion"
    BRIEF = "brief"
    NOTICE = "notice"
    COMPLAINT = "complaint"
    ORDER = "order"
    DEPOSITION = "deposition"
    MEMO = "memo"
    CORRESPONDENCE = "correspondence"
    EXHIBIT = "exhibit"
    UNKNOWN = "unknown"


@dataclass
class ExtractedField:
    """A single field extracted from a document with confidence."""
    key: str
    value: Any
    confidence: float  # 0.0–1.0
    method: str = "regex"  # "regex" | "llm" | "heuristic"
    source_text: str = ""  # snippet of text the value was pulled from


@dataclass
class StructuredDocument:
    """
    Structured output of the extraction stage.
    Downstream stages (retrieval, generation) consume this.
    """
    doc_id: str
    source_path: str
    document_type: DocumentType = DocumentType.UNKNOWN
    document_type_confidence: float = 0.0

    # Core legal fields
    case_number: Optional[str] = None
    docket_id: Optional[str] = None
    parties: List[str] = field(default_factory=list)       # all named parties
    plaintiff: Optional[str] = None
    defendant: Optional[str] = None
    counsel: List[str] = field(default_factory=list)       # attorney names
    court: Optional[str] = None
    jurisdiction: Optional[str] = None

    # Dates
    filing_date: Optional[str] = None
    hearing_date: Optional[str] = None
    effective_date: Optional[str] = None
    all_dates: List[str] = field(default_factory=list)     # every date found

    # Financial
    monetary_amounts: List[str] = field(default_factory=list)
    penalties: List[str] = field(default_factory=list)

    # Key content
    key_clauses: List[str] = field(default_factory=list)
    subject_matter: Optional[str] = None
    summary: Optional[str] = None                          # short LLM summary

    # All extracted fields (for transparency)
    all_fields: List[ExtractedField] = field(default_factory=list)

    # Quality flags
    extraction_warnings: List[str] = field(default_factory=list)
    ocr_quality: Optional[float] = None

    def to_dict(self) -> dict:
        return {
            "doc_id": self.doc_id,
            "source_path": self.source_path,
            "document_type": self.document_type.value,
            "document_type_confidence": self.document_type_confidence,
            "case_number": self.case_number,
            "docket_id": self.docket_id,
            "parties": self.parties,
            "plaintiff": self.plaintiff,
            "defendant": self.defendant,
            "counsel": self.counsel,
            "court": self.court,
            "jurisdiction": self.jurisdiction,
            "filing_date": self.filing_date,
            "hearing_date": self.hearing_date,
            "effective_date": self.effective_date,
            "all_dates": self.all_dates,
            "monetary_amounts": self.monetary_amounts,
            "penalties": self.penalties,
            "key_clauses": self.key_clauses,
            "subject_matter": self.subject_matter,
            "summary": self.summary,
            "extraction_warnings": self.extraction_warnings,
        }
