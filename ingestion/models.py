"""
Data models for the ingestion stage.
"""
from dataclasses import dataclass, field
from typing import List, Optional
from enum import Enum


class InputType(str, Enum):
    PDF_NATIVE = "pdf_native"       # PDF with selectable text
    PDF_SCANNED = "pdf_scanned"     # PDF that is image-based
    IMAGE = "image"                 # standalone image (JPG, PNG, TIFF)
    TEXT = "text"                   # plain text file


@dataclass
class PageContent:
    """Extracted content from a single page."""
    page_number: int
    raw_text: str
    ocr_confidence: Optional[float] = None   # 0.0–1.0; None if native PDF text
    word_count: int = 0
    is_ocr: bool = False
    warnings: List[str] = field(default_factory=list)

    def __post_init__(self):
        self.word_count = len(self.raw_text.split())


@dataclass
class ExtractedDocument:
    """
    Full output of the ingestion stage for a single file.
    Ready to be passed directly into extraction and retrieval stages.
    """
    doc_id: str                              # UUID assigned at ingest time
    source_path: str                         # original file path
    input_type: InputType
    pages: List[PageContent] = field(default_factory=list)
    full_text: str = ""                      # concatenated text of all pages
    total_pages: int = 0
    avg_ocr_confidence: Optional[float] = None
    file_size_bytes: int = 0
    ingestion_warnings: List[str] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)  # arbitrary extra metadata

    def __post_init__(self):
        if not self.full_text and self.pages:
            self.full_text = "\n\n".join(p.raw_text for p in self.pages)
        self.total_pages = len(self.pages)
        confidences = [p.ocr_confidence for p in self.pages if p.ocr_confidence is not None]
        if confidences:
            self.avg_ocr_confidence = sum(confidences) / len(confidences)
