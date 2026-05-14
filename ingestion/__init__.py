"""Ingestion package."""
from ingestion.document_loader import load_document, load_multiple_documents
from ingestion.models import ExtractedDocument, PageContent, InputType

__all__ = [
    "load_document",
    "load_multiple_documents",
    "ExtractedDocument",
    "PageContent",
    "InputType",
]
