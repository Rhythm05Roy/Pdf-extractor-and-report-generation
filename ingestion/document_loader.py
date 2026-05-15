"""
Document loader: routes each file to the correct extractor,
assigns doc IDs, and returns an ExtractedDocument.
"""
import uuid
import os
from pathlib import Path
from typing import Optional
from loguru import logger

from ingestion.models import ExtractedDocument, InputType, PageContent


SUPPORTED_EXTENSIONS = {
    ".pdf": InputType.PDF_NATIVE,  # will be refined to SCANNED if needed
    ".jpg": InputType.IMAGE,
    ".jpeg": InputType.IMAGE,
    ".png": InputType.IMAGE,
    ".tiff": InputType.IMAGE,
    ".tif": InputType.IMAGE,
    ".bmp": InputType.IMAGE,
    ".txt": InputType.TEXT,
    ".md": InputType.TEXT,
}


def load_document(file_path: str, doc_id: Optional[str] = None) -> ExtractedDocument:
    """
    Main entry point. Accept a file path, route it to the right extractor,
    and return a fully populated ExtractedDocument.

    Args:
        file_path: Absolute or relative path to the source file.
        doc_id:    Optional pre-assigned document ID (UUID string).
                   If None, a new UUID is generated.

    Returns:
        ExtractedDocument ready for downstream stages.
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Document not found: {file_path}")

    ext = path.suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Unsupported file type: {ext}. "
            f"Supported: {list(SUPPORTED_EXTENSIONS.keys())}"
        )

    doc_id = doc_id or str(uuid.uuid4())
    file_size = path.stat().st_size
    logger.info(f"Loading document [{doc_id[:8]}]: {path.name} ({file_size:,} bytes)")

    if ext == ".pdf":
        doc = _load_pdf(str(path), doc_id, file_size)
    elif ext in (".jpg", ".jpeg", ".png", ".tiff", ".tif", ".bmp"):
        doc = _load_image(str(path), doc_id, file_size)
    else:
        doc = _load_text(str(path), doc_id, file_size)

    # Collect all page warnings at doc level
    all_warnings = []
    for p in doc.pages:
        all_warnings.extend(p.warnings)
    doc.ingestion_warnings = all_warnings

    logger.info(
        f"Ingested [{doc_id[:8]}] {path.name}: "
        f"{doc.total_pages} page(s), {len(doc.full_text):,} chars"
        + (f", avg OCR conf {doc.avg_ocr_confidence:.0%}" if doc.avg_ocr_confidence else "")
    )
    return doc


def _load_pdf(path: str, doc_id: str, file_size: int) -> ExtractedDocument:
    from ingestion.pdf_extractor import extract_from_pdf, detect_pdf_type
    input_type = detect_pdf_type(path)
    pages = extract_from_pdf(path)
    doc = ExtractedDocument(
        doc_id=doc_id,
        source_path=path,
        input_type=input_type,
        pages=pages,
        file_size_bytes=file_size,
        metadata={"filename": Path(path).name},
    )
    doc.__post_init__()
    return doc


def _load_image(path: str, doc_id: str, file_size: int) -> ExtractedDocument:
    from PIL import Image
    import config

    text, confidence = "", 0.0

    # Try Mistral OCR first
    if config.USE_MISTRAL_OCR and config.MISTRAL_API_KEY:
        try:
            from ingestion.mistral_ocr import ocr_image_with_mistral, is_mistral_ocr_available
            if is_mistral_ocr_available():
                pil_image = Image.open(path)
                text, confidence = ocr_image_with_mistral(pil_image)
                logger.info(f"Mistral image OCR: {len(text):,} chars (conf≈{confidence:.0%})")
        except Exception as e:
            logger.warning(f"Mistral image OCR failed: {e}, falling back to Tesseract")

    # Fallback to Tesseract
    if not text:
        try:
            from ingestion.ocr_engine import extract_text_with_layout
            pil_image = Image.open(path)
            text, confidence = extract_text_with_layout(pil_image)
        except Exception as e:
            logger.error(f"Failed to OCR image {path}: {e}")
            text, confidence = "", 0.0

    warnings = []
    if confidence < 0.4:
        warnings.append(f"Low OCR confidence on image ({confidence:.0%})")

    page = PageContent(
        page_number=1,
        raw_text=text,
        ocr_confidence=confidence,
        is_ocr=True,
        warnings=warnings,
    )
    doc = ExtractedDocument(
        doc_id=doc_id,
        source_path=path,
        input_type=InputType.IMAGE,
        pages=[page],
        file_size_bytes=file_size,
        metadata={"filename": Path(path).name},
    )
    doc.__post_init__()
    return doc


def _load_text(path: str, doc_id: str, file_size: int) -> ExtractedDocument:
    try:
        text = Path(path).read_text(encoding="utf-8", errors="replace")
    except Exception as e:
        logger.error(f"Failed to read text file {path}: {e}")
        text = ""

    page = PageContent(
        page_number=1,
        raw_text=text,
        ocr_confidence=None,
        is_ocr=False,
    )
    doc = ExtractedDocument(
        doc_id=doc_id,
        source_path=path,
        input_type=InputType.TEXT,
        pages=[page],
        file_size_bytes=file_size,
        metadata={"filename": Path(path).name},
    )
    doc.__post_init__()
    return doc


def load_multiple_documents(file_paths: list) -> list:
    """Convenience: load a batch of documents. Skips failures with a warning."""
    results = []
    for fp in file_paths:
        try:
            results.append(load_document(fp))
        except Exception as e:
            logger.warning(f"Skipping {fp}: {e}")
    return results
