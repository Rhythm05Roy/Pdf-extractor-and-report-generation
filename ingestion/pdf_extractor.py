"""
PDF text extraction using PyMuPDF with Mistral OCR (primary) or
Tesseract (fallback) for scanned / image-only pages.
"""
import io
import os
from pathlib import Path
from typing import List, Optional
from loguru import logger

import fitz  # PyMuPDF
from PIL import Image

from ingestion.models import PageContent, InputType
from ingestion.ocr_engine import extract_text_with_layout, is_tesseract_available

# Threshold: if a page has fewer chars than this, assume it's scanned
MIN_NATIVE_TEXT_CHARS = 50
# DPI for rendering PDF pages to images for Tesseract OCR
OCR_DPI = 200


def extract_from_pdf(pdf_path: str) -> List[PageContent]:
    """
    Extract text from every page of a PDF.

    Strategy (in priority order):
      1. Mistral OCR (whole-document, if API key is configured) — best quality
      2. PyMuPDF native text per page
      3. Tesseract OCR per image page (if Tesseract available)

    Returns a list of PageContent (one per page).
    """
    import config

    # --- Try Mistral whole-document OCR first ---
    if config.USE_MISTRAL_OCR and config.MISTRAL_API_KEY:
        try:
            from ingestion.mistral_ocr import ocr_pdf_with_mistral, is_mistral_ocr_available
            if is_mistral_ocr_available():
                full_text, confidence = ocr_pdf_with_mistral(pdf_path)
                if full_text:
                    logger.info(
                        f"Mistral OCR succeeded for {Path(pdf_path).name} "
                        f"({len(full_text):,} chars, conf≈{confidence:.0%})"
                    )
                    # Split into per-page chunks for consistent downstream handling
                    return _split_into_pages(pdf_path, full_text, confidence)
        except Exception as e:
            logger.warning(f"Mistral OCR failed, falling back to Tesseract: {e}")

    # --- Per-page extraction (PyMuPDF native + Tesseract fallback) ---
    return _extract_per_page(pdf_path)


def _split_into_pages(pdf_path: str, full_text: str, confidence: float) -> List[PageContent]:
    """
    When Mistral returns a single blob, try to honour the real page count
    by detecting Mistral's page separators or splitting evenly.
    """
    pages: List[PageContent] = []

    # Mistral markdown typically emits a horizontal rule (---) between pages
    import re
    raw_pages = re.split(r"\n---\n", full_text)

    # Also check the real page count from the PDF
    try:
        doc = fitz.open(pdf_path)
        real_page_count = len(doc)
        doc.close()
    except Exception:
        real_page_count = len(raw_pages)

    # If Mistral's split doesn't match, treat as one page per chunk
    if len(raw_pages) != real_page_count:
        # Best effort: split by equal thirds of text if multiple pages
        if real_page_count > 1 and len(raw_pages) == 1:
            chunk_size = max(1, len(full_text) // real_page_count)
            raw_pages = [
                full_text[i * chunk_size: (i + 1) * chunk_size]
                for i in range(real_page_count)
            ]

    for i, page_text in enumerate(raw_pages, 1):
        pages.append(PageContent(
            page_number=i,
            raw_text=page_text.strip(),
            ocr_confidence=confidence,
            is_ocr=True,
        ))

    return pages


def _extract_per_page(pdf_path: str) -> List[PageContent]:
    """Original per-page extraction using PyMuPDF + Tesseract fallback."""
    pages: List[PageContent] = []
    tesseract_ok = is_tesseract_available()

    try:
        doc = fitz.open(pdf_path)
    except Exception as e:
        logger.error(f"Cannot open PDF {pdf_path}: {e}")
        return []

    logger.info(f"Processing PDF (per-page): {pdf_path} ({len(doc)} pages)")

    for page_num in range(len(doc)):
        page = doc[page_num]
        warnings: List[str] = []

        # Try native text first
        native_text = page.get_text("text").strip()

        if len(native_text) >= MIN_NATIVE_TEXT_CHARS:
            page_content = PageContent(
                page_number=page_num + 1,
                raw_text=native_text,
                ocr_confidence=None,
                is_ocr=False,
            )
            logger.debug(f"Page {page_num+1}: native text ({len(native_text)} chars)")
        else:
            # Page is image-based or near-empty → OCR
            if not tesseract_ok:
                warnings.append(
                    f"Page {page_num+1} appears scanned but Tesseract is not available. "
                    "Text may be missing."
                )
                page_content = PageContent(
                    page_number=page_num + 1,
                    raw_text=native_text or "",
                    ocr_confidence=0.0,
                    is_ocr=False,
                    warnings=warnings,
                )
            else:
                logger.debug(f"Page {page_num+1}: rendering for OCR (native had {len(native_text)} chars)")
                pil_image = _render_page_to_image(page)
                ocr_text, confidence = extract_text_with_layout(pil_image)

                if confidence < 0.4:
                    warnings.append(
                        f"Page {page_num+1}: low OCR confidence ({confidence:.0%}). "
                        "Content may be partially illegible."
                    )

                page_content = PageContent(
                    page_number=page_num + 1,
                    raw_text=ocr_text,
                    ocr_confidence=confidence,
                    is_ocr=True,
                    warnings=warnings,
                )

        pages.append(page_content)

    doc.close()
    return pages


def detect_pdf_type(pdf_path: str) -> InputType:
    """
    Quick scan to determine whether PDF is native text or scanned.
    Returns InputType.PDF_NATIVE or InputType.PDF_SCANNED.
    """
    try:
        doc = fitz.open(pdf_path)
        sample_pages = min(3, len(doc))
        total_chars = 0
        for i in range(sample_pages):
            total_chars += len(doc[i].get_text("text").strip())
        doc.close()
        avg_chars = total_chars / max(sample_pages, 1)
        return InputType.PDF_NATIVE if avg_chars > MIN_NATIVE_TEXT_CHARS else InputType.PDF_SCANNED
    except Exception:
        return InputType.PDF_SCANNED


def _render_page_to_image(page: fitz.Page) -> Image.Image:
    """Render a PyMuPDF page to a PIL Image at OCR_DPI resolution."""
    mat = fitz.Matrix(OCR_DPI / 72, OCR_DPI / 72)  # 72 dpi is PDF default
    pix = page.get_pixmap(matrix=mat, colorspace=fitz.csRGB)
    img_bytes = pix.tobytes("png")
    return Image.open(io.BytesIO(img_bytes))
