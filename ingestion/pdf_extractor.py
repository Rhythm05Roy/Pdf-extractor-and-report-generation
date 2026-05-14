"""
PDF text extraction using PyMuPDF with OCR fallback.
Handles both native (selectable text) and scanned PDFs.
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
# DPI for rendering PDF pages to images for OCR
OCR_DPI = 200


def extract_from_pdf(pdf_path: str) -> List[PageContent]:
    """
    Extract text from every page of a PDF.
    - If page has selectable text → use it directly
    - If page is essentially blank → render and OCR
    Returns a list of PageContent (one per page).
    """
    pages: List[PageContent] = []
    tesseract_ok = is_tesseract_available()

    try:
        doc = fitz.open(pdf_path)
    except Exception as e:
        logger.error(f"Cannot open PDF {pdf_path}: {e}")
        return []

    logger.info(f"Processing PDF: {pdf_path} ({len(doc)} pages)")

    for page_num in range(len(doc)):
        page = doc[page_num]
        warnings: List[str] = []

        # Try native text first
        native_text = page.get_text("text").strip()

        if len(native_text) >= MIN_NATIVE_TEXT_CHARS:
            # Good native text — use as-is
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
