"""
Mistral OCR engine.
Uses the Mistral Document AI API (mistral-ocr-latest) for high-quality
extraction from scanned or noisy PDFs.  Falls back to Tesseract if the
API key is not configured or the call fails.
"""
import base64
import io
import os
from pathlib import Path
from typing import Tuple, List, Optional
from loguru import logger


def _get_mistral_client():
    """Lazy-load the Mistral client."""
    try:
        from mistralai.client import Mistral
        import config
        api_key = config.MISTRAL_API_KEY
        if not api_key:
            return None
        return Mistral(api_key=api_key)
    except Exception as e:
        logger.debug(f"Mistral client unavailable: {e}")
        return None


def is_mistral_ocr_available() -> bool:
    """Return True if Mistral is configured and importable."""
    try:
        import config
        if not config.MISTRAL_API_KEY:
            return False
        from mistralai.client import Mistral  # noqa: F401
        return True
    except Exception:
        return False


def ocr_pdf_with_mistral(pdf_path: str) -> Tuple[str, float]:
    """
    Send a PDF file to the Mistral OCR endpoint and return
    (full_text, estimated_confidence).

    Confidence is approximated as 0.92 when the API returns content
    (Mistral does not expose per-word confidence scores).
    Falls back to (empty string, 0.0) on any error.
    """
    client = _get_mistral_client()
    if client is None:
        logger.warning("Mistral client not available for OCR")
        return "", 0.0

    try:
        import config
        model = config.MISTRAL_OCR_MODEL

        with open(pdf_path, "rb") as fh:
            pdf_bytes = fh.read()

        b64_pdf = base64.standard_b64encode(pdf_bytes).decode("utf-8")

        logger.info(f"Sending {Path(pdf_path).name} to Mistral OCR ({model})…")

        response = client.ocr.process(
            model=model,
            document={
                "type": "document_url",
                "document_url": f"data:application/pdf;base64,{b64_pdf}",
            },
        )

        # Collect text from all pages
        pages_text: List[str] = []
        for page in response.pages:
            if hasattr(page, "markdown") and page.markdown:
                pages_text.append(page.markdown)
            elif hasattr(page, "text") and page.text:
                pages_text.append(page.text)

        full_text = "\n\n".join(pages_text).strip()
        confidence = 0.92 if full_text else 0.0

        logger.info(
            f"Mistral OCR complete: {len(pages_text)} pages, "
            f"{len(full_text):,} chars"
        )
        return full_text, confidence

    except Exception as e:
        logger.error(f"Mistral OCR failed for {pdf_path}: {e}")
        return "", 0.0


def ocr_image_with_mistral(pil_image) -> Tuple[str, float]:
    """
    Send a PIL image to the Mistral OCR endpoint.
    The image is encoded as base64 PNG and submitted as an image_url.
    """
    client = _get_mistral_client()
    if client is None:
        return "", 0.0

    try:
        import config
        model = config.MISTRAL_OCR_MODEL

        buf = io.BytesIO()
        pil_image.save(buf, format="PNG")
        b64_img = base64.standard_b64encode(buf.getvalue()).decode("utf-8")

        logger.info("Sending image to Mistral OCR…")

        response = client.ocr.process(
            model=model,
            document={
                "type": "image_url",
                "image_url": f"data:image/png;base64,{b64_img}",
            },
        )

        pages_text: List[str] = []
        for page in response.pages:
            if hasattr(page, "markdown") and page.markdown:
                pages_text.append(page.markdown)
            elif hasattr(page, "text") and page.text:
                pages_text.append(page.text)

        full_text = "\n\n".join(pages_text).strip()
        confidence = 0.92 if full_text else 0.0
        logger.info(f"Mistral image OCR complete: {len(full_text):,} chars")
        return full_text, confidence

    except Exception as e:
        logger.error(f"Mistral image OCR failed: {e}")
        return "", 0.0
