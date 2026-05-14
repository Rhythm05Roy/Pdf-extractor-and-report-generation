"""
OCR engine wrapper around Tesseract.
Provides word-level confidence scoring and structured page output.
"""
import os
import re
from typing import Tuple, Optional
from loguru import logger

import pytesseract
from PIL import Image

from config import TESSERACT_CMD
from ingestion.image_preprocessor import preprocess_image_for_ocr, estimate_image_quality

# Point pytesseract at the right binary
pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD


# Tesseract page segmentation modes
PSM_SINGLE_COLUMN = "4"   # single column of text
PSM_AUTO = "3"             # fully automatic (default)
PSM_SPARSE = "11"          # sparse text, find as much as possible


def extract_text_from_image(
    pil_image: Image.Image,
    preprocess: bool = True,
    psm: str = PSM_AUTO,
) -> Tuple[str, float]:
    """
    Run OCR on a PIL Image.

    Returns:
        (text, confidence)  where confidence is 0.0–1.0
    """
    if preprocess:
        quality_before = estimate_image_quality(pil_image)
        pil_image = preprocess_image_for_ocr(pil_image)
        quality_after = estimate_image_quality(pil_image)
        logger.debug(f"Image quality: {quality_before:.3f} → {quality_after:.3f}")

    config = f"--psm {psm} --oem 3"

    try:
        # Get detailed data with per-word confidence
        data = pytesseract.image_to_data(
            pil_image,
            config=config,
            output_type=pytesseract.Output.DICT,
        )
        confidences = [
            int(c) for c in data["conf"]
            if str(c).strip() not in ("-1", "")
        ]
        text = pytesseract.image_to_string(pil_image, config=config)
        text = _clean_ocr_text(text)

        if confidences:
            avg_conf = sum(confidences) / len(confidences) / 100.0  # → 0–1
        else:
            avg_conf = 0.0

        return text, avg_conf

    except pytesseract.TesseractError as e:
        logger.warning(f"Tesseract error: {e} — retrying with sparse PSM")
        try:
            text = pytesseract.image_to_string(pil_image, config=f"--psm {PSM_SPARSE}")
            text = _clean_ocr_text(text)
            return text, 0.3  # low confidence for fallback
        except Exception as e2:
            logger.error(f"OCR completely failed: {e2}")
            return "", 0.0


def extract_text_with_layout(pil_image: Image.Image) -> Tuple[str, float]:
    """
    Attempt layout-aware OCR (single-column mode first, then auto).
    Better for multi-column legal documents.
    """
    text_col, conf_col = extract_text_from_image(pil_image, psm=PSM_SINGLE_COLUMN)
    text_auto, conf_auto = extract_text_from_image(pil_image, psm=PSM_AUTO)

    # Pick the result with higher confidence and more content
    score_col = conf_col * len(text_col)
    score_auto = conf_auto * len(text_auto)

    if score_col >= score_auto:
        return text_col, conf_col
    return text_auto, conf_auto


def _clean_ocr_text(text: str) -> str:
    """Post-process raw OCR output."""
    if not text:
        return ""
    # Remove excessive blank lines
    text = re.sub(r"\n{3,}", "\n\n", text)
    # Fix broken hyphenation (word- \nnext → wordnext)
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)
    # Strip trailing whitespace per line
    lines = [line.rstrip() for line in text.splitlines()]
    return "\n".join(lines).strip()


def is_tesseract_available() -> bool:
    """Check if Tesseract is installed and accessible."""
    try:
        pytesseract.get_tesseract_version()
        return True
    except Exception:
        return False
