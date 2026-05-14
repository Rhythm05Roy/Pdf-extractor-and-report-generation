"""
Document type classifier.
Uses keyword heuristics for fast first-pass, then optionally LLM for edge cases.
"""
import re
from typing import Tuple
from loguru import logger

from extraction.models import DocumentType


# Keyword signatures per document type (case-insensitive)
TYPE_SIGNATURES: dict = {
    DocumentType.CONTRACT: [
        r"\b(agreement|contract|covenant|parties agree|witnesseth|whereas|hereinafter)\b",
        r"\b(terms and conditions|obligations|consideration|executed|signed)\b",
    ],
    DocumentType.AFFIDAVIT: [
        r"\b(affidavit|sworn|subscribed|notary|deponent|attests|under oath)\b",
    ],
    DocumentType.MOTION: [
        r"\b(motion|moves the court|respectfully requests|memorandum in support)\b",
        r"\b(plaintiff moves|defendant moves|counsel moves)\b",
    ],
    DocumentType.BRIEF: [
        r"\b(brief|argument|statement of facts|table of contents|table of authorities)\b",
        r"\b(introduction|conclusion|relief requested|summary of argument)\b",
    ],
    DocumentType.NOTICE: [
        r"\b(notice|notification|hereby notified|take notice|you are hereby)\b",
        r"\b(notice of|notice to|written notice)\b",
    ],
    DocumentType.COMPLAINT: [
        r"\b(complaint|plaintiff alleges|cause of action|wherefore|demands judgment)\b",
        r"\b(civil complaint|counts|allegations|relief)\b",
    ],
    DocumentType.ORDER: [
        r"\b(order|ordered|it is hereby|so ordered|court orders|the court finds)\b",
        r"\b(judgment|decree|ruling|adjudicated)\b",
    ],
    DocumentType.DEPOSITION: [
        r"\b(deposition|deposed|transcript|examination|cross-examination|witness)\b",
        r"\b(q:|a:|question:|answer:)\b",
    ],
    DocumentType.MEMO: [
        r"\b(memorandum|memo|to:|from:|re:|subject:|date:)\b",
        r"\b(internal|privileged and confidential|attorney.client)\b",
    ],
    DocumentType.CORRESPONDENCE: [
        r"\b(dear|sincerely|yours truly|regards|letter|correspondence)\b",
        r"\b(enclosures?|cc:|bcc:|via certified)\b",
    ],
    DocumentType.EXHIBIT: [
        r"\b(exhibit [a-z0-9]+|attachment [a-z0-9]+|schedule [a-z0-9]+)\b",
        r"\b(appended|hereto attached|incorporated by reference)\b",
    ],
}


def classify_document(text: str) -> Tuple[DocumentType, float]:
    """
    Classify a document based on keyword heuristics.

    Returns:
        (DocumentType, confidence 0.0–1.0)
    """
    if not text or len(text.strip()) < 30:
        return DocumentType.UNKNOWN, 0.0

    text_lower = text.lower()
    scores: dict = {}

    for doc_type, patterns in TYPE_SIGNATURES.items():
        match_count = 0
        for pattern in patterns:
            if re.search(pattern, text_lower, re.IGNORECASE):
                match_count += 1
        if match_count > 0:
            scores[doc_type] = match_count / len(patterns)

    if not scores:
        return DocumentType.UNKNOWN, 0.0

    best_type = max(scores, key=scores.__getitem__)
    confidence = min(scores[best_type] + 0.1, 1.0)  # small bonus for any match

    logger.debug(f"Document classified as {best_type.value} (conf={confidence:.2f})")
    return best_type, confidence
