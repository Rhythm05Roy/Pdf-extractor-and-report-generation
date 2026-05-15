"""
Field extractor: pull structured legal fields from raw text.
Two-pass approach:
  1. Regex patterns handle dates, case numbers, money, parties (fast, reliable)
  2. LLM pass fills gaps and extracts summary + subject matter
"""
import re
import json
from typing import List, Optional, Tuple
from loguru import logger

from extraction.models import StructuredDocument, ExtractedField, DocumentType
from extraction.document_classifier import classify_document
from ingestion.models import ExtractedDocument



CASE_NUMBER_PATTERNS = [
    r"\bCase\s+No\.?\s*[:\-]?\s*([A-Z0-9\-:\/]+)",
    r"\bDocket\s+No\.?\s*[:\-]?\s*([A-Z0-9\-:\/]+)",
    r"\bCiv\.?\s*(?:Action\s+)?No\.?\s*[:\-]?\s*([A-Z0-9\-:\/]+)",
    r"\b(\d{1,2}[-:]\d{2,5}[-:][A-Z]{2,6}[-:]\d+)\b",
]

DATE_PATTERNS = [
    r"\b(\d{1,2}[\/\-]\d{1,2}[\/\-]\d{2,4})\b",                   # 01/15/2024
    r"\b((?:January|February|March|April|May|June|July|August|"
    r"September|October|November|December)\s+\d{1,2},?\s+\d{4})\b",  # January 15, 2024
    r"\b(\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)"
    r"\.?\s+\d{4})\b",                                              # 15 Jan 2024
    r"\b(\d{4}-\d{2}-\d{2})\b",                                    # 2024-01-15
]

MONEY_PATTERNS = [
    r"\$\s*\d[\d,]*(?:\.\d{1,2})?(?:\s*(?:million|billion|thousand))?\b",
    r"\b\d[\d,]*(?:\.\d{1,2})?\s*(?:USD|dollars?)\b",
]

# Party indicators
PLAINTIFF_PATTERNS = [
    r"(?:Plaintiff[s]?|Petitioner[s]?|Claimant[s]?)\s*[,:\-]?\s*([A-Z][A-Za-z\s,\.]+?)(?:\n|,|vs?\.)",
    r"([A-Z][A-Za-z\s,\.]+?),?\s+(?:Plaintiff|Petitioner)",
]
DEFENDANT_PATTERNS = [
    r"(?:Defendant[s]?|Respondent[s]?)\s*[,:\-]?\s*([A-Z][A-Za-z\s,\.]+?)(?:\n|,|and\b)",
    r"([A-Z][A-Za-z\s,\.]+?),?\s+(?:Defendant|Respondent)",
]
COUNSEL_PATTERNS = [
    r"(?:Attorney|Counsel|Esquire|Esq\.?|Bar\s+No\.?)[:\s]+([A-Z][A-Za-z\s,\.]+?)(?:\n|,)",
    r"([A-Z][A-Za-z\s]+),?\s+(?:Esq\.?|Attorney\s+at\s+Law)",
]
COURT_PATTERNS = [
    r"(?:IN\s+THE\s+|BEFORE\s+THE\s+)([A-Z][A-Za-z\s]+COURT[A-Za-z\s]*)",
    r"((?:United\s+States|U\.S\.)\s+District\s+Court[A-Za-z\s,]*)",
    r"((?:Superior|Circuit|District|Appeals?|Supreme)\s+Court[A-Za-z\s,]*)",
]
CLAUSE_HEADERS = [
    "WHEREAS", "NOW THEREFORE", "COVENANTS", "REPRESENTATIONS",
    "WARRANTIES", "INDEMNIFICATION", "LIMITATION OF LIABILITY",
    "GOVERNING LAW", "ARBITRATION", "CONFIDENTIALITY", "TERMINATION",
    "FORCE MAJEURE", "ASSIGNMENT", "NOTICES",
]


def extract_fields(extracted_doc: ExtractedDocument) -> StructuredDocument:
    """
    Main extraction entry point.
    Returns a StructuredDocument populated with all extractable fields.
    """
    text = extracted_doc.full_text
    doc_type, type_conf = classify_document(text)

    struct = StructuredDocument(
        doc_id=extracted_doc.doc_id,
        source_path=extracted_doc.source_path,
        document_type=doc_type,
        document_type_confidence=type_conf,
        ocr_quality=extracted_doc.avg_ocr_confidence,
    )

    struct.case_number = _first_match(CASE_NUMBER_PATTERNS, text, "case_number", struct)
    struct.all_dates = _all_matches(DATE_PATTERNS, text)
    struct.filing_date = _infer_filing_date(text, struct.all_dates)
    struct.hearing_date = _infer_hearing_date(text, struct.all_dates)
    struct.monetary_amounts = _all_matches(MONEY_PATTERNS, text)
    struct.plaintiff = _first_match(PLAINTIFF_PATTERNS, text, "plaintiff", struct)
    struct.defendant = _first_match(DEFENDANT_PATTERNS, text, "defendant", struct)
    struct.counsel = _all_matches(COUNSEL_PATTERNS, text, max_results=4)
    struct.court = _first_match(COURT_PATTERNS, text, "court", struct)
    struct.parties = _collect_parties(struct)
    struct.key_clauses = _find_clauses(text)

    try:
        _enrich_with_llm(struct, text)
    except Exception as e:
        logger.warning(f"LLM enrichment failed: {e} — using regex results only")
        struct.extraction_warnings.append(f"LLM enrichment unavailable: {e}")

    logger.info(
        f"Extracted [{struct.doc_id[:8]}]: type={struct.document_type.value}, "
        f"case={struct.case_number}, {len(struct.all_dates)} dates, "
        f"{len(struct.monetary_amounts)} amounts"
    )
    return struct



def _first_match(
    patterns: List[str], text: str, field_name: str, struct: StructuredDocument
) -> Optional[str]:
    for pat in patterns:
        m = re.search(pat, text, re.IGNORECASE | re.MULTILINE)
        if m:
            val = m.group(1).strip() if m.lastindex else m.group(0).strip()
            val = re.sub(r"\s+", " ", val).strip(".,;: ")
            if val:
                struct.all_fields.append(
                    ExtractedField(field_name, val, 0.8, "regex", m.group(0))
                )
                return val
    return None


def _all_matches(
    patterns: List[str], text: str, max_results: int = 20
) -> List[str]:
    results = set()
    for pat in patterns:
        for m in re.finditer(pat, text, re.IGNORECASE | re.MULTILINE):
            val = (m.group(1).strip() if m.lastindex else m.group(0).strip())
            val = re.sub(r"\s+", " ", val).strip(".,;: ")
            if val and len(val) > 1:
                results.add(val)
            if len(results) >= max_results:
                break
    return list(results)[:max_results]


def _collect_parties(struct: StructuredDocument) -> List[str]:
    parties = []
    if struct.plaintiff:
        parties.append(struct.plaintiff)
    if struct.defendant:
        parties.append(struct.defendant)
    parties.extend(struct.counsel)
    return list(dict.fromkeys(parties))  # deduplicate, preserve order


def _find_clauses(text: str) -> List[str]:
    """Find clause headers and return the first ~100 chars of each."""
    clauses = []
    for header in CLAUSE_HEADERS:
        idx = text.upper().find(header)
        if idx != -1:
            snippet = text[idx: idx + 120].strip().replace("\n", " ")
            clauses.append(snippet)
    return clauses


def _infer_filing_date(text: str, all_dates: List[str]) -> Optional[str]:
    """Try to find a date near 'filed' or 'filing date' keywords."""
    m = re.search(
        r"(?:filed?|filing\s+date|date\s+filed)\s*[:\-]?\s*"
        r"(\d{1,2}[\/\-]\d{1,2}[\/\-]\d{2,4}|"
        r"(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+\d{1,2},?\s+\d{4})",
        text, re.IGNORECASE,
    )
    if m:
        return m.group(1).strip()
    return all_dates[0] if all_dates else None


def _infer_hearing_date(text: str, all_dates: List[str]) -> Optional[str]:
    m = re.search(
        r"(?:hearing|trial|scheduled\s+for|set\s+for|return\s+date)\s*[:\-]?\s*"
        r"(\d{1,2}[\/\-]\d{1,2}[\/\-]\d{2,4}|"
        r"(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+\d{1,2},?\s+\d{4})",
        text, re.IGNORECASE,
    )
    return m.group(1).strip() if m else None


def _enrich_with_llm(struct: StructuredDocument, text: str) -> None:
    """
    Use Gemini to extract subject_matter and summary.
    Only called if Gemini key is available.
    """
    from generation.llm_client import get_llm_client
    client = get_llm_client()
    if not client:
        return

    snippet = text[:3000]  # first 3K chars is enough for summary
    prompt = f"""You are a legal analyst. Extract the following from the document excerpt:
1. subject_matter: one sentence describing what this document is about
2. summary: 2–3 sentence summary of the key content

Return ONLY valid JSON with keys "subject_matter" and "summary".

DOCUMENT EXCERPT:
{snippet}
"""
    response = client.generate(prompt, temperature=0.1)
    # Parse JSON from response
    json_match = re.search(r"\{.*?\}", response, re.DOTALL)
    if json_match:
        data = json.loads(json_match.group(0))
        struct.subject_matter = data.get("subject_matter", "").strip()
        struct.summary = data.get("summary", "").strip()
        struct.all_fields.append(
            ExtractedField("subject_matter", struct.subject_matter, 0.85, "llm")
        )
        struct.all_fields.append(
            ExtractedField("summary", struct.summary, 0.85, "llm")
        )
