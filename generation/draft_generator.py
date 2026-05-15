"""
Draft generator: orchestrates the full generation pipeline.
Retrieves evidence → builds prompt → calls LLM → parses output → links citations.
"""
import uuid
import json
import re
from typing import List, Optional
from loguru import logger

from extraction.models import StructuredDocument
from retrieval.models import RetrievedChunk
from retrieval.hybrid_retriever import HybridRetriever
from generation.models import DraftMemo, DraftSection, EvidenceRef
from generation.prompt_builder import build_generation_prompt
from generation.citation_linker import link_citations, extract_chunk_ids_from_llm_output
from generation.llm_client import get_llm_client
from learning.pattern_store import get_patterns_for_doc_type
import config


def generate_draft(
    structured_doc: StructuredDocument,
    retriever: HybridRetriever,
    query: Optional[str] = None,
) -> DraftMemo:
    """
    Generate a grounded Case Fact Summary Memo for a document.

    Args:
        structured_doc: Structured extraction result.
        retriever:      Initialized hybrid retriever (already indexed).
        query:          Optional custom query for retrieval.
                        Defaults to a summary of the document.

    Returns:
        A DraftMemo with sections, evidence refs, and citations.
    """
    warnings: List[str] = []
    memo_id = str(uuid.uuid4())

    if not query:
        parts = []
        if structured_doc.subject_matter:
            parts.append(structured_doc.subject_matter)
        if structured_doc.case_number:
            parts.append(f"case number {structured_doc.case_number}")
        if structured_doc.document_type.value != "unknown":
            parts.append(structured_doc.document_type.value)
        if structured_doc.parties:
            parts.append(" ".join(structured_doc.parties[:2]))
        query = " ".join(parts) if parts else "legal document key facts parties dates"

    logger.info(f"Generating draft for [{structured_doc.doc_id[:8]}], query: {query[:80]}...")

    retrieved: List[RetrievedChunk] = retriever.retrieve(
        query=query,
        top_k=config.MAX_EVIDENCE_CHUNKS,
        doc_id_filter=structured_doc.doc_id,
    )

    if not retrieved:
        logger.warning("No chunks retrieved — attempting without doc filter")
        retrieved = retriever.retrieve(query=query, top_k=config.MAX_EVIDENCE_CHUNKS)
        if not retrieved:
            warnings.append("No relevant evidence found in document. Draft is unsupported.")

    learned_patterns = []
    try:
        learned_patterns = get_patterns_for_doc_type(structured_doc.document_type.value)
    except Exception as e:
        logger.debug(f"Could not load patterns: {e}")

    prompt = build_generation_prompt(structured_doc, retrieved, learned_patterns)

    client = get_llm_client()
    try:
        raw_response = client.generate(prompt, temperature=0.15)
    except Exception as e:
        logger.error(f"LLM generation failed: {e}")
        warnings.append(f"LLM failed: {e}. Using minimal draft.")
        raw_response = _minimal_fallback_json(structured_doc)

    try:
        memo_data = _parse_llm_response(raw_response)
    except Exception as e:
        logger.error(f"Failed to parse LLM response: {e}")
        logger.debug(f"Raw response: {raw_response[:500]}")
        warnings.append(f"LLM output parse error: {e}")
        memo_data = _minimal_fallback_dict(structured_doc)

    sections = _build_sections(memo_data)
    section_chunk_ids = extract_chunk_ids_from_llm_output(memo_data)
    sections = link_citations(sections, retrieved, section_chunk_ids)

    re_field = memo_data.get("memo_subject") or (
        f"Case Fact Summary — {structured_doc.case_number or structured_doc.document_type.value.title()}"
    )

    memo = DraftMemo(
        memo_id=memo_id,
        doc_id=structured_doc.doc_id,
        re_field=re_field,
        sections=sections,
        total_evidence_chunks=len(retrieved),
        model_used=client.model_name(),
        generation_warnings=warnings,
    )

    grounding_pct = _calculate_grounding(sections)
    logger.info(
        f"Draft generated [{memo_id[:8]}]: {len(sections)} sections, "
        f"{len(retrieved)} evidence chunks, {grounding_pct:.0%} grounded"
    )
    return memo


def _parse_llm_response(text: str) -> dict:
    """Extract and parse JSON from LLM response text."""
    # Try direct parse first
    try:
        return json.loads(text.strip())
    except json.JSONDecodeError:
        pass

    # Try to find JSON block in markdown code fence
    fence_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fence_match:
        return json.loads(fence_match.group(1))

    # Try to find any JSON object
    brace_match = re.search(r"\{.*\}", text, re.DOTALL)
    if brace_match:
        return json.loads(brace_match.group(0))

    raise ValueError("No valid JSON found in LLM response")


def _build_sections(data: dict) -> List[DraftSection]:
    """Convert parsed LLM JSON into DraftSection objects."""
    sections = []

    # Executive Summary
    summary = data.get("executive_summary", "")
    if summary:
        sections.append(DraftSection(
            section_id="executive_summary",
            title="Executive Summary",
            content=summary,
        ))

    # Parties
    parties = data.get("parties", [])
    if parties:
        lines = []
        for p in parties:
            role = p.get("role", "Party")
            name = p.get("name", "[Name not identified]")
            lines.append(f"**{role}:** {name}")
        sections.append(DraftSection(
            section_id="parties",
            title="Parties",
            content="\n".join(lines),
        ))

    # Material Facts
    facts = data.get("material_facts", [])
    if facts:
        lines = []
        for i, f in enumerate(facts, 1):
            fact_text = f.get("fact", str(f)) if isinstance(f, dict) else str(f)
            lines.append(f"{i}. {fact_text}")
        sections.append(DraftSection(
            section_id="material_facts",
            title="Material Facts",
            content="\n".join(lines),
        ))

    # Key Dates
    dates = data.get("key_dates", [])
    if dates:
        lines = ["| Date | Event |", "|------|-------|"]
        for d in dates:
            date_val = d.get("date", "") if isinstance(d, dict) else str(d)
            event_val = d.get("event", "") if isinstance(d, dict) else ""
            lines.append(f"| {date_val} | {event_val} |")
        sections.append(DraftSection(
            section_id="key_dates",
            title="Key Dates & Deadlines",
            content="\n".join(lines),
        ))

    # Relevant Provisions
    provisions = data.get("relevant_provisions", [])
    if provisions:
        lines = []
        for p in provisions:
            title = p.get("title", "Provision") if isinstance(p, dict) else "Provision"
            excerpt = p.get("excerpt", "") if isinstance(p, dict) else str(p)
            lines.append(f"**{title}**\n> {excerpt}\n")
        sections.append(DraftSection(
            section_id="relevant_provisions",
            title="Relevant Clauses / Provisions",
            content="\n".join(lines),
        ))

    # Open Issues
    issues = data.get("open_issues", [])
    if issues:
        lines = [f"- {issue}" for issue in issues if issue]
        sections.append(DraftSection(
            section_id="open_issues",
            title="Open Issues / Flags",
            content="\n".join(lines) if lines else "None identified.",
        ))

    return sections


def _calculate_grounding(sections: List[DraftSection]) -> float:
    """Fraction of sections that have at least one evidence reference."""
    if not sections:
        return 0.0
    supported = sum(1 for s in sections if s.is_supported)
    return supported / len(sections)


def _minimal_fallback_json(doc: StructuredDocument) -> str:
    return json.dumps(_minimal_fallback_dict(doc))


def _minimal_fallback_dict(doc: StructuredDocument) -> dict:
    return {
        "memo_subject": f"Case Fact Summary — {doc.document_type.value.title()}",
        "executive_summary": f"Document type: {doc.document_type.value}. LLM analysis unavailable.",
        "parties": [{"role": "Plaintiff", "name": doc.plaintiff or "[Not identified]", "source": ""},
                    {"role": "Defendant", "name": doc.defendant or "[Not identified]", "source": ""}],
        "material_facts": [{"fact": "Full AI analysis requires a valid API key.", "source": ""}],
        "key_dates": [{"date": d, "event": "Date identified in document", "source": ""} for d in doc.all_dates[:3]],
        "relevant_provisions": [],
        "open_issues": ["Configure OPENAI_API_KEY or MISTRAL_API_KEY for full AI-powered analysis."],
    }
