"""
Citation linker: maps each draft section's content back to the
evidence chunks that supported it.
"""
from typing import List, Dict, Optional
from retrieval.models import RetrievedChunk
from generation.models import EvidenceRef, DraftSection


def link_citations(
    sections: List[DraftSection],
    retrieved_chunks: List[RetrievedChunk],
    section_chunk_ids: Dict[str, List[str]],
) -> List[DraftSection]:
    """
    Populate evidence_refs for each section based on the chunk IDs
    the LLM reported using.

    Args:
        sections:           Draft sections to annotate.
        retrieved_chunks:   All chunks retrieved for this query.
        section_chunk_ids:  Map of section_id → list of chunk_ids used.

    Returns:
        Sections with populated evidence_refs and is_supported flags.
    """
    chunk_map: Dict[str, RetrievedChunk] = {
        r.chunk_id: r for r in retrieved_chunks
    }

    for section in sections:
        chunk_ids = section_chunk_ids.get(section.section_id, [])
        refs = []
        for cid in chunk_ids:
            if cid in chunk_map:
                rc = chunk_map[cid]
                excerpt = rc.text[:200].replace("\n", " ")
                refs.append(EvidenceRef(
                    chunk_id=cid,
                    source_label=rc.source_label,
                    excerpt=excerpt,
                    relevance_score=rc.score,
                ))
        section.evidence_refs = refs
        section.is_supported = len(refs) > 0

    return sections


def extract_chunk_ids_from_llm_output(raw_data: dict) -> Dict[str, List[str]]:
    """
    Parse the LLM JSON output and collect chunk_id references per section.
    Returns a dict mapping section_id → [chunk_ids].
    """
    result: Dict[str, List[str]] = {}

    # Executive summary — no per-item source in our schema
    result["executive_summary"] = []

    # Parties
    party_ids = []
    for item in raw_data.get("parties", []):
        if isinstance(item, dict):
            src = item.get("source", "")
            if src:
                party_ids.append(src)
    result["parties"] = party_ids

    # Material facts
    fact_ids = []
    for item in raw_data.get("material_facts", []):
        if isinstance(item, dict):
            src = item.get("source", "")
            if src:
                fact_ids.append(src)
    result["material_facts"] = fact_ids

    # Key dates
    date_ids = []
    for item in raw_data.get("key_dates", []):
        if isinstance(item, dict):
            src = item.get("source", "")
            if src:
                date_ids.append(src)
    result["key_dates"] = date_ids

    # Provisions
    prov_ids = []
    for item in raw_data.get("relevant_provisions", []):
        if isinstance(item, dict):
            src = item.get("source", "")
            if src:
                prov_ids.append(src)
    result["relevant_provisions"] = prov_ids

    # Open issues — no direct source citations
    result["open_issues"] = []

    return result
