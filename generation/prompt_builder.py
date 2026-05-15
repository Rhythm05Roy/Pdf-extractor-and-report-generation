"""
Prompt builder for the generation stage.
Assembles evidence chunks, operator-learned patterns, and structured doc info
into a grounded prompt for the LLM.
"""
from typing import List, Optional
from retrieval.models import RetrievedChunk
from extraction.models import StructuredDocument
import config

SYSTEM_INSTRUCTION = """You are a junior associate at Pearson Specter Litt, a top-tier law firm.
Your task is to draft an internal Case Fact Summary Memo.

STRICT RULES:
1. ONLY state facts that are directly supported by the EVIDENCE CONTEXT provided below.
2. If a fact is not in the evidence, do NOT fabricate or assume it. Instead, flag it as [UNVERIFIED].
3. Every section must be grounded in the provided evidence.
4. Use formal legal memo language — precise, structured, professional.
5. Return ONLY valid JSON matching the exact schema specified.
"""

MEMO_SCHEMA = """{
  "executive_summary": "string (2-3 sentences grounded in evidence)",
  "parties": [{"role": "string", "name": "string", "source": "chunk_id"}],
  "material_facts": [{"fact": "string", "source": "chunk_id"}],
  "key_dates": [{"date": "string", "event": "string", "source": "chunk_id"}],
  "relevant_provisions": [{"title": "string", "excerpt": "string", "source": "chunk_id"}],
  "open_issues": ["string"],
  "memo_subject": "string"
}"""


def build_generation_prompt(
    structured_doc: StructuredDocument,
    retrieved_chunks: List[RetrievedChunk],
    learned_patterns: Optional[List[dict]] = None,
) -> str:
    """
    Assemble the full generation prompt.

    Structure:
    1. System instruction (role + grounding rules)
    2. Learned operator style patterns (if any)
    3. Structured document metadata (parties, dates, type)
    4. Evidence context (retrieved chunks with IDs)
    5. Output schema + generation instruction
    """
    parts = [SYSTEM_INSTRUCTION, ""]

    if learned_patterns:
        patterns = learned_patterns[:config.MAX_PATTERNS_IN_PROMPT]
        parts.append("=== OPERATOR STYLE RULES (learned from prior edits) ===")
        for i, p in enumerate(patterns, 1):
            desc = p.get("description", "")
            example = p.get("example", "")
            if desc:
                parts.append(f"{i}. {desc}")
            if example:
                parts.append(f"   Example: {example}")
        parts.append("")

    parts.append("=== DOCUMENT METADATA ===")
    parts.append(f"Document Type: {structured_doc.document_type.value}")
    parts.append(f"Case Number: {structured_doc.case_number or 'Not identified'}")
    if structured_doc.plaintiff:
        parts.append(f"Plaintiff: {structured_doc.plaintiff}")
    if structured_doc.defendant:
        parts.append(f"Defendant: {structured_doc.defendant}")
    if structured_doc.court:
        parts.append(f"Court: {structured_doc.court}")
    if structured_doc.all_dates:
        parts.append(f"Dates Found: {', '.join(structured_doc.all_dates[:5])}")
    if structured_doc.monetary_amounts:
        parts.append(f"Amounts: {', '.join(structured_doc.monetary_amounts[:4])}")
    parts.append("")

    parts.append("=== EVIDENCE CONTEXT ===")
    parts.append("The following passages are the ONLY basis for your draft.")
    parts.append("Reference them by their [CHUNK_ID] in your JSON output.\n")

    for chunk in retrieved_chunks[:config.MAX_EVIDENCE_CHUNKS]:
        parts.append(f"[{chunk.chunk_id}] Source: {chunk.source_label} (score: {chunk.score:.2f})")
        parts.append(f'"""{chunk.text[:600]}"""')
        parts.append("")

    parts.append("=== TASK ===")
    parts.append(
        "Using ONLY the evidence above, generate the Case Fact Summary Memo. "
        "Return ONLY valid JSON matching this schema exactly:\n"
    )
    parts.append(MEMO_SCHEMA)
    parts.append(
        "\nIMPORTANT: In 'source' fields, use the exact chunk_id from [CHUNK_ID] tags above. "
        "Do NOT include facts not found in the evidence."
    )

    return "\n".join(parts)
