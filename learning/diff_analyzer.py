"""
Diff analyzer: extracts semantic patterns from operator edits.
Identifies the *type* of change (citation_addition, tone_shift,
specificity_increase, structural_reformat, etc.) and stores
reusable descriptions.
"""
import re
import difflib
from typing import Optional, List, Tuple
from loguru import logger


# Pattern type taxonomy
class PatternType:
    CITATION_ADDITION = "citation_addition"       # operator added a source reference
    SPECIFICITY_INCREASE = "specificity_increase" # vague → specific (names, numbers, dates)
    TONE_FORMALIZATION = "tone_formalization"     # casual → formal language
    STRUCTURAL_REFORMAT = "structural_reformat"   # changed paragraph to list, etc.
    EXHIBIT_REFERENCE = "exhibit_reference"       # added "per Exhibit X"
    QUALIFIER_ADDITION = "qualifier_addition"     # added "alleged", "purported", "disputed"
    FACT_CORRECTION = "fact_correction"           # changed a stated fact
    EXPANSION = "expansion"                       # added significant new content
    CONDENSATION = "condensation"                 # removed/shortened content
    GENERIC_EDIT = "generic_edit"                 # catch-all


def analyze_edit(original: str, edited: str) -> Tuple[str, str, str]:
    """
    Classify an edit and generate a human-readable pattern description.

    Returns:
        (pattern_type, description, example_phrase)
    """
    original = original.strip()
    edited = edited.strip()

    if not original and not edited:
        return PatternType.GENERIC_EDIT, "Empty edit", ""

    ratio = difflib.SequenceMatcher(None, original, edited).ratio()

    citation_pattern = r"(per\s+exhibit|exhibit\s+[a-z0-9]+|p\.\s*\d+|page\s+\d+|\[.*?\])"
    added_citations = re.findall(citation_pattern, edited, re.IGNORECASE)
    original_citations = re.findall(citation_pattern, original, re.IGNORECASE)
    if len(added_citations) > len(original_citations):
        return (
            PatternType.CITATION_ADDITION,
            "When stating a fact, cite the specific exhibit, page, or document reference",
            f'"{_short(original)}" → "{_short(edited)}"',
        )

    if re.search(r"exhibit\s+[a-zA-Z0-9]+", edited, re.IGNORECASE) and \
       not re.search(r"exhibit\s+[a-zA-Z0-9]+", original, re.IGNORECASE):
        return (
            PatternType.EXHIBIT_REFERENCE,
            "Reference specific exhibits by letter/number when citing document sections",
            f'"{_short(original)}" → "{_short(edited)}"',
        )

    qualifiers = ["alleged", "purported", "disputed", "claimed", "contended",
                  "according to", "as stated in", "per the record"]
    added_quals = [q for q in qualifiers if q in edited.lower() and q not in original.lower()]
    if added_quals:
        return (
            PatternType.QUALIFIER_ADDITION,
            f"Add legal qualifiers ({', '.join(added_quals[:2])}) to unverified assertions",
            f'"{_short(original)}" → "{_short(edited)}"',
        )

    formal_indicators = {
        "governing agreement": ["contract", "deal"],
        "pursuant to": ["under", "according to"],
        "herein": ["here", "in this"],
        "therein": ["there", "in that"],
        "whereas": ["while", "since"],
        "notwithstanding": ["despite", "even though"],
    }
    for formal, informals in formal_indicators.items():
        if formal in edited.lower():
            for inf in informals:
                if inf in original.lower() and inf not in edited.lower():
                    return (
                        PatternType.TONE_FORMALIZATION,
                        f'Use formal legal terminology: "{formal}" instead of "{inf}"',
                        f'"{_short(original)}" → "{_short(edited)}"',
                    )

        # More numbers / proper nouns in edited version
    original_specifics = len(re.findall(r"\b\d+\b|[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+", original))
    edited_specifics = len(re.findall(r"\b\d+\b|[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+", edited))
    if edited_specifics > original_specifics + 1:
        return (
            PatternType.SPECIFICITY_INCREASE,
            "Replace vague references with specific names, dates, or dollar amounts from the document",
            f'"{_short(original)}" → "{_short(edited)}"',
        )

    original_bullets = original.count("\n-") + original.count("\n•")
    edited_bullets = edited.count("\n-") + edited.count("\n•")
    if edited_bullets > original_bullets + 1:
        return (
            PatternType.STRUCTURAL_REFORMAT,
            "Break dense paragraphs into bulleted lists for readability",
            "",
        )

    len_ratio = len(edited) / max(len(original), 1)
    if len_ratio > 1.4:
        return (
            PatternType.EXPANSION,
            "Expand terse statements with supporting context from the document",
            f'"{_short(original)}" → "{_short(edited)}"',
        )
    if len_ratio < 0.6:
        return (
            PatternType.CONDENSATION,
            "Condense verbose statements to focus on the key legal point",
            f'"{_short(original)}" → "{_short(edited)}"',
        )

    return (
        PatternType.GENERIC_EDIT,
        "Operator refined phrasing for clarity",
        f'"{_short(original)}" → "{_short(edited)}"',
    )


def _short(text: str, max_len: int = 60) -> str:
    """Truncate text for display in pattern examples."""
    text = text.replace("\n", " ").strip()
    return text[:max_len] + "..." if len(text) > max_len else text


def get_unified_diff(original: str, edited: str) -> str:
    """Return a unified diff of two texts."""
    original_lines = original.splitlines(keepends=True)
    edited_lines = edited.splitlines(keepends=True)
    diff = difflib.unified_diff(
        original_lines, edited_lines,
        fromfile="original", tofile="edited", lineterm=""
    )
    return "".join(diff)
