"""
Feedback injector: formats learned patterns for injection into prompts.
Called by the draft generator to enrich future generation with operator preferences.
"""
from typing import List
from loguru import logger

from learning.pattern_store import get_patterns_for_doc_type


def get_prompt_injections(doc_type: str, max_patterns: int = 5) -> List[dict]:
    """
    Return the top operator-learned patterns formatted for prompt injection.

    Each item has:
      - description: what the operator prefers
      - example:     a before/after snippet
      - frequency:   how often this pattern was seen
    """
    patterns = get_patterns_for_doc_type(doc_type, limit=max_patterns * 2)

    # Deduplicate by description prefix (30 chars)
    seen_keys = set()
    deduplicated = []
    for p in patterns:
        key = p["description"][:30].lower()
        if key not in seen_keys:
            seen_keys.add(key)
            deduplicated.append(p)
        if len(deduplicated) >= max_patterns:
            break

    if deduplicated:
        logger.debug(
            f"Injecting {len(deduplicated)} learned patterns "
            f"for doc_type='{doc_type}'"
        )
    return deduplicated


def summarize_learning_state(doc_type: str = "all") -> dict:
    """
    Return a summary of the current learning state for display in the UI.
    """
    from learning.edit_capture import count_edits, get_all_edits
    from learning.pattern_store import get_all_patterns
    from collections import Counter

    all_patterns = get_all_patterns()
    total_edits = count_edits()

    # Pattern type distribution
    type_counts = Counter(p["pattern_type"] for p in all_patterns)

    # Most frequent patterns
    top_patterns = sorted(all_patterns, key=lambda p: p.get("frequency", 0), reverse=True)[:5]

    return {
        "total_edits_captured": total_edits,
        "total_patterns_learned": len(all_patterns),
        "pattern_type_distribution": dict(type_counts),
        "top_patterns": [
            {
                "type": p["pattern_type"],
                "description": p["description"],
                "frequency": p.get("frequency", 1),
            }
            for p in top_patterns
        ],
    }
