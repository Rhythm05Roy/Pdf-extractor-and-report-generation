"""
Pattern store: persists learned edit patterns to SQLite + JSON.
Patterns are keyed by type and doc_type, deduplicated by similarity,
and frequency-weighted so the most common ones surface first.
"""
import json
import uuid
import sqlite3
from datetime import datetime
from typing import List, Dict, Optional
from pathlib import Path
from loguru import logger

import config
from learning.edit_capture import _get_connection
from learning.diff_analyzer import analyze_edit, PatternType


def process_edit_into_pattern(
    memo_id: str,
    doc_id: str,
    section_id: str,
    original_text: str,
    edited_text: str,
    doc_type: str = "unknown",
) -> Optional[str]:
    """
    Full pipeline for a single operator edit:
      1. Save raw edit to SQLite
      2. Analyze diff → classify pattern
      3. Upsert pattern (increment frequency if similar exists)

    Returns:
        pattern_id if a reusable pattern was extracted, else None.
    """
    from learning.edit_capture import save_edit

    # Skip trivial edits (whitespace-only, identical)
    if original_text.strip() == edited_text.strip():
        logger.debug("Edit is identical — skipping")
        return None

    # Save raw edit
    save_edit(memo_id, doc_id, section_id, original_text, edited_text, doc_type)

    # Analyze
    pattern_type, description, example = analyze_edit(original_text, edited_text)

    if not description:
        return None

    # Upsert pattern
    pattern_id = _upsert_pattern(
        pattern_type=pattern_type,
        description=description,
        example_before=original_text[:200],
        example_after=edited_text[:200],
        doc_type=doc_type,
    )

    logger.info(
        f"Pattern learned: [{pattern_type}] '{description[:60]}' "
        f"(doc_type={doc_type})"
    )
    return pattern_id


def _upsert_pattern(
    pattern_type: str,
    description: str,
    example_before: str,
    example_after: str,
    doc_type: str,
) -> str:
    """
    Insert a new pattern or increment frequency of an existing similar one.
    Similarity check: same pattern_type + doc_type + description (first 80 chars).
    """
    conn = _get_connection()
    now = datetime.now().isoformat()
    desc_key = description[:80]

    try:
        # Check for existing similar pattern
        existing = conn.execute(
            """SELECT pattern_id, frequency FROM edit_patterns
               WHERE pattern_type = ? AND doc_type = ?
               AND SUBSTR(description, 1, 80) = ?""",
            (pattern_type, doc_type, desc_key),
        ).fetchone()

        if existing:
            pattern_id = existing["pattern_id"]
            conn.execute(
                """UPDATE edit_patterns
                   SET frequency = frequency + 1, last_seen = ?
                   WHERE pattern_id = ?""",
                (now, pattern_id),
            )
            conn.commit()
        else:
            pattern_id = str(uuid.uuid4())
            conn.execute(
                """INSERT INTO edit_patterns
                   (pattern_id, pattern_type, description, example_before,
                    example_after, doc_type, frequency, created_at, last_seen)
                   VALUES (?, ?, ?, ?, ?, ?, 1, ?, ?)""",
                (pattern_id, pattern_type, description,
                 example_before, example_after, doc_type, now, now),
            )
            conn.commit()
    finally:
        conn.close()

    # Keep JSON file in sync
    _sync_patterns_to_json()
    return pattern_id


def get_patterns_for_doc_type(doc_type: str, limit: int = 10) -> List[dict]:
    """
    Return top patterns for a given document type, ordered by frequency.
    Falls back to generic patterns if no doc-type-specific ones exist.
    """
    conn = _get_connection()
    try:
        rows = conn.execute(
            """SELECT pattern_type, description, example_before, example_after, frequency
               FROM edit_patterns
               WHERE doc_type = ? OR doc_type = 'unknown'
               ORDER BY frequency DESC, last_seen DESC
               LIMIT ?""",
            (doc_type, limit),
        ).fetchall()
        return [
            {
                "type": r["pattern_type"],
                "description": r["description"],
                "example": f'Before: "{r["example_before"][:80] if r["example_before"] else ""}" → '
                           f'After: "{r["example_after"][:80] if r["example_after"] else ""}"',
                "frequency": r["frequency"],
            }
            for r in rows
        ]
    finally:
        conn.close()


def get_all_patterns() -> List[dict]:
    """Return all stored patterns for inspection."""
    conn = _get_connection()
    try:
        rows = conn.execute(
            "SELECT * FROM edit_patterns ORDER BY frequency DESC"
        ).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


def _sync_patterns_to_json() -> None:
    """Keep a human-readable JSON copy of patterns in sync."""
    try:
        patterns = get_all_patterns()
        Path(config.PATTERNS_FILE).parent.mkdir(parents=True, exist_ok=True)
        with open(config.PATTERNS_FILE, "w", encoding="utf-8") as f:
            json.dump(patterns, f, indent=2, ensure_ascii=False)
    except Exception as e:
        logger.warning(f"Could not sync patterns to JSON: {e}")


def load_patterns_from_json() -> List[dict]:
    """Load patterns from the JSON snapshot (faster than SQLite for reads)."""
    try:
        if Path(config.PATTERNS_FILE).exists():
            with open(config.PATTERNS_FILE, encoding="utf-8") as f:
                return json.load(f)
    except Exception as e:
        logger.warning(f"Could not load patterns from JSON: {e}")
    return []
