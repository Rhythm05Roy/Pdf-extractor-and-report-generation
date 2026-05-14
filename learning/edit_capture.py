"""
Edit capture: stores operator before/after edits in SQLite.
Every edit record is the atomic unit of the improvement loop.
"""
import sqlite3
import json
import uuid
from datetime import datetime
from typing import Optional, List
from loguru import logger
from pathlib import Path

import config


def _get_connection() -> sqlite3.Connection:
    Path(config.EDITS_DB_PATH).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(config.EDITS_DB_PATH)
    conn.row_factory = sqlite3.Row
    _initialize_schema(conn)
    return conn


def _initialize_schema(conn: sqlite3.Connection) -> None:
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS edits (
            edit_id       TEXT PRIMARY KEY,
            memo_id       TEXT NOT NULL,
            doc_id        TEXT NOT NULL,
            section_id    TEXT NOT NULL,
            original_text TEXT NOT NULL,
            edited_text   TEXT NOT NULL,
            doc_type      TEXT DEFAULT 'unknown',
            timestamp     TEXT NOT NULL,
            operator_id   TEXT DEFAULT 'operator',
            notes         TEXT DEFAULT ''
        );

        CREATE TABLE IF NOT EXISTS edit_patterns (
            pattern_id    TEXT PRIMARY KEY,
            pattern_type  TEXT NOT NULL,
            description   TEXT NOT NULL,
            example_before TEXT,
            example_after  TEXT,
            doc_type      TEXT DEFAULT 'unknown',
            frequency     INTEGER DEFAULT 1,
            created_at    TEXT NOT NULL,
            last_seen     TEXT NOT NULL
        );

        CREATE INDEX IF NOT EXISTS idx_edits_doc_id ON edits(doc_id);
        CREATE INDEX IF NOT EXISTS idx_edits_section ON edits(section_id);
        CREATE INDEX IF NOT EXISTS idx_patterns_type ON edit_patterns(pattern_type);
    """)
    conn.commit()


def save_edit(
    memo_id: str,
    doc_id: str,
    section_id: str,
    original_text: str,
    edited_text: str,
    doc_type: str = "unknown",
    operator_id: str = "operator",
    notes: str = "",
) -> str:
    """
    Persist a single operator edit.

    Returns:
        edit_id of the saved record.
    """
    edit_id = str(uuid.uuid4())
    timestamp = datetime.now().isoformat()

    conn = _get_connection()
    try:
        conn.execute(
            """INSERT INTO edits
               (edit_id, memo_id, doc_id, section_id, original_text,
                edited_text, doc_type, timestamp, operator_id, notes)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (edit_id, memo_id, doc_id, section_id, original_text,
             edited_text, doc_type, timestamp, operator_id, notes),
        )
        conn.commit()
        logger.debug(f"Edit saved: {edit_id[:8]} (section={section_id})")
    finally:
        conn.close()

    return edit_id


def get_edits_for_doc(doc_id: str) -> List[dict]:
    """Return all edits for a given document."""
    conn = _get_connection()
    try:
        rows = conn.execute(
            "SELECT * FROM edits WHERE doc_id = ? ORDER BY timestamp DESC",
            (doc_id,),
        ).fetchall()
        return [dict(row) for row in rows]
    finally:
        conn.close()


def get_all_edits(limit: int = 200) -> List[dict]:
    """Return recent edits across all documents."""
    conn = _get_connection()
    try:
        rows = conn.execute(
            "SELECT * FROM edits ORDER BY timestamp DESC LIMIT ?",
            (limit,),
        ).fetchall()
        return [dict(row) for row in rows]
    finally:
        conn.close()


def count_edits() -> int:
    conn = _get_connection()
    try:
        return conn.execute("SELECT COUNT(*) FROM edits").fetchone()[0]
    finally:
        conn.close()
