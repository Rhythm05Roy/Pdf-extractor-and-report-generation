"""
Tests for the generation and learning stages.
"""
import sys
import json
import pytest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


class TestDiffAnalyzer:

    def test_citation_addition_detected(self):
        from learning.diff_analyzer import analyze_edit, PatternType
        original = "The contract was executed on January 15, 2024."
        edited = "Per Exhibit A (p.3), the contract was executed on January 15, 2024."
        pattern_type, description, example = analyze_edit(original, edited)
        assert pattern_type == PatternType.CITATION_ADDITION
        assert "cite" in description.lower() or "reference" in description.lower()

    def test_qualifier_addition_detected(self):
        from learning.diff_analyzer import analyze_edit, PatternType
        original = "The defendant committed fraud."
        edited = "The defendant allegedly committed fraud."
        pattern_type, description, _ = analyze_edit(original, edited)
        assert pattern_type == PatternType.QUALIFIER_ADDITION

    def test_tone_formalization_detected(self):
        from learning.diff_analyzer import analyze_edit, PatternType
        original = "Under the contract, both parties must comply."
        edited = "Pursuant to the governing agreement, both parties must comply."
        pattern_type, description, _ = analyze_edit(original, edited)
        assert pattern_type == PatternType.TONE_FORMALIZATION

    def test_identical_edit_is_generic(self):
        from learning.diff_analyzer import analyze_edit, PatternType
        text = "Some legal text."
        # slightly different but generic
        pattern_type, desc, _ = analyze_edit(text, text + " Additionally.")
        assert desc != ""  # should produce some description

    def test_unified_diff(self):
        from learning.diff_analyzer import get_unified_diff
        diff = get_unified_diff("line one\nline two", "line one\nline three")
        assert "-line two" in diff
        assert "+line three" in diff


class TestEditCapture:

    def test_save_and_retrieve_edit(self, tmp_path):
        import config
        config.EDITS_DB_PATH = str(tmp_path / "test_edits.db")

        from learning.edit_capture import save_edit, get_edits_for_doc
        edit_id = save_edit(
            memo_id="memo-001",
            doc_id="doc-001",
            section_id="material_facts",
            original_text="The contract was signed.",
            edited_text="Per Exhibit A, the contract was signed on January 15, 2024.",
            doc_type="contract",
        )
        assert edit_id is not None
        edits = get_edits_for_doc("doc-001")
        assert len(edits) == 1
        assert edits[0]["section_id"] == "material_facts"

    def test_count_edits(self, tmp_path):
        import config
        config.EDITS_DB_PATH = str(tmp_path / "test_edits2.db")

        from learning.edit_capture import save_edit, count_edits
        save_edit("m1", "d1", "s1", "orig", "edited", "contract")
        save_edit("m2", "d1", "s2", "orig2", "edited2", "contract")
        assert count_edits() == 2


class TestPatternStore:

    def test_process_edit_creates_pattern(self, tmp_path):
        import config
        config.EDITS_DB_PATH = str(tmp_path / "ps_edits.db")
        config.PATTERNS_FILE = str(tmp_path / "patterns.json")

        from learning.pattern_store import process_edit_into_pattern, get_all_patterns
        pid = process_edit_into_pattern(
            memo_id="memo-x",
            doc_id="doc-x",
            section_id="material_facts",
            original_text="The contract was signed on that date.",
            edited_text="Per Exhibit A (p.2), the contract was executed on January 15, 2024.",
            doc_type="contract",
        )
        assert pid is not None
        patterns = get_all_patterns()
        assert len(patterns) >= 1

    def test_frequency_increments(self, tmp_path):
        import config
        config.EDITS_DB_PATH = str(tmp_path / "freq_edits.db")
        config.PATTERNS_FILE = str(tmp_path / "freq_patterns.json")

        from learning.pattern_store import process_edit_into_pattern, get_all_patterns
        # Same pattern type twice
        for _ in range(2):
            process_edit_into_pattern(
                memo_id="memo-y", doc_id="doc-y", section_id="parties",
                original_text="The defendant is accused.",
                edited_text="The defendant allegedly committed the act.",
                doc_type="complaint",
            )
        patterns = get_all_patterns()
        freq = patterns[0].get("frequency", 1)
        assert freq >= 2

    def test_get_patterns_for_doc_type(self, tmp_path):
        import config
        config.EDITS_DB_PATH = str(tmp_path / "dtype_edits.db")
        config.PATTERNS_FILE = str(tmp_path / "dtype_patterns.json")

        from learning.pattern_store import process_edit_into_pattern, get_patterns_for_doc_type
        process_edit_into_pattern(
            "m", "d", "s",
            "contract was signed", "Per Exhibit A, the governing agreement was executed",
            "contract",
        )
        patterns = get_patterns_for_doc_type("contract")
        assert isinstance(patterns, list)


class TestDraftModels:

    def test_draft_memo_to_markdown(self):
        # Import directly from module file to avoid __init__ chain → embedder → sklearn
        import importlib.util, sys
        spec = importlib.util.spec_from_file_location(
            "generation.models",
            str(Path(__file__).parent.parent / "generation" / "models.py")
        )
        gen_models = importlib.util.module_from_spec(spec)
        sys.modules["generation.models"] = gen_models
        spec.loader.exec_module(gen_models)
        DraftMemo = gen_models.DraftMemo
        DraftSection = gen_models.DraftSection
        memo = DraftMemo(
            memo_id="test-memo-001",
            doc_id="test-doc-001",
            re_field="Test Case Summary",
            sections=[
                DraftSection(
                    section_id="executive_summary",
                    title="Executive Summary",
                    content="This is a test summary.",
                )
            ],
            model_used="test-model",
        )
        md = memo.to_markdown()
        assert "MEMORANDUM" in md
        assert "Executive Summary" in md
        assert "Test Case Summary" in md

    def test_draft_memo_to_dict(self):
        import importlib.util, sys
        spec = importlib.util.spec_from_file_location(
            "generation.models",
            str(Path(__file__).parent.parent / "generation" / "models.py")
        )
        gen_models = importlib.util.module_from_spec(spec)
        sys.modules.setdefault("generation.models", gen_models)
        spec.loader.exec_module(gen_models)
        DraftMemo = gen_models.DraftMemo
        DraftSection = gen_models.DraftSection
        memo = DraftMemo(
            memo_id="test-memo-002",
            doc_id="test-doc-002",
            re_field="Re: Test",
            sections=[],
            model_used="gemini-test",
        )
        d = memo.to_dict()
        assert d["memo_id"] == "test-memo-002"
        assert isinstance(d["sections"], list)
