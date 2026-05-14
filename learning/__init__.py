"""Learning package."""
from learning.edit_capture import save_edit, get_edits_for_doc, get_all_edits
from learning.diff_analyzer import analyze_edit
from learning.pattern_store import process_edit_into_pattern, get_patterns_for_doc_type, get_all_patterns
from learning.feedback_injector import get_prompt_injections, summarize_learning_state

__all__ = [
    "save_edit",
    "get_edits_for_doc",
    "get_all_edits",
    "analyze_edit",
    "process_edit_into_pattern",
    "get_patterns_for_doc_type",
    "get_all_patterns",
    "get_prompt_injections",
    "summarize_learning_state",
]
