"""
app.py — Streamlit UI for LegalMind
Operator workflow: Upload → Process → Review Draft → Edit → Save Feedback
"""
import sys
import json
import tempfile
import os
from pathlib import Path

import streamlit as st

st.set_page_config(
    page_title="LegalMind — Pearson Specter Litt",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

  html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

  .main-header {
    background: linear-gradient(135deg, #0f172a 0%, #1e3a5f 100%);
    padding: 2rem 2.5rem;
    border-radius: 12px;
    margin-bottom: 1.5rem;
    border: 1px solid #2d4a6e;
  }
  .main-header h1 { color: #e2e8f0; margin: 0; font-size: 2rem; font-weight: 700; }
  .main-header p  { color: #94a3b8; margin: 0.3rem 0 0; font-size: 0.95rem; }

  .metric-card {
    background: #1e293b;
    border: 1px solid #334155;
    border-radius: 10px;
    padding: 1rem 1.25rem;
    text-align: center;
  }
  .metric-card .label { color: #64748b; font-size: 0.8rem; text-transform: uppercase; letter-spacing: 0.05em; }
  .metric-card .value { color: #e2e8f0; font-size: 1.6rem; font-weight: 700; margin-top: 0.2rem; }

  .evidence-pill {
    display: inline-block;
    background: #1e3a5f;
    color: #93c5fd;
    border: 1px solid #2d4a6e;
    border-radius: 20px;
    padding: 2px 10px;
    font-size: 0.78rem;
    margin: 2px;
  }

  .warning-box {
    background: #422006;
    border: 1px solid #92400e;
    border-radius: 8px;
    padding: 0.75rem 1rem;
    color: #fcd34d;
    font-size: 0.85rem;
  }

  .section-card {
    background: #0f172a;
    border: 1px solid #1e293b;
    border-radius: 10px;
    padding: 1.25rem;
    margin-bottom: 1rem;
  }

  .pattern-badge {
    background: #064e3b;
    border: 1px solid #065f46;
    border-radius: 6px;
    padding: 0.4rem 0.8rem;
    color: #6ee7b7;
    font-size: 0.82rem;
    margin-bottom: 0.4rem;
    display: block;
  }

  .stTabs [data-baseweb="tab-list"] { gap: 8px; }
  .stTabs [data-baseweb="tab"] {
    background: #1e293b;
    border-radius: 8px 8px 0 0;
    color: #94a3b8;
    padding: 0.5rem 1.25rem;
  }
  .stTabs [aria-selected="true"] {
    background: #1e3a5f !important;
    color: #e2e8f0 !important;
  }
</style>
""", unsafe_allow_html=True)


def _init_state():
    defaults = {
        "extracted_doc": None,
        "structured_doc": None,
        "chunks": None,
        "retriever": None,
        "draft": None,
        "edit_saved": False,
        "processing": False,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init_state()


st.markdown("""
<div class="main-header">
  <h1>⚖️ LegalMind</h1>
  <p>Pearson Specter Litt — Internal AI Document Workflow &nbsp;|&nbsp; Confidential</p>
</div>
""", unsafe_allow_html=True)


with st.sidebar:
    st.markdown("### ⚙️ Configuration")

    # OpenAI key (priority 1)
    openai_key = st.text_input(
        "OpenAI API Key (priority 1)",
        value=os.getenv("OPENAI_API_KEY", ""),
        type="password",
        placeholder="sk-...",
        help="Get a key at platform.openai.com",
        key="openai_key_input",
    )
    if openai_key and not openai_key.startswith("your_"):
        os.environ["OPENAI_API_KEY"] = openai_key
        import config as _cfg
        _cfg.OPENAI_API_KEY = openai_key

    # Gemini key (priority 2)
    gemini_key = st.text_input(
        "Gemini API Key (priority 2)",
        value=os.getenv("GEMINI_API_KEY", ""),
        type="password",
        placeholder="AIza...",
        help="Get a key at aistudio.google.com",
        key="gemini_key_input",
    )
    if gemini_key and not gemini_key.startswith("your_"):
        os.environ["GEMINI_API_KEY"] = gemini_key
        import config as _cfg
        _cfg.GEMINI_API_KEY = gemini_key

    # Mistral key (priority 2 — OCR + LLM)
    mistral_key = st.text_input(
        "Mistral API Key (OCR + priority 2 LLM)",
        value=os.getenv("MISTRAL_API_KEY", ""),
        type="password",
        placeholder="5Chu...",
        help="Used for Mistral OCR (best quality) and as LLM fallback. Get a key at console.mistral.ai",
        key="mistral_key_input",
    )
    if mistral_key and not mistral_key.startswith("your_"):
        os.environ["MISTRAL_API_KEY"] = mistral_key
        import config as _cfg
        _cfg.MISTRAL_API_KEY = mistral_key

    if openai_key or gemini_key or mistral_key:
        from generation.llm_client import reset_client
        reset_client()

    st.divider()
    st.markdown("### 🧠 Learning State")
    try:
        from learning.feedback_injector import summarize_learning_state
        state = summarize_learning_state()
        col1, col2 = st.columns(2)
        col1.metric("Edits Captured", state["total_edits_captured"])
        col2.metric("Patterns Learned", state["total_patterns_learned"])

        if state["top_patterns"]:
            st.markdown("**Top Patterns:**")
            for p in state["top_patterns"][:3]:
                st.markdown(
                    f'<span class="pattern-badge">× {p["frequency"]} &nbsp; {p["description"][:55]}...</span>',
                    unsafe_allow_html=True,
                )
    except Exception:
        st.caption("No learning data yet")

    st.divider()
    st.markdown("### 📋 Quick Actions")
    if st.button("🗑 Clear Session", use_container_width=True):
        for k in ["extracted_doc", "structured_doc", "chunks", "retriever", "draft"]:
            st.session_state[k] = None
        st.rerun()


tab_upload, tab_draft, tab_edit, tab_evidence, tab_patterns = st.tabs([
    "📤 Upload & Process",
    "📄 Draft Memo",
    "✏️ Review & Edit",
    "🔍 Evidence Inspector",
    "📊 Learning Dashboard",
])


# TAB 1: Upload & Process
with tab_upload:
    st.markdown("#### Upload a Legal Document")
    st.caption("Accepts: PDF (native or scanned), images (JPG/PNG/TIFF), plain text")

    col_upload, col_opts = st.columns([2, 1])

    with col_upload:
        uploaded = st.file_uploader(
            "Drop file here or browse",
            type=["pdf", "jpg", "jpeg", "png", "tiff", "tif", "txt", "md"],
            key="doc_uploader",
        )

    with col_opts:
        custom_query = st.text_area(
            "Custom retrieval query (optional)",
            placeholder="e.g. breach of contract damages clause",
            height=80,
            key="custom_query",
        )
        use_sample = st.checkbox("Use bundled sample document", value=False)

    if use_sample:
        sample_dir = Path("data/sample_inputs")
        samples = list(sample_dir.glob("*.pdf")) + list(sample_dir.glob("*.txt"))
        if samples:
            chosen = st.selectbox("Choose sample:", [s.name for s in samples])
            sample_path = sample_dir / chosen
        else:
            st.warning("No sample documents found. Run `python scripts/generate_samples.py` first.")
            sample_path = None
    else:
        sample_path = None

    process_btn = st.button(
        "⚡ Process Document",
        type="primary",
        use_container_width=True,
        disabled=(not uploaded and not sample_path),
        key="process_btn",
    )

    if process_btn:
        # Determine file path
        if uploaded:
            suffix = Path(uploaded.name).suffix
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
            tmp.write(uploaded.getvalue())
            tmp.flush()
            file_path = tmp.name
            display_name = uploaded.name
        else:
            file_path = str(sample_path)
            display_name = sample_path.name

        with st.spinner("Running pipeline... this may take a moment for scanned PDFs"):
            try:
                from ingestion import load_document
                from extraction import extract_fields
                from retrieval import chunk_document, get_retriever
                from generation import generate_draft
                from generation.llm_client import reset_client

                reset_client()  # pick up any new API key from sidebar

                # Stage 1
                extracted = load_document(file_path)
                st.session_state.extracted_doc = extracted

                # Stage 2
                structured = extract_fields(extracted)
                st.session_state.structured_doc = structured

                # Stage 3
                chunks = chunk_document(extracted)
                retriever = get_retriever()
                retriever.index_chunks(chunks)
                st.session_state.chunks = chunks
                st.session_state.retriever = retriever

                # Stage 4
                query = custom_query.strip() or None
                draft = generate_draft(structured, retriever, query=query)
                st.session_state.draft = draft

                st.success(f"✅ **{display_name}** processed successfully!")

            except Exception as e:
                st.error(f"❌ Pipeline failed: {e}")
                import traceback
                st.code(traceback.format_exc(), language="text")

    # Show extraction summary if available
    if st.session_state.extracted_doc and st.session_state.structured_doc:
        ext = st.session_state.extracted_doc
        stt = st.session_state.structured_doc

        st.divider()
        st.markdown("#### 📊 Extraction Summary")

        c1, c2, c3, c4 = st.columns(4)
        c1.markdown(f'<div class="metric-card"><div class="label">Pages</div><div class="value">{ext.total_pages}</div></div>', unsafe_allow_html=True)
        c2.markdown(f'<div class="metric-card"><div class="label">Characters</div><div class="value">{len(ext.full_text):,}</div></div>', unsafe_allow_html=True)
        ocr_str = f"{ext.avg_ocr_confidence:.0%}" if ext.avg_ocr_confidence else "N/A"
        c3.markdown(f'<div class="metric-card"><div class="label">OCR Conf.</div><div class="value">{ocr_str}</div></div>', unsafe_allow_html=True)
        c4.markdown(f'<div class="metric-card"><div class="label">Doc Type</div><div class="value">{stt.document_type.value.title()}</div></div>', unsafe_allow_html=True)

        with st.expander("🔎 Extracted Fields"):
            col_l, col_r = st.columns(2)
            with col_l:
                st.markdown(f"**Case Number:** `{stt.case_number or 'Not found'}`")
                st.markdown(f"**Plaintiff:** {stt.plaintiff or 'Not found'}")
                st.markdown(f"**Defendant:** {stt.defendant or 'Not found'}")
                st.markdown(f"**Court:** {stt.court or 'Not found'}")
            with col_r:
                st.markdown(f"**Dates found:** {', '.join(stt.all_dates[:4]) or 'None'}")
                st.markdown(f"**Filing date:** {stt.filing_date or 'Not found'}")
                st.markdown(f"**Amounts:** {', '.join(stt.monetary_amounts[:3]) or 'None'}")
                st.markdown(f"**Clauses:** {len(stt.key_clauses)} identified")

            if stt.summary:
                st.markdown(f"**AI Summary:** {stt.summary}")

        if ext.ingestion_warnings:
            with st.expander("⚠️ Ingestion Warnings"):
                for w in ext.ingestion_warnings:
                    st.markdown(f'<div class="warning-box">⚠️ {w}</div>', unsafe_allow_html=True)


# TAB 2: Draft Memo
with tab_draft:
    draft = st.session_state.draft
    if draft is None:
        st.info("Upload and process a document first (Tab 1).")
    else:
        col_meta, col_btn = st.columns([3, 1])
        with col_meta:
            st.markdown(f"### 📄 {draft.re_field}")
            st.caption(
                f"Generated: {draft.generated_at[:19]} &nbsp;|&nbsp; "
                f"Model: `{draft.model_used}` &nbsp;|&nbsp; "
                f"Evidence chunks: {draft.total_evidence_chunks}"
            )
        with col_btn:
            md_text = draft.to_markdown()
            st.download_button(
                "⬇️ Download .md",
                data=md_text,
                file_name=f"draft_memo_{draft.memo_id[:8]}.md",
                mime="text/markdown",
                use_container_width=True,
            )

        if draft.generation_warnings:
            for w in draft.generation_warnings:
                st.warning(f"⚠️ {w}")

        st.divider()

        # Render each section
        for section in draft.sections:
            supported_icon = "✅" if section.is_supported else "⚠️"
            with st.expander(f"{supported_icon} {section.title}", expanded=True):
                st.markdown(section.content)

                if section.evidence_refs:
                    st.markdown("")
                    pills = "".join(
                        f'<span class="evidence-pill">📎 {ev.source_label}</span>'
                        for ev in section.evidence_refs
                    )
                    st.markdown(f"**Evidence:** {pills}", unsafe_allow_html=True)
                elif not section.is_supported:
                    st.markdown(
                        '<div class="warning-box">⚠️ No direct evidence found for this section.</div>',
                        unsafe_allow_html=True,
                    )

        st.divider()
        with st.expander("📄 Full Markdown (raw)"):
            st.code(md_text, language="markdown")


# TAB 3: Review & Edit (Operator edit capture)
with tab_edit:
    draft = st.session_state.draft
    structured = st.session_state.structured_doc

    if draft is None:
        st.info("Process a document first.")
    else:
        st.markdown("#### ✏️ Operator Review")
        st.caption(
            "Edit the draft sections below. Your edits are automatically captured "
            "and used to improve future drafts."
        )

        edits_made = {}

        for section in draft.sections:
            st.markdown(f"**{section.title}**")
            edited = st.text_area(
                label=f"Edit section: {section.title}",
                value=section.content,
                height=150,
                key=f"edit_{section.section_id}",
                label_visibility="collapsed",
            )
            edits_made[section.section_id] = {
                "original": section.content,
                "edited": edited,
            }
            st.markdown("")

        st.divider()
        col_save, col_notes = st.columns([1, 2])
        with col_notes:
            notes = st.text_input(
                "Notes for this review (optional)",
                placeholder="e.g. Added exhibit citations, formalized tone",
                key="review_notes",
            )
        with col_save:
            save_btn = st.button(
                "💾 Save Edits & Learn",
                type="primary",
                use_container_width=True,
                key="save_edits_btn",
            )

        if save_btn:
            from learning.pattern_store import process_edit_into_pattern
            doc_type = structured.document_type.value if structured else "unknown"
            saved_count = 0
            pattern_count = 0

            for section_id, texts in edits_made.items():
                orig = texts["original"]
                edited = texts["edited"]
                if orig.strip() == edited.strip():
                    continue  # no change
                pid = process_edit_into_pattern(
                    memo_id=draft.memo_id,
                    doc_id=draft.doc_id,
                    section_id=section_id,
                    original_text=orig,
                    edited_text=edited,
                    doc_type=doc_type,
                )
                saved_count += 1
                if pid:
                    pattern_count += 1

            if saved_count > 0:
                st.success(
                    f"✅ Saved {saved_count} edit(s), extracted {pattern_count} reusable pattern(s). "
                    "Future drafts will incorporate these preferences."
                )
                st.session_state.edit_saved = True
            else:
                st.info("No changes detected — nothing to save.")


# TAB 4: Evidence Inspector
with tab_evidence:
    draft = st.session_state.draft
    if draft is None:
        st.info("Process a document first.")
    else:
        st.markdown("#### 🔍 Evidence Inspector")
        st.caption("Inspect which source passages grounded each section of the draft.")

        # Collect all unique evidence refs
        all_evidence = {}
        for section in draft.sections:
            for ev in section.evidence_refs:
                if ev.chunk_id not in all_evidence:
                    all_evidence[ev.chunk_id] = {"ev": ev, "sections": []}
                all_evidence[ev.chunk_id]["sections"].append(section.title)

        if not all_evidence:
            st.warning("No evidence references found in this draft.")
        else:
            st.markdown(f"**{len(all_evidence)} unique evidence chunks** used across {len(draft.sections)} sections")
            st.divider()

            for cid, info in all_evidence.items():
                ev = info["ev"]
                sections_str = ", ".join(info["sections"])
                with st.expander(f"📎 {ev.source_label} &nbsp; (score: {ev.relevance_score:.3f})"):
                    st.caption(f"Used in: **{sections_str}**")
                    st.markdown(f"> {ev.excerpt}")
                    st.code(f"Chunk ID: {ev.chunk_id}", language="text")

        # Manual retrieval test
        st.divider()
        st.markdown("#### 🧪 Manual Retrieval Test")
        test_query = st.text_input(
            "Test query",
            placeholder="Enter a query to see what the retriever surfaces...",
            key="test_query",
        )
        if test_query and st.session_state.retriever:
            from retrieval.embedder import embed_single
            results = st.session_state.retriever.retrieve(test_query, top_k=5)
            for i, r in enumerate(results, 1):
                with st.expander(f"#{i} — {r.source_label} (score: {r.score:.3f})"):
                    st.caption(
                        f"Dense: {r.dense_score:.3f} &nbsp;|&nbsp; "
                        f"BM25: {r.bm25_score:.3f} &nbsp;|&nbsp; "
                        f"Method: {r.retrieval_method}"
                    )
                    st.markdown(f"> {r.text[:400]}")


# TAB 5: Learning Dashboard
with tab_patterns:
    st.markdown("#### 📊 Learning Dashboard")
    st.caption("Live view of patterns learned from operator edits.")

    try:
        from learning.feedback_injector import summarize_learning_state
        from learning.pattern_store import get_all_patterns
        from learning.edit_capture import get_all_edits

        state = summarize_learning_state()

        c1, c2, c3 = st.columns(3)
        c1.markdown(f'<div class="metric-card"><div class="label">Total Edits</div><div class="value">{state["total_edits_captured"]}</div></div>', unsafe_allow_html=True)
        c2.markdown(f'<div class="metric-card"><div class="label">Patterns Learned</div><div class="value">{state["total_patterns_learned"]}</div></div>', unsafe_allow_html=True)

        pattern_types = state.get("pattern_type_distribution", {})
        top_type = max(pattern_types, key=pattern_types.get) if pattern_types else "—"
        c3.markdown(f'<div class="metric-card"><div class="label">Top Pattern Type</div><div class="value" style="font-size:1rem">{top_type.replace("_", " ").title()}</div></div>', unsafe_allow_html=True)

        st.divider()

        # Pattern list
        st.markdown("#### 🧠 All Learned Patterns")
        patterns = get_all_patterns()
        if not patterns:
            st.info("No patterns yet. Process documents and save operator edits to start learning.")
        else:
            for p in patterns:
                with st.expander(
                    f"× {p.get('frequency', 1)} &nbsp; [{p['pattern_type']}] &nbsp; {p['description'][:70]}"
                ):
                    if p.get("example_before"):
                        st.markdown("**Before:**")
                        st.code(p["example_before"][:200], language="text")
                    if p.get("example_after"):
                        st.markdown("**After:**")
                        st.code(p["example_after"][:200], language="text")
                    st.caption(
                        f"Doc type: {p.get('doc_type', '?')} &nbsp;|&nbsp; "
                        f"Created: {p.get('created_at', '')[:10]} &nbsp;|&nbsp; "
                        f"Last seen: {p.get('last_seen', '')[:10]}"
                    )

        st.divider()

        # Raw edit history
        st.markdown("#### 📜 Edit History")
        edits = get_all_edits(limit=50)
        if not edits:
            st.info("No edit history yet.")
        else:
            for edit in edits[:20]:
                with st.expander(
                    f"[{edit['section_id']}] &nbsp; {edit['timestamp'][:16]} &nbsp; "
                    f"(doc: {edit['doc_id'][:8]}...)"
                ):
                    col_a, col_b = st.columns(2)
                    with col_a:
                        st.markdown("**Original:**")
                        st.code(edit["original_text"][:300], language="text")
                    with col_b:
                        st.markdown("**Edited:**")
                        st.code(edit["edited_text"][:300], language="text")

    except Exception as e:
        st.error(f"Could not load learning state: {e}")
