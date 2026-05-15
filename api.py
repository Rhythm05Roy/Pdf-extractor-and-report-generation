"""
api.py — FastAPI REST endpoints for LegalMind.

Endpoints
---------
POST /api/process          Upload + run the full pipeline. Returns draft memo.
GET  /api/documents/{id}   Fetch structured doc by doc_id.
POST /api/retrieve         Run retrieval for a query against a processed doc.
GET  /api/draft/{memo_id}  Fetch a previously generated draft by memo_id.
POST /api/edit             Submit operator edits; triggers pattern learning.
GET  /api/patterns         List all learned patterns.
GET  /api/learning/state   Summary of learning state (edits captured, patterns).
GET  /api/health           Health check.

Run with:
    uvicorn api:app --host 0.0.0.0 --port 8000 --reload
"""
import os
import uuid
import tempfile
from pathlib import Path
from typing import Optional, List, Dict, Any

from fastapi import FastAPI, File, UploadFile, HTTPException, Form, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from loguru import logger

# ── App ────────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="LegalMind API",
    description=(
        "Internal AI-powered legal document workflow for Pearson Specter Litt. "
        "Ingest messy legal documents, extract structured fields, retrieve grounded "
        "evidence, and generate first-pass draft memos."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── In-memory session store (single-process; swap for Redis in production) ────
_documents: Dict[str, Any] = {}   # doc_id → structured_doc dict
_drafts:    Dict[str, Any] = {}   # memo_id → draft dict
_retrievers: Dict[str, Any] = {}  # doc_id → HybridRetriever


# ── Pydantic schemas ───────────────────────────────────────────────────────────

class RetrieveRequest(BaseModel):
    doc_id: str = Field(..., description="doc_id returned by /api/process")
    query: str  = Field(..., description="Free-text retrieval query")
    top_k: int  = Field(6, ge=1, le=20, description="Number of chunks to return")


class EditRequest(BaseModel):
    memo_id:      str            = Field(..., description="memo_id of the draft being edited")
    doc_id:       str            = Field(..., description="doc_id of the source document")
    section_edits: List[Dict]   = Field(
        ...,
        description=(
            "List of {section_id, original_text, edited_text} dicts. "
            "Only sections with actual changes need be included."
        ),
        example=[
            {
                "section_id": "executive_summary",
                "original_text": "The contract was breached.",
                "edited_text": "The contract was materially breached on 12 March 2024.",
            }
        ],
    )
    notes: Optional[str] = Field(None, description="Optional reviewer notes")


class ProcessResponse(BaseModel):
    doc_id:           str
    memo_id:          str
    document_type:    str
    case_number:      Optional[str]
    total_pages:      int
    chars_extracted:  int
    avg_ocr_confidence: Optional[float]
    sections_count:   int
    evidence_chunks:  int
    model_used:       str
    generation_warnings: List[str]
    draft:            Dict[str, Any]
    structured:       Dict[str, Any]


# ── Helper ─────────────────────────────────────────────────────────────────────

def _run_pipeline(file_path: str, query: Optional[str] = None):
    """Run the full 4-stage pipeline and cache results."""
    from ingestion import load_document
    from extraction import extract_fields
    from retrieval import chunk_document, get_retriever
    from generation import generate_draft
    from generation.llm_client import reset_client

    reset_client()

    # Stage 1 — Ingestion
    extracted = load_document(file_path)

    # Stage 2 — Extraction
    structured = extract_fields(extracted)

    # Stage 3 — Indexing
    chunks = chunk_document(extracted)
    retriever = get_retriever()
    retriever.index_chunks(chunks)

    # Stage 4 — Draft generation
    draft = generate_draft(structured, retriever, query=query)

    # Cache
    _documents[extracted.doc_id]  = {"structured": structured, "extracted": extracted}
    _drafts[draft.memo_id]         = draft
    _retrievers[extracted.doc_id]  = retriever

    return extracted, structured, draft


# ── Endpoints ──────────────────────────────────────────────────────────────────

@app.get("/api/health", tags=["System"])
def health_check():
    """Liveness / readiness probe."""
    import config
    return {
        "status": "ok",
        "mistral_ocr": bool(config.MISTRAL_API_KEY),
        "openai":      bool(config.OPENAI_API_KEY),
        "gemini":      bool(config.GEMINI_API_KEY),
        "documents_cached": len(_documents),
        "drafts_cached":    len(_drafts),
    }


@app.post("/api/process", response_model=ProcessResponse, tags=["Pipeline"])
async def process_document(
    file:  UploadFile = File(..., description="Legal document (PDF, image, or text)"),
    query: Optional[str] = Form(None, description="Optional custom retrieval query"),
):
    """
    **Full pipeline in one call.**

    Accepts any supported file type (PDF, JPG, PNG, TIFF, TXT, MD).
    Returns the structured extraction result and the generated draft memo.
    Scanned PDFs are processed with Mistral OCR (if configured) or Tesseract.
    """
    allowed = {".pdf", ".jpg", ".jpeg", ".png", ".tiff", ".tif", ".txt", ".md"}
    suffix = Path(file.filename).suffix.lower()
    if suffix not in allowed:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported file type '{suffix}'. Allowed: {sorted(allowed)}",
        )

    # Write upload to temp file
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    try:
        extracted, structured, draft = _run_pipeline(tmp_path, query=query)
    except Exception as e:
        logger.error(f"Pipeline failed for {file.filename}: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        try:
            os.unlink(tmp_path)
        except Exception:
            pass

    return ProcessResponse(
        doc_id=extracted.doc_id,
        memo_id=draft.memo_id,
        document_type=structured.document_type.value,
        case_number=structured.case_number,
        total_pages=extracted.total_pages,
        chars_extracted=len(extracted.full_text),
        avg_ocr_confidence=extracted.avg_ocr_confidence,
        sections_count=len(draft.sections),
        evidence_chunks=draft.total_evidence_chunks,
        model_used=draft.model_used,
        generation_warnings=draft.generation_warnings,
        draft=draft.to_dict(),
        structured=structured.to_dict(),
    )


@app.get("/api/documents/{doc_id}", tags=["Pipeline"])
def get_document(doc_id: str):
    """Fetch structured extraction result for a previously processed document."""
    if doc_id not in _documents:
        raise HTTPException(
            status_code=404,
            detail=f"Document '{doc_id}' not found. Process it first via POST /api/process.",
        )
    structured = _documents[doc_id]["structured"]
    return structured.to_dict()


@app.post("/api/retrieve", tags=["Retrieval"])
def retrieve_evidence(req: RetrieveRequest):
    """
    Run retrieval over a processed document.

    Returns the top-k most relevant chunks, with scores for dense, BM25,
    and the fused hybrid score. Use this to inspect what evidence the
    draft generator would surface.
    """
    if req.doc_id not in _retrievers:
        raise HTTPException(
            status_code=404,
            detail=f"No retriever found for doc_id '{req.doc_id}'. Process the document first.",
        )
    retriever = _retrievers[req.doc_id]
    try:
        chunks = retriever.retrieve(
            query=req.query,
            top_k=req.top_k,
            doc_id_filter=req.doc_id,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return {
        "doc_id": req.doc_id,
        "query":  req.query,
        "top_k":  req.top_k,
        "results": [
            {
                "chunk_id":        c.chunk_id,
                "text":            c.text[:600],
                "score":           round(c.score, 4),
                "dense_score":     round(c.dense_score, 4),
                "bm25_score":      round(c.bm25_score, 4),
                "source_label":    c.source_label,
                "page_number":     c.page_number,
                "retrieval_method": c.retrieval_method,
            }
            for c in chunks
        ],
    }


@app.get("/api/draft/{memo_id}", tags=["Pipeline"])
def get_draft(memo_id: str):
    """
    Fetch a previously generated draft memo by its memo_id.
    Returns all sections with evidence references and citation details.
    """
    if memo_id not in _drafts:
        raise HTTPException(
            status_code=404,
            detail=f"Draft '{memo_id}' not found. Generate it via POST /api/process.",
        )
    return _drafts[memo_id].to_dict()


@app.post("/api/edit", tags=["Learning"])
def submit_edits(req: EditRequest):
    """
    **Operator edit capture + pattern learning.**

    Submit the edits an operator made to a draft. Each changed section is:
    - Persisted to SQLite (raw edit history)
    - Analyzed for reusable patterns (tone shift, citation addition, etc.)
    - Frequency-weighted so the most common patterns surface in future prompts

    Only sections with actual text changes need to be included.
    """
    from learning.pattern_store import process_edit_into_pattern

    # Get doc_type for better pattern scoping
    doc_type = "unknown"
    if req.doc_id in _documents:
        doc_type = _documents[req.doc_id]["structured"].document_type.value

    saved   = 0
    learned = 0
    pattern_ids = []

    for edit in req.section_edits:
        section_id    = edit.get("section_id", "unknown")
        original_text = edit.get("original_text", "")
        edited_text   = edit.get("edited_text", "")

        if not original_text or not edited_text:
            continue
        if original_text.strip() == edited_text.strip():
            continue

        pid = process_edit_into_pattern(
            memo_id=req.memo_id,
            doc_id=req.doc_id,
            section_id=section_id,
            original_text=original_text,
            edited_text=edited_text,
            doc_type=doc_type,
        )
        saved += 1
        if pid:
            learned += 1
            pattern_ids.append(pid)

    return {
        "status":        "ok",
        "edits_saved":   saved,
        "patterns_learned": learned,
        "pattern_ids":   pattern_ids,
        "notes":         req.notes,
        "message": (
            f"Saved {saved} edit(s) and extracted {learned} reusable pattern(s). "
            "Future drafts will incorporate these preferences."
        ),
    }


@app.get("/api/patterns", tags=["Learning"])
def list_patterns(
    doc_type: Optional[str] = Query(None, description="Filter by document type"),
    limit:    int            = Query(50, ge=1, le=200),
):
    """
    List all learned patterns, ordered by frequency.
    Optionally filter by document type (e.g. 'contract', 'notice', 'court_filing').
    """
    from learning.pattern_store import get_all_patterns, get_patterns_for_doc_type

    if doc_type:
        patterns = get_patterns_for_doc_type(doc_type, limit=limit)
    else:
        patterns = get_all_patterns()[:limit]

    return {
        "total":    len(patterns),
        "doc_type": doc_type or "all",
        "patterns": patterns,
    }


@app.get("/api/learning/state", tags=["Learning"])
def learning_state():
    """
    Summary of the learning system:
    - Total edits captured
    - Total patterns learned
    - Pattern type distribution
    - Top 5 most frequent patterns
    """
    try:
        from learning.feedback_injector import summarize_learning_state
        return summarize_learning_state()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/edits", tags=["Learning"])
def list_edits(limit: int = Query(50, ge=1, le=500)):
    """Raw edit history (most recent first)."""
    from learning.edit_capture import get_all_edits
    return {"edits": get_all_edits(limit=limit)}


# ── Startup ────────────────────────────────────────────────────────────────────

@app.on_event("startup")
async def startup_event():
    """Log startup info and verify critical dependencies."""
    import config
    logger.info("LegalMind API starting up…")
    logger.info(f"  Mistral OCR : {'enabled' if config.MISTRAL_API_KEY else 'disabled (set MISTRAL_API_KEY)'}")
    logger.info(f"  OpenAI LLM  : {'enabled' if config.OPENAI_API_KEY else 'disabled'}")
    logger.info(f"  Gemini LLM  : {'enabled' if config.GEMINI_API_KEY else 'disabled'}")
    logger.info("  Docs        : http://localhost:8000/docs")


# ── Run directly ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
