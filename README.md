# LegalMind — Pearson Specter Litt Internal AI Workflow

> An end-to-end AI pipeline that ingests messy legal documents, extracts structured fields, retrieves grounded evidence, and generates editable draft memos — with an improvement loop that learns from operator edits.

---

## Quick Start

### Prerequisites

```bash
# System dependencies
sudo apt-get install -y tesseract-ocr poppler-utils

# Python 3.10+
python --version
```

### Installation

```bash
cd legalMind

# Copy and configure environment
cp .env.example .env
# Edit .env and add your GEMINI_API_KEY

# Install Python dependencies
pip install -r requirements.txt
```

### Generate Sample Documents

```bash
python scripts/generate_samples.py
```

### Run on a Document (CLI)

```bash
# Basic usage
python main.py --file data/sample_inputs/01_services_agreement.txt

# With custom retrieval query
python main.py --file data/sample_inputs/02_notice_of_default.txt \
               --query "breach of lease default cure period"

# Demo mode (runs on all sample docs)
python main.py --demo

# Run evaluation suite
python main.py --evaluate
```

### Launch the Streamlit UI

```bash
streamlit run app.py
# Opens at http://localhost:8501
```

### Run Tests

```bash
pytest tests/ -v --tb=short

# With coverage
pytest tests/ --cov=. --cov-report=term-missing
```

---

## Project Structure

```
legalMind/
├── app.py                       # Streamlit operator UI
├── main.py                      # CLI entrypoint
├── config.py                    # Central configuration
├── requirements.txt
├── .env.example
│
├── ingestion/                   # Stage 1: Document ingestion & OCR
│   ├── document_loader.py       # File router
│   ├── pdf_extractor.py         # PyMuPDF + OCR fallback
│   ├── ocr_engine.py            # Tesseract wrapper + confidence
│   ├── image_preprocessor.py   # OpenCV deskew/denoise
│   └── models.py                # ExtractedDocument, PageContent
│
├── extraction/                  # Stage 2: Structured field extraction
│   ├── field_extractor.py       # Regex + LLM two-pass extractor
│   ├── document_classifier.py   # Keyword-based doc type detection
│   └── models.py                # StructuredDocument, ExtractedField
│
├── retrieval/                   # Stage 3: Hybrid retrieval
│   ├── chunker.py               # Sliding-window text chunker
│   ├── embedder.py              # sentence-transformers wrapper
│   ├── vector_store.py          # ChromaDB interface
│   ├── bm25_index.py            # BM25 keyword index
│   ├── hybrid_retriever.py      # RRF fusion of dense + sparse
│   └── models.py                # TextChunk, RetrievedChunk
│
├── generation/                  # Stage 4: Grounded draft generation
│   ├── draft_generator.py       # Orchestrator
│   ├── prompt_builder.py        # Evidence + patterns → prompt
│   ├── llm_client.py            # Gemini → Ollama → Stub fallback
│   ├── citation_linker.py       # Maps sections to evidence chunks
│   └── models.py                # DraftMemo, DraftSection, EvidenceRef
│
├── learning/                    # Stage 5: Improvement from edits
│   ├── edit_capture.py          # SQLite edit store
│   ├── diff_analyzer.py         # Semantic edit classifier
│   ├── pattern_store.py         # Pattern dedup + frequency weighting
│   └── feedback_injector.py    # Formats patterns for prompt injection
│
├── data/
│   ├── sample_inputs/           # Sample legal documents
│   ├── chroma_db/               # Persistent vector store (auto-created)
│   ├── edits.db                 # SQLite edit history (auto-created)
│   └── edit_patterns.json       # Human-readable patterns snapshot
│
├── scripts/
│   └── generate_samples.py      # Synthetic document generator
│
├── tests/
│   ├── test_ingestion.py
│   ├── test_extraction.py
│   ├── test_retrieval.py
│   ├── test_generation_learning.py
│   └── evaluation.py            # Metrics: MRR, Recall@K, grounding rate
│
├── sample_outputs/              # Generated draft memos (auto-created)
└── docs/
    ├── architecture.md
    ├── assumptions_and_tradeoffs.md
    └── evaluation_report.md
```

---

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `GEMINI_API_KEY` | — | Google Gemini API key (required for full AI) |
| `USE_OLLAMA_FALLBACK` | `false` | Use local Ollama LLM if Gemini unavailable |
| `OLLAMA_MODEL` | `llama3` | Ollama model name |
| `TESSERACT_CMD` | `/usr/bin/tesseract` | Path to Tesseract binary |
| `CHROMA_DB_PATH` | `./data/chroma_db` | ChromaDB storage directory |
| `EDITS_DB_PATH` | `./data/edits.db` | SQLite edit history file |
| `RETRIEVAL_TOP_K` | `8` | Chunks retrieved per query |
| `CHUNK_SIZE` | `1000` | Characters per chunk |
| `CHUNK_OVERLAP` | `150` | Overlap between chunks |
| `LOG_LEVEL` | `INFO` | Logging verbosity |

---

## Pipeline at a Glance

```
Input File (PDF/Image/Text)
        │
        ▼
┌─────────────────────┐
│   1. INGESTION      │  PyMuPDF native text + Tesseract OCR fallback
│   + Preprocessing   │  OpenCV deskew, denoise, adaptive threshold
└──────────┬──────────┘
           │ ExtractedDocument
           ▼
┌─────────────────────┐
│   2. EXTRACTION     │  Regex first pass → LLM enrichment
│   Structured Fields │  Doc type, parties, dates, amounts, clauses
└──────────┬──────────┘
           │ StructuredDocument
           ▼
┌─────────────────────┐
│   3. RETRIEVAL      │  Sliding-window chunking → sentence-transformers
│   Hybrid Search     │  ChromaDB (dense) + BM25 (sparse) → RRF fusion
└──────────┬──────────┘
           │ RetrievedChunk[]
           ▼
┌─────────────────────┐
│   4. GENERATION     │  Evidence + metadata + patterns → LLM
│   Grounded Draft    │  Gemini 1.5 Flash (structured JSON output)
│   + Citation Links  │  Each section cites source chunks
└──────────┬──────────┘
           │ DraftMemo (Markdown + JSON)
           ▼
┌─────────────────────┐
│   5. LEARNING       │  Operator edits captured in SQLite
│   Improvement Loop  │  Diff analyzer → pattern classification
│                     │  Frequency-weighted pattern store
│                     │  Injected as few-shot examples next generation
└─────────────────────┘
```

---

## What Gets Generated

The system generates an **Internal Case Fact Summary Memo** in Markdown format with:

- Executive Summary (grounded)
- Parties (plaintiff, defendant, counsel)
- Material Facts (bulleted, each with source citation)
- Key Dates & Deadlines (table)
- Relevant Clauses / Provisions (quoted excerpts)
- Open Issues / Flags
- Supporting Evidence Index (every chunk used)

---

## Evaluation

```bash
python main.py --evaluate
```

Outputs scores for:
- **Extraction accuracy** (field F1 against ground truth)
- **Retrieval MRR** (Mean Reciprocal Rank)
- **Retrieval Recall@3**
- **Grounding rate** (% sections with evidence backing)
- **Edit classification accuracy**

See `docs/evaluation_report.md` for detailed results.
