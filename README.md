# LegalMind — Pearson Specter Litt Internal AI Workflow

> An end-to-end AI pipeline that ingests messy legal documents, extracts structured fields,
> retrieves grounded evidence, and generates editable draft memos — with a learning loop that
> improves from every operator edit.

---

## Features

| Capability | Implementation |
|---|---|
| **Messy document ingestion** | Mistral OCR (cloud) → Tesseract (local fallback) with OpenCV preprocessing |
| **Structured field extraction** | Regex first-pass + LLM enrichment for summary / subject matter |
| **Document classification** | Keyword-signature scoring (contract, notice, complaint, affidavit, memo…) |
| **Hybrid retrieval** | ChromaDB (dense) + BM25 (sparse) fused with Reciprocal Rank Fusion |
| **Grounded draft generation** | Evidence-cited Case Fact Summary Memo with `[CHUNK_ID]` references |
| **Multi-provider LLMs** | OpenAI GPT-4o → Mistral Large → Gemini 2.0 Flash → Ollama → Stub |
| **Operator learning loop** | SQLite edit history → semantic diff → frequency-weighted pattern injection |
| **REST API** | FastAPI with Swagger UI, health check, and full CORS support |
| **Operator dashboard** | Streamlit UI with inline edit, pattern viewer, and API key sidebar |

---

## Quick Start

### Prerequisites

```bash
# System dependencies (needed for local OCR fallback and PDF rendering)
sudo apt-get install -y tesseract-ocr poppler-utils
```

### 1. Install

```bash
cd legalMind
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Configure

```bash
cp .env.example .env
# Open .env and set at least ONE of:
#   OPENAI_API_KEY   — best quality drafts
#   MISTRAL_API_KEY  — also enables Mistral OCR
#   GEMINI_API_KEY   — fast, large-context generation
```

### 3. Run

Choose one of three interfaces:

#### A — Streamlit UI (Operator Dashboard)
```bash
streamlit run app.py
# → http://localhost:8501
```

#### B — FastAPI REST Server
```bash
uvicorn api:app --reload
# → http://localhost:8000/docs   (Swagger UI)
# → http://localhost:8000/redoc  (ReDoc)
```

#### C — Docker (Recommended — runs both together)
```bash
docker-compose up --build
# → API: http://localhost:8000/docs
# → UI:  http://localhost:8501
```

#### D — CLI
```bash
# Single document
python main.py --file data/sample_inputs/01_services_agreement.txt

# With custom retrieval query
python main.py --file contract.pdf --query "breach of contract damages"

# Demo mode (runs all sample documents)
python main.py --demo

# Evaluation suite
python main.py --evaluate

# Launch UI from CLI
python main.py --ui
```

---

## Docker

### Architecture

Two services share one image build (`legalmind:latest`):

```
┌─────────────────────────────────────────────────┐
│  docker-compose                                  │
│                                                  │
│  ┌──────────────────┐   ┌──────────────────────┐ │
│  │  legalmind-api   │   │   legalmind-ui       │ │
│  │  FastAPI :8000   │   │   Streamlit :8501    │ │
│  └────────┬─────────┘   └──────────────────────┘ │
│           │  (healthcheck before UI starts)        │
│           ▼                                        │
│  ┌──────────────────────────────────────────────┐ │
│  │  Named volume: legalmind_data                │ │
│  │  /app/data  (ChromaDB + SQLite + patterns)   │ │
│  └──────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────┘
```

### Commands

```bash
# Build and start both services
docker-compose up --build

# Start in background
docker-compose up -d --build

# View logs
docker-compose logs -f api
docker-compose logs -f ui

# Stop everything
docker-compose down

# Stop and wipe all volumes (clears DB, ChromaDB)
docker-compose down -v

# Rebuild without cache
docker-compose build --no-cache
```

### Volumes

| Volume / Mount | Purpose |
|---|---|
| `legalmind_data` (named) | Persists ChromaDB, SQLite edits, and pattern JSON across restarts |
| `./data/sample_inputs` (read-only) | Drop PDFs/TXTs here to make them available inside containers |
| `./sample_outputs` | Generated draft Markdown files written by the CLI |

---

## Project Structure

```
legalMind/
│
├── api.py                        # FastAPI REST interface (8 endpoints)
├── app.py                        # Streamlit operator dashboard
├── main.py                       # CLI: --file, --demo, --evaluate, --ui
├── config.py                     # All env-var settings with typed defaults
├── requirements.txt
├── Dockerfile                    # Multi-stage build (builder + runtime)
├── docker-compose.yml            # api + ui services, named volume, healthcheck
├── .dockerignore
├── .env.example                  # Template — copy to .env
│
├── ingestion/                    # STAGE 1: Ingestion & OCR
│   ├── document_loader.py        # File-type router (PDF/image/text)
│   ├── pdf_extractor.py          # Native text → Mistral OCR → Tesseract chain
│   ├── mistral_ocr.py            # Mistral Cloud OCR integration
│   ├── ocr_engine.py             # Local Tesseract wrapper + per-word confidence
│   ├── image_preprocessor.py     # OpenCV: deskew, denoise, adaptive threshold
│   └── models.py                 # ExtractedDocument, PageContent
│
├── extraction/                   # STAGE 2: Structured Field Extraction
│   ├── field_extractor.py        # Two-pass: regex + LLM enrichment
│   ├── document_classifier.py    # Keyword-score classifier → DocumentType enum
│   └── models.py                 # StructuredDocument, ExtractedField, DocumentType
│
├── retrieval/                    # STAGE 3: Hybrid Retrieval (RAG)
│   ├── chunker.py                # Sliding-window sentence chunker
│   ├── embedder.py               # all-MiniLM-L6-v2 (sentence-transformers)
│   ├── vector_store.py           # ChromaDB CRUD (upsert-idempotent)
│   ├── bm25_index.py             # BM25 keyword index (legal-aware tokeniser)
│   ├── hybrid_retriever.py       # RRF fusion: dense(0.6) + BM25(0.4)
│   └── models.py                 # TextChunk, RetrievedChunk
│
├── generation/                   # STAGE 4: Grounded Draft Generation
│   ├── draft_generator.py        # Pipeline orchestrator
│   ├── prompt_builder.py         # Evidence + metadata + patterns → full prompt
│   ├── llm_client.py             # Multi-model router: OpenAI → Mistral → Gemini → Ollama
│   ├── citation_linker.py        # Maps LLM chunk_id refs to real RetrievedChunks
│   └── models.py                 # DraftMemo, DraftSection, EvidenceRef
│
├── learning/                     # STAGE 5: Operator Feedback Loop
│   ├── edit_capture.py           # Persist before/after edits to SQLite
│   ├── diff_analyzer.py          # Classify edits into 9 semantic pattern types
│   ├── pattern_store.py          # Upsert + frequency-weight patterns (SQLite + JSON)
│   └── feedback_injector.py      # Format top-N patterns for prompt injection
│
├── data/
│   ├── sample_inputs/            # Drop test documents here
│   ├── chroma_db/                # Auto-created vector store (gitignored)
│   ├── edits.db                  # SQLite edit history (gitignored)
│   └── edit_patterns.json        # Human-readable pattern snapshot (gitignored)
│
├── scripts/
│   └── generate_samples.py       # Synthetic legal document generator (4 doc types)
│
├── tests/
│   ├── test_ingestion.py         # 10 unit tests
│   ├── test_extraction.py        # 11 unit tests
│   ├── test_retrieval.py         # Unit tests
│   ├── test_generation_learning.py # 12 unit tests
│   └── evaluation.py             # MRR, Recall@K, grounding rate, edit accuracy
│
├── sample_outputs/               # Generated draft Markdown files (gitignored)
└── docs/
    ├── architecture.md
    ├── assumptions_and_tradeoffs.md
    └── evaluation_report.md
```

---

## REST API Endpoints

| Method | Path | Description |
|---|---|---|
| `POST` | `/api/process` | Upload a document → run full pipeline → return draft memo |
| `GET` | `/api/documents/{doc_id}` | Fetch structured extraction result |
| `POST` | `/api/retrieve` | Run hybrid retrieval for a query against a processed doc |
| `GET` | `/api/draft/{memo_id}` | Fetch a previously generated draft |
| `POST` | `/api/edit` | Submit operator edits → persist + learn patterns |
| `GET` | `/api/patterns` | List all learned patterns (filterable by doc type) |
| `GET` | `/api/learning/state` | Learning system summary |
| `GET` | `/api/edits` | Raw edit history |
| `GET` | `/api/health` | Liveness / readiness probe |

Interactive docs: **http://localhost:8000/docs**

---

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `OPENAI_API_KEY` | — | OpenAI key (priority 1 LLM) |
| `OPENAI_MODEL` | `gpt-4o-mini` | OpenAI model name |
| `MISTRAL_API_KEY` | — | Mistral key (OCR + priority 2 LLM) |
| `MISTRAL_OCR_MODEL` | `mistral-ocr-latest` | Mistral OCR model |
| `USE_MISTRAL_OCR` | `true` | Toggle Mistral Cloud OCR |
| `GEMINI_API_KEY` | — | Gemini key (priority 3 LLM) |
| `GEMINI_MODEL` | `gemini-2.0-flash` | Gemini model name |
| `USE_OLLAMA_FALLBACK` | `false` | Enable local Ollama LLM |
| `OLLAMA_MODEL` | `llama3` | Ollama model name |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama server URL |
| `TESSERACT_CMD` | `/usr/bin/tesseract` | Path to Tesseract binary |
| `CHROMA_DB_PATH` | `./data/chroma_db` | ChromaDB storage path |
| `EDITS_DB_PATH` | `./data/edits.db` | SQLite edit history path |
| `PATTERNS_FILE` | `./data/edit_patterns.json` | Pattern JSON snapshot |
| `RETRIEVAL_TOP_K` | `8` | Chunks retrieved per query |
| `DENSE_WEIGHT` | `0.6` | RRF weight for dense retrieval |
| `BM25_WEIGHT` | `0.4` | RRF weight for BM25 retrieval |
| `CHUNK_SIZE` | `1000` | Characters per chunk |
| `CHUNK_OVERLAP` | `150` | Chunk overlap characters |
| `MAX_EVIDENCE_CHUNKS` | `6` | Max chunks injected into prompt |
| `MAX_PATTERNS_IN_PROMPT` | `5` | Max learned patterns per prompt |
| `LOG_LEVEL` | `INFO` | Logging verbosity |

---

## Pipeline Flow

```
Input File (PDF / Image / Text)
        │
        ▼
┌──────────────────────┐
│   1. INGESTION       │  Native text → Mistral OCR → Tesseract
│   + Preprocessing    │  OpenCV: deskew, denoise, threshold
└──────────┬───────────┘
           │ ExtractedDocument
           ▼
┌──────────────────────┐
│   2. EXTRACTION      │  Regex (case no., dates, money, parties)
│   Structured Fields  │  + LLM enrichment (subject matter, summary)
└──────────┬───────────┘
           │ StructuredDocument
           ▼
┌──────────────────────┐
│   3. RETRIEVAL       │  Sentence chunking → MiniLM embeddings
│   Hybrid RAG         │  ChromaDB (dense) + BM25 → RRF fusion
└──────────┬───────────┘
           │ RetrievedChunk[]
           ▼
┌──────────────────────┐
│   4. GENERATION      │  Evidence + metadata + operator patterns
│   Grounded Draft     │  → GPT-4o / Mistral / Gemini (JSON output)
│   + Citation Links   │  Each section cites source chunk IDs
└──────────┬───────────┘
           │ DraftMemo (Markdown + JSON)
           ▼
┌──────────────────────┐
│   5. LEARNING        │  Operator edits → SQLite → diff analysis
│   Improvement Loop   │  Pattern classification → frequency weighting
│                      │  → injected as few-shot rules next generation
└──────────────────────┘
```

---

## Draft Output Format

The system generates an **Internal Case Fact Summary Memo** with:

- **Executive Summary** — 2–3 sentence overview, grounded in evidence
- **Parties** — Plaintiff, defendant, counsel with source chunk citations
- **Material Facts** — Bulleted list; each fact cites its `[CHUNK_ID]`
- **Key Dates & Deadlines** — Table of dates with events and sources
- **Relevant Clauses / Provisions** — Quoted excerpts with chunk references
- **Open Issues / Flags** — Inferred gaps flagged `[UNVERIFIED]`

---

## Testing

```bash
# All unit tests
pytest tests/ -v --tb=short

# With coverage
pytest tests/ --cov=. --cov-report=term-missing

# Evaluation metrics (MRR, Recall@K, grounding rate)
python tests/evaluation.py

# Or via main CLI
python main.py --evaluate
```

---

## Development Notes

- **Adding a new LLM**: Implement the `LLMClient` protocol in `generation/llm_client.py` and add it to the priority chain in `get_llm_client()`.
- **Adding a new OCR engine**: Wrap it in `ingestion/` and update the priority chain in `pdf_extractor.py`.
- **Scaling the API**: The in-memory session store (`_documents`, `_drafts`) in `api.py` is single-process only. Migrate to Redis for multi-worker deployments.
- **ChromaDB persistence in Docker**: ChromaDB data lives in the `legalmind_data` named volume at `/app/data/chroma_db`. It persists across `docker-compose down` but is wiped by `docker-compose down -v`.
