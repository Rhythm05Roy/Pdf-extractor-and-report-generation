# LegalMind — Architecture Overview

## System Design Philosophy

LegalMind is built as a **staged pipeline with clean module boundaries**.
Each stage has a single responsibility, its own data model, and no hidden dependencies
on adjacent stages. This makes it easy to swap implementations (e.g., replace Tesseract
with AWS Textract, or ChromaDB with Pinecone) without touching other stages.

---

## Stage 1: Document Ingestion

**Entry:** Raw file (PDF, image, text)
**Exit:** `ExtractedDocument` with per-page text + OCR confidence

### How it works

1. `document_loader.py` detects file type and routes to the correct extractor.
2. For PDFs:
   - `pdf_extractor.py` tries native text extraction via `PyMuPDF` first.
   - If a page has fewer than 50 native characters, it renders the page to a 200 DPI PNG
     and falls back to Tesseract OCR.
3. For images: goes directly to `ocr_engine.py`.
4. Before OCR, `image_preprocessor.py` runs:
   - **Deskew** (Hough transform angle detection + rotation)
   - **Denoising** (fastNlMeans denoising)
   - **Adaptive thresholding** (handles uneven lighting on scanned pages)
   - **Morphological cleanup** (removes speckles)
5. `ocr_engine.py` runs `image_to_data()` for per-word confidence scoring.
6. Low-confidence pages (< 40%) are flagged in `ingestion_warnings`.

### Key design choice

Two-mode extraction (native text vs. OCR) avoids the cost of OCR on digital PDFs while
ensuring scanned documents are handled correctly. The 50-character threshold is conservative
by design — it's better to OCR a page that has some native text than to miss content.

---

## Stage 2: Structured Extraction

**Entry:** `ExtractedDocument`
**Exit:** `StructuredDocument` with typed fields + confidence scores

### Two-pass approach

**Pass 1 — Regex (fast, deterministic):**
- Case numbers, docket IDs: handles common legal formats (`CV-XXXX-XXX`, `1:24-cv-XXXXX`)
- Dates: 4 regex patterns covering US and ISO formats
- Monetary amounts: captures `$X,XXX.XX`, `USD`, `million` variants
- Parties: pattern matches near `Plaintiff:`, `Defendant:`, `v.`
- Counsel: matches near `Esq.`, `Attorney at Law`, `Bar No.`
- Key clauses: scans for 15 standard clause headers (WHEREAS, CONFIDENTIALITY, etc.)

**Pass 2 — LLM enrichment (when API available):**
- Subject matter (one sentence)
- Document summary (2–3 sentences)
- Uses only the first 3,000 characters to minimize token cost

### Document classification

Keyword signature scoring: each doc type has 2 regex patterns. Score = fraction matched.
The type with the highest score wins. Avoids ML model dependency for a task where
simple keyword matching is already highly accurate on legal documents.

---

## Stage 3: Retrieval

**Entry:** `ExtractedDocument` (for indexing) or query string (for retrieval)
**Exit:** `RetrievedChunk[]` with hybrid scores

### Chunking

Sliding window over sentences (not tokens), with configurable size and overlap.
Legal text uses periods and semicolons as natural boundaries. Very long sentences
are split at paragraph breaks.

### Embedding

`sentence-transformers/all-MiniLM-L6-v2` — 384-dimensional, fast inference,
L2-normalized for cosine similarity. Runs locally with no API calls.

### ChromaDB

Persistent local storage. Upsert is idempotent — re-indexing a document
doesn't create duplicates. HNSW index with cosine distance.

### BM25 (rank-bm25)

In-memory per session. Legal-aware tokenizer: preserves hyphenated terms,
removes legal stopwords. Complements dense retrieval for exact legal citation matches
(case numbers, statute references) where semantic similarity may be low.

### Hybrid Fusion (Reciprocal Rank Fusion)

RRF score = `dense_weight / (60 + dense_rank)` + `bm25_weight / (60 + bm25_rank)`

Weights: dense=0.6, BM25=0.4. These are configurable in `.env`.
RRF is preferred over score normalization because it's stable across different
score distributions from different retrievers.

---

## Stage 4: Generation

**Entry:** `StructuredDocument` + `HybridRetriever` + optional learned patterns
**Exit:** `DraftMemo` with sections and per-section evidence citations

### Prompt structure

```
[System instruction — role + grounding rules]
[Operator style rules — injected from learning stage]
[Document metadata — doc type, parties, dates, amounts]
[Evidence context — top-K retrieved chunks with IDs]
[Schema — exact JSON structure to return]
```

### Grounding enforcement

- LLM is instructed to cite chunks by their `[CHUNK_ID]` in output JSON.
- `citation_linker.py` walks the LLM's JSON output and maps cited IDs back
  to the actual retrieved chunks, populating `evidence_refs` on each section.
- Sections with no cited evidence are flagged with `is_supported=False`.

### LLM selection

Priority chain: `GeminiClient` → `OllamaClient` → `StubClient`.
`StubClient` returns a templated response — no AI content, but structurally valid.
This means the system degrades gracefully even without any API key.

---

## Stage 5: Improvement from Operator Edits

**Entry:** Before/after text per section
**Exit:** Classified patterns stored in SQLite + JSON; injected into future prompts

### Edit capture

`save_edit()` persists every non-trivial edit to SQLite with:
- `memo_id`, `doc_id`, `section_id`
- `original_text`, `edited_text`
- `doc_type`, `timestamp`, `operator_id`

### Pattern classification

`diff_analyzer.py` classifies each edit into one of 9 types:

| Type | Trigger |
|---|---|
| `citation_addition` | Added `per Exhibit X`, `p.N`, `[...]` references |
| `exhibit_reference` | Added `Exhibit A/B/...` |
| `qualifier_addition` | Added `alleged`, `purported`, `disputed` |
| `tone_formalization` | Substituted formal legal terms |
| `specificity_increase` | More proper nouns/numbers in edited |
| `structural_reformat` | Paragraph → bullet list |
| `expansion` | Length ratio > 1.4 |
| `condensation` | Length ratio < 0.6 |
| `generic_edit` | Catch-all |

### Pattern deduplication + frequency weighting

Patterns are upserted by `(pattern_type, doc_type, description[:80])` key.
Duplicate patterns increment `frequency` instead of creating new records.
Retrieval sorts by `frequency DESC` so the most-reinforced preferences appear first in prompts.

### Prompt injection

`feedback_injector.py` fetches the top-N patterns for the current document type
and prepends them as "OPERATOR STYLE RULES" in the next generation prompt.
This is few-shot learning at the prompt level — no model fine-tuning required.

---

## Data Flow Diagram

```
                         ┌─────────────┐
Raw File ────────────────► Ingestion   ├──────────────────────────────┐
                         └──────┬──────┘                              │
                                │ ExtractedDocument                   │
                         ┌──────▼──────┐                              │
                         │ Extraction  ├─────► StructuredDocument     │
                         └──────┬──────┘              │               │
                                │ text pages           │               │
                         ┌──────▼──────┐              │               │
                         │  Chunking   │              │               │
                         └──────┬──────┘              │               │
                                │ TextChunk[]          │               │
                    ┌───────────┤                      │               │
                    │           │                      │               │
             ┌──────▼──┐  ┌────▼─────┐               │               │
             │ ChromaDB│  │  BM25    │               │               │
             └──────┬──┘  └────┬─────┘               │               │
                    │           │                      │               │
             ┌──────▼───────────▼──┐                  │               │
             │  Hybrid Retriever   │◄─────────────────┘               │
             │  (RRF fusion)       │                                   │
             └──────────┬──────────┘                                   │
                        │ RetrievedChunk[]                              │
             ┌──────────▼──────────┐                                   │
             │    Prompt Builder   │◄─── Learned Patterns              │
             └──────────┬──────────┘                                   │
                        │ prompt string                                 │
             ┌──────────▼──────────┐                                   │
             │    Gemini / Ollama  │                                   │
             └──────────┬──────────┘                                   │
                        │ JSON response                                 │
             ┌──────────▼──────────┐                                   │
             │  Citation Linker    │                                   │
             └──────────┬──────────┘                                   │
                        │ DraftMemo                                     │
             ┌──────────▼──────────┐                                   │
             │  Operator UI / CLI  │◄──────────────────────────────────┘
             └──────────┬──────────┘
                        │ edited sections
             ┌──────────▼──────────┐
             │  Edit Capture +     │
             │  Pattern Learning   ├──► patterns injected next time
             └─────────────────────┘
```
