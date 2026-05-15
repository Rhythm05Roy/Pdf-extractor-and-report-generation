# LegalMind — Architecture Overview

## System Design Philosophy

LegalMind is built as a **staged pipeline with clean module boundaries**. Each stage has a single responsibility, its own data model, and no hidden dependencies on adjacent stages. This makes it easy to swap implementations (e.g., replacing Tesseract with AWS Textract) without touching other modules.

---

## Stage 1: Document Ingestion & OCR

**Entry:** Raw file (PDF, image, text)  
**Exit:** `ExtractedDocument` with per-page text + OCR confidence

### Priority-Based Extraction
1.  **Digital PDF**: `pdf_extractor.py` first attempts native text extraction via `PyMuPDF`.
2.  **Mistral Cloud OCR (Primary Fallback)**: If a page is scanned or native text is insufficient, the system uses `mistral-ocr-latest`. This engine is superior for legal documents as it understands layout, tables, and complex formatting better than traditional OCR.
3.  **Local Tesseract (Secondary Fallback)**: If `USE_MISTRAL_OCR=false` or the API is unavailable, the system uses a local Tesseract engine with an OpenCV-based preprocessing pipeline (deskew, denoise, adaptive thresholding).

---

## Stage 2: Structured Extraction

**Entry:** `ExtractedDocument`  
**Exit:** `StructuredDocument` with typed fields + confidence scores

### Two-Pass Approach
*   **Pass 1 — Regex (Deterministic)**: Fast extraction of case numbers, monetary amounts, and standard dates.
*   **Pass 2 — LLM Enrichment**: Uses the priority LLM to extract subject matter and a high-level document summary.

---

## Stage 3: Retrieval (Hybrid RAG)

**Entry:** `ExtractedDocument` (for indexing) or query string (for retrieval)  
**Exit:** `RetrievedChunk[]` with hybrid scores

### Reciprocal Rank Fusion (RRF)
To handle legal queries—which often involve both semantic concepts (e.g., "negligence") and exact identifiers (e.g., "Exhibit B-4")—the system fuses results from two search engines:
1.  **Dense Retrieval**: `ChromaDB` with `all-MiniLM-L6-v2` embeddings for semantic similarity.
2.  **Sparse Retrieval**: `BM25` for keyword and identifier matching.

The final rank is calculated using the formula: `RRF score = sum(weight / (60 + rank_i))`.

---

## Stage 4: Generation

**Entry:** `StructuredDocument` + `HybridRetriever` + optional learned patterns  
**Exit:** `DraftMemo` with grounded sections

### Multi-Model Priority Chain
To ensure maximum availability, LegalMind uses a chain-of-command for LLM generation:
1.  **OpenAI (GPT-4o)**: Best-in-class reasoning and grounding.
2.  **Mistral (Large)**: Native integration with Mistral OCR; excellent for legal terminology.
3.  **Google Gemini (2.0 Flash)**: High speed and large context support.
4.  **Ollama (Local)**: Privacy-preserving fallback for offline environments.

---

## Stage 5: The Learning Loop (Improvement)

**Entry:** Before/after text per section  
**Exit:** Classified patterns injected into future prompts

### How it works
Every operator edit is captured and compared against the AI's original version. The `diff_analyzer.py` classifies the edit into semantic types:
*   **Citation Addition**: Operator added `[Exh. A]` or page references.
*   **Tone Formalization**: Operator replaced "contract" with "Governing Agreement".
*   **Qualifier Addition**: Added "alleged" or "purported".

These patterns are weighted by frequency and injected into the prompt as **Few-Shot Examples**, effectively fine-tuning the AI's behavior in real-time without retraining.

---

## Data Flow Diagram

```
       ┌─────────────┐        ┌─────────────┐
       │   CLI / UI  │◄──────►│  FastAPI    │
       └──────┬──────┘        └─────────────┘
              │                      │
              └──────────┬───────────┘
                         ▼
                  ┌─────────────┐
                  │ Ingestion   ├─────────┐
                  │ (Mistral/T) │         │
                  └──────┬──────┘         │
                         │                │
                  ┌──────▼──────┐         │
                  │ Extraction  │         │
                  └──────┬──────┘         │
                         │                │
                  ┌──────▼──────┐         │
                  │  Hybrid RAG │         │
                  │ (Chroma/BM) │         │
                  └──────┬──────┘         │
                         │                │
                  ┌──────▼──────┐         │
                  │ Generation  │◄────────┘ (Evidence Context)
                  │ (OA/Mist/Gem)│
                  └──────┬──────┘
                         │
                  ┌──────▼──────┐
                  │ Operator Edit│
                  └──────┬──────┘
                         │
                  ┌──────▼──────┐
                  │Learning Loop│
                  │ (SQLite)    ├─────────┐
                  └─────────────┘         │
                         ▲                │
                         └────────────────┘
                         (Patterns injected into Generation)
```
