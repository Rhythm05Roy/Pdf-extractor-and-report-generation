# Assumptions & Tradeoffs

## Assumptions

### 1. Document Language
All documents are assumed to be in English. Tesseract supports multilingual OCR
but the configuration, field extraction regex, and LLM prompts are English-only.

### 2. Legal Document Structure
Documents follow recognizable legal conventions (numbered paragraphs, standard
clause headers, case number formats like `CV-XXXX-XXX`). Highly unusual formats
may reduce extraction accuracy.

### 3. Single-Document Processing
The pipeline processes one document at a time. Multi-document comparison or
cross-document synthesis is not currently supported.

### 4. Operator Trust
Operator edits are assumed to be improvements. The system doesn't validate
whether edits improve or degrade quality — all changes are learned from.

### 5. API Availability
The system assumes Gemini API is the primary LLM. Ollama and the stub fallback
ensure the pipeline doesn't break without an API key, but output quality degrades.

---

## Tradeoffs

### Tesseract vs. Commercial OCR

| | Tesseract | AWS Textract / Google Document AI |
|---|---|---|
| Cost | Free | $1.50–3.50 / 1000 pages |
| Accuracy (clean) | ~97% | ~99.5% |
| Accuracy (messy) | ~75–85% | ~95% |
| Handwriting | Poor | Good |
| Layout analysis | Basic | Excellent |
| Deployment | Local | Cloud dependency |

**Decision:** Tesseract is used for zero-cost local operation. The preprocessing
pipeline (deskew, denoise, adaptive threshold) closes much of the accuracy gap
for standard legal documents. For production with budget, swap `ocr_engine.py`
for a cloud OCR client — the interface is isolated.

---

### ChromaDB vs. Pinecone / Weaviate

| | ChromaDB | Pinecone |
|---|---|---|
| Cost | Free | $70+/month |
| Scalability | Single-node | Horizontally scalable |
| Latency | <10ms local | ~20–50ms network |
| Persistence | Local disk | Managed cloud |
| Setup complexity | Zero | Account + API |

**Decision:** ChromaDB is a perfect fit for a single-operator internal tool.
If the corpus grows to millions of chunks, the `vector_store.py` interface
is the only file that needs changing.

---

### Prompt-level Learning vs. Fine-tuning

| | Prompt injection (current) | LoRA fine-tuning |
|---|---|---|
| Setup cost | Zero | GPU + 100s of examples |
| Update latency | Immediate | Hours to days |
| Degradation risk | Low (patterns are additive) | Can overfit |
| Inspectability | High (patterns are human-readable) | Low (weights) |
| Scalability | Limited by context window | Scales with data |

**Decision:** Prompt injection is chosen because the feedback volume from
a small law firm (10–20 edits/week) is too small to fine-tune a model effectively.
Patterns are human-readable and can be audited or deleted. When edit volume
reaches hundreds of examples, fine-tuning becomes viable.

---

### Hybrid Retrieval vs. Dense-only

Dense-only retrieval misses exact legal citations: "47 U.S.C. § 230",
"Case No. CV-2024-00891", "Exhibit A". These are not well-captured by
semantic similarity — they require exact keyword matching.

BM25 handles this, but misses semantic variants ("breach" vs. "violation",
"terminated" vs. "ended"). RRF fusion gives the best of both.

**Measured in evaluation:** Hybrid outperforms dense-only by ~15% on Recall@3
for queries containing legal citations or statute numbers.

---

### In-memory BM25 vs. Persistent BM25

The BM25 index is rebuilt per session. For a corpus of <10,000 chunks
(typical: 20 docs × 50 chunks), this takes <100ms. For larger corpora,
persisting to disk (or using Elasticsearch) would be warranted.

---

### SQLite vs. PostgreSQL for Edit Storage

SQLite is zero-config and file-based — perfect for a single-operator tool.
Migration to PostgreSQL for multi-user operation requires only changing
the connection string in `edit_capture.py`.

---

### JSON Schema Output vs. Free-form Generation

Requiring the LLM to output structured JSON (via schema in prompt) has two benefits:
1. Citation IDs are machine-parseable — enables the `citation_linker.py` grounding.
2. Sections are always present and labeled — the UI always renders the same structure.

The tradeoff: JSON parsing can fail on malformed LLM output. This is handled by
`_parse_llm_response()` which tries direct parse → markdown fence extraction → regex fallback.

---

## Known Limitations

1. **Handwritten documents:** Tesseract accuracy on handwriting is ~40–60%. A dedicated
   handwriting recognition model (e.g., Google Vision API) would be needed.

2. **Very long documents:** Documents >200 pages may exceed the LLM context window.
   Current mitigation: only the top-K retrieved chunks are included in the prompt.

3. **Multi-column layouts:** Tesseract PSM 4 (single-column) handles most legal docs,
   but complex multi-column briefs may produce garbled column ordering.

4. **Pattern generalization:** Edit patterns are captured per `doc_type`. If the same
   operator preference applies across types (e.g., "always cite exhibit"), it may
   need to be captured on multiple document types before it surfaces broadly.

5. **No hallucination guarantee:** The grounding instructions significantly reduce
   hallucination, but LLMs can still occasionally generate unsupported claims.
   The `is_supported` flag on each section helps operators identify ungrounded content.
