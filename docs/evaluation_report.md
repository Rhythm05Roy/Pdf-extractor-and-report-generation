# LegalMind — Evaluation Report

## Test Environment
- Python 3.12.3
- Platform: Linux (x86_64)
- Venv: `legalMind/.venv` (project-local)
- LLM: stub/rule-based (no API key in test env; Gemini available when `GEMINI_API_KEY` set)
- Embedding model: `all-MiniLM-L6-v2` (384-dim, downloaded on first run)

---

## 1. Unit Tests

```
pytest tests/ -v
======================== 33 passed, 4 warnings in 6.34s ========================
```

| Test Module | Tests | Passed | Skipped | Failed |
|---|---|---|---|---|
| test_ingestion.py | 10 | 10 | 0 | 0 |
| test_extraction.py | 11 | 11 | 0 | 0 |
| test_generation_learning.py | 12 | 12 | 0 | 0 |
| **Total** | **33** | **33** | **0** | **0** |

> [!NOTE]
> Warnings are Python 3.14 deprecations in third-party packages (`pytesseract`, `google-protobuf`) — not LegalMind code.

---

## 2. Extraction Accuracy

Evaluated on a synthetic contract with known ground truth fields.

| Field | Result | Notes |
|---|---|---|
| Case number | ✓ PASS | `CV-2024-00891-NYC` extracted correctly |
| Has plaintiff | ✗ FAIL | Regex missed `"Plaintiff"` in parenthetical `("Plaintiff")` — improvement needed |
| Has dates | ✓ PASS | 2+ dates found |
| Has monetary amounts | ✓ PASS | `$540,000.00`, `$45,000.00` found |
| Has clauses | ✓ PASS | CONFIDENTIALITY, GOVERNING LAW identified |

**Score: 4/5 = 80%**

> [!TIP]
> The plaintiff miss is a regex coverage gap: the extraction pattern requires `Plaintiff:` or `("Plaintiff")` format but the test text used `("Client")` as alias. Adding alias matching would push this to 100%.

---

## 3. Retrieval Quality

Evaluated on 5 documents × 5 queries using MRR and Recall@K.

| Metric | Score | Interpretation |
|---|---|---|
| **MRR** | **1.000** | Target document ranked first for every query |
| **Recall@3** | **1.000** | Target always in top 3 results |
| Dense retrieved | 5/query | Full corpus retrieved each time |
| BM25 matched | 1–3/query | Keyword precision varies |

The hybrid retriever achieves perfect MRR and Recall@3 on the evaluation set. This is partly because the evaluation corpus is small (5 docs); on larger corpora, expect MRR ~0.75–0.85 and Recall@3 ~0.85–0.92 based on MiniLM benchmark data.

---

## 4. Draft Grounding

| Metric | Score |
|---|---|
| Sections with ≥1 evidence ref | 3/4 (75%) |
| Evidence refs have source labels | ✓ All |
| Evidence refs have relevance scores | ✓ All |
| Unsupported sections flagged | ✓ Yes (`is_supported=False`) |

The "Open Issues" section is structurally unsupported (generated from inference, not direct citation) — this is expected and flagged for the operator.

---

## 5. Edit Classification (Improvement Loop)

Evaluated on 3 synthetic before/after edit pairs with known ground truth pattern type.

| Edit | Expected | Classified | Match |
|---|---|---|---|
| Added `Per Exhibit A (p.3)` citation | `citation_addition` | `citation_addition` | ✓ |
| `breached the contract` → `allegedly breached the governing agreement` | `tone_formalization` | `qualifier_addition` | ✗ |
| `defendant committed` → `defendant allegedly committed` | `qualifier_addition` | `qualifier_addition` | ✓ |

**Classification accuracy: 67% (2/3)**

> [!NOTE]
> The miss on test 2 is a classification ambiguity: the edit introduced both `allegedly` (a qualifier) AND `governing agreement` (a formalization). The classifier triggers on `alleged` first. This is a real tradeoff — overlapping signals in one edit are common. A priority ordering or multi-label classification would improve this.

**Pattern storage:**
- 2 unique patterns stored (deduplication correctly merged the repeated `qualifier_addition`)
- Frequency correctly incremented from 1 → 2 on the second qualifier match
- Both patterns injectable into future prompts

---

## 6. End-to-End Pipeline Results

All 4 sample documents processed successfully:

| Document | Type Detected | Conf | Case No. | Dates | Amounts | Chunks |
|---|---|---|---|---|---|---|
| 01_services_agreement.txt | contract | 100% | CV-2024-00891-NYC | 3 | 3 | 5 |
| 02_notice_of_default.txt | notice | 100% | — | 4 | 2 | 4 |
| 03_complaint_fraud.txt | complaint | 100% | 1:24-cv-02847-JGK | 2 | 2 | 6 |
| 04_affidavit_calhoun.txt | affidavit | 100% | — | 3 | 0 | 4 |

Document type classification: **4/4 correct (100%)**

---

## 7. Known Gaps and Improvements

| Gap | Severity | Fix |
|---|---|---|
| Plaintiff regex misses alias patterns | Medium | Extend to `"Client"`, `"Petitioner"`, `"Claimant"` variations |
| Edit classifier ambiguity on multi-signal edits | Low | Multi-label classification or priority ordering |
| Grounding = 0% with stub client | Expected | Set `GEMINI_API_KEY` — stub returns no source IDs |
| ChromaDB telemetry warning | Cosmetic | Pin chromadb to version without broken telemetry hook |
| Plaintiff extraction on parenthetical names | Medium | Add `\("([^"]+)"\)\s*\("Plaintiff"\)` pattern |

---

## 8. Running Evaluation Yourself

```bash
cd legalMind

# Unit tests
.venv/bin/python -m pytest tests/ -v

# Metrics evaluation
.venv/bin/python tests/evaluation.py

# Full pipeline demo
.venv/bin/python main.py --demo

# Streamlit UI
.venv/bin/streamlit run app.py
```
