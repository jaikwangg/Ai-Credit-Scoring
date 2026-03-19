# AI Credit Scoring Assistant — Research Knowledge Base
## Task A: Source Discovery — Output Package

**Compiled:** 2026-03-17
**Geography:** Thailand (primary) + Global (context)
**Total Sources:** 38

---

## Files in This Package

| File | Description |
|------|-------------|
| `task_a_sources.json` | Master source inventory — full JSON schema with all metadata, key points, compliance notes, and tags per source |
| `task_a_source_inventory.csv` | Flat CSV version of the same inventory — for spreadsheet filtering and review |
| `task_a_topic_map.md` | Knowledge base topic map — domain structure, concept index, cross-domain relationships |
| `task_a_gap_analysis.md` | Gap analysis — what's missing, priority level, and recommended next search actions |

---

## Domain Coverage Summary

| Domain | Sources | Coverage |
|--------|---------|----------|
| Thai Regulatory Compliance | 6 | ✅ Core covered |
| Global Regulatory Compliance | 5 | ✅ Good |
| Credit Scoring Fundamentals | 4 | ✅ Good |
| Scorecard Methodology (WoE/LR) | 3 | ✅ Good |
| Machine Learning Models | 4 | ✅ Good |
| Alternative Data | 5 | ✅ Good |
| Explainability (XAI / SHAP) | 4 | ✅ Good |
| Fairness and Bias | 4 | ✅ Good |
| SME Lending / Financial Inclusion | 7 | ⚠️ Partial (Thai empirical missing) |
| Model Governance and Monitoring | 3 | ✅ Good |
| RAG System Design | 1 | ⚠️ Minimal |

---

## Top 5 Critical Gaps to Address Before Task B

1. **BOT Model Risk Management / AI Governance Circular** — Thai-specific binding guidance
2. **BOT Home Loan / Mortgage Underwriting Standards** — LTV, appraisal, income doc requirements
3. **NCB Thailand Technical Score Documentation** — Actual factor weights, not FICO analogy
4. **Reject Inference Methodology** — Fundamental credit modeling bias correction
5. **MAS FEAT Principles (Singapore)** — ASEAN regional AI governance benchmark

---

## Next Steps: Task B

Task B will extract RAG-ready chunks from the highest-priority sources:
1. Download / fetch full text from top sources
2. Split into semantic chunks (512–1024 tokens, 128 token overlap)
3. Write chunks with metadata to `task_b_chunks.json`
4. Embed and prepare for vector store ingestion

**Chunking strategy:** Semantic sections (not fixed-size), preserve section headers as metadata, include source ID + title + domain in chunk metadata for filtering.
