# Gap Analysis: AI Credit Scoring Knowledge Base
## Task A — Source Discovery Gaps and Recommendations

**Project:** RAG-based AI Credit Scoring Assistant
**Compiled:** 2026-03-17
**Sources Reviewed:** 38
**Gap Priority Levels:** 🔴 Critical | 🟡 Important | 🟢 Nice-to-Have

---

## Summary

The current source inventory covers core methodology, global regulatory standards, and Thailand's primary regulatory framework well. However, several critical Thai-market-specific sources, operational deployment guides, and empirical benchmarks remain unmapped. These gaps — if unaddressed — will reduce the RAG system's ability to answer deployment-specific, Thailand-context, and model operations questions accurately.

---

## Critical Gaps 🔴

### GAP-01: BOT Model Risk Management Circular (Thai-specific)
**Missing:** Bank of Thailand's own circular on model risk management for supervised institutions.
**Why critical:** While SR 11-7 (US Federal Reserve) is in the inventory as a global proxy, BOT may have issued its own model risk circular or supervisory expectations document. Thai banks need BOT-specific guidance, not just the US equivalent.
**Recommended action:** Search BOT's "Supervisory Notifications" section for model risk, AI governance, and model validation circulars. Search terms: "BOT AI governance", "แนวปฏิบัติโมเดลความเสี่ยง".
**Source type to find:** BOT Notification or Supervisory Circular (authoritative, Thai primary)

---

### GAP-02: BOT AI/ML in Financial Services Policy or Guideline
**Missing:** BOT-specific AI/ML in financial services guidance. Many central banks have released dedicated AI guidance (MAS Singapore, HKMA, BoE) — BOT equivalent unknown.
**Why critical:** If BOT has issued AI-specific requirements (beyond responsible lending), these would be directly binding on Thai credit scoring deployments and must be in the knowledge base.
**Recommended action:** Search BOT website for "AI", "machine learning", "algorithmic decision" publications. Also search: MAS FEAT principles (Singapore) as a regional comparator.
**Source type to find:** Central bank guidance (authoritative, Thai primary or ASEAN regional)

---

### GAP-03: Actual NCB Thailand Score Factor Weights and Data Dictionary
**Missing:** The actual factor weights used by Thailand's NCB score (not just the FICO analogy). NCB's exact scoring model, factor weights, and data fields used are not documented in the current inventory.
**Why critical:** Building a credit model that aligns with or augments NCB scores requires knowing NCB's exact methodology. Feature engineering for Thai models should mirror NCB factor categories.
**Recommended action:** Obtain NCB technical documentation, member institution data dictionary, or NCB's published credit score guide for consumers. May require NCB member access.
**Source type to find:** NCB technical documentation (authoritative, Thai primary)

---

### GAP-04: Thai Home Loan (Mortgage) Underwriting Standards
**Missing:** Specific BOT or Thai Bankers Association guidance on mortgage underwriting criteria beyond DSR — LTV ratios, appraisal requirements, income documentation standards.
**Why critical:** The project context includes CIMB Thai home loan use case (from corpus cleaning task). Mortgage-specific credit criteria and risk factors are distinct from general consumer credit.
**Recommended action:** Search BOT for LTV regulations, Thai Bankers Association underwriting guidelines.
**Source type to find:** BOT mortgage regulation, industry standard (authoritative, Thai primary)

---

### GAP-05: Thailand SME Credit Scoring Empirical Dataset or Benchmark
**Missing:** Any empirical study or published dataset benchmarking credit scoring model performance on Thai SME data specifically.
**Why critical:** All ML model performance benchmarks in the current inventory use global (typically US/European) datasets. Performance on Thai SME data may differ significantly due to different default patterns, data availability, and economic cycle.
**Recommended action:** Search for Thai academic papers on SME credit scoring, BOT working papers, or NIDA/Chulalongkorn finance department research.
**Source type to find:** Academic paper or working paper (Thai context, medium-high trust)

---

## Important Gaps 🟡

### GAP-06: TFRS 9 Implementation Guidance — Practical Model-Building Guide
**Missing:** Practical guide for Thai banks on building TFRS 9 ECL models. Current inventory has the accounting standard (TH-TFRS9-001) but not a practitioner guide on model architecture, macro-economic overlays, SICR definition, and validation.
**Recommended action:** Search for BOT TFRS 9 implementation notices, FAP guidance notes, Big 4 (Deloitte/PwC/EY/KPMG) TFRS 9 publications for Thai banks.
**Source type to find:** Technical implementation guide (high trust, Thai primary)

---

### GAP-07: Reject Inference Methodology
**Missing:** Sources on reject inference — the statistical problem that credit scoring models are trained only on approved loans, creating selection bias. Critical for any model development from lending book data.
**Why important:** Reject inference is a fundamental challenge in credit modeling. Without addressing it, models systematically underestimate default risk for the applicant population.
**Recommended action:** Search for academic papers on reject inference (Feelders 2000, Hand & Henley 1997), or practitioner guides from SAS, Experian.
**Source type to find:** Academic paper or practitioner guide (global, high trust)

---

### GAP-08: Causal Inference Methods for Credit Scoring
**Missing:** Causal ML approaches (DoWhy, CausalML, Shapley-based counterfactuals) for credit scoring. Important for fairness analysis and counterfactual explanations ("what would change the decision?").
**Why important:** Counterfactual explanations are required by EU AI Act and are best practice for adverse action explanations under CFPB guidance. Pure SHAP is associational, not causal.
**Recommended action:** Search for causal ML in credit scoring papers (arXiv 2023-2024), Judea Pearl causal framework applied to fairness.
**Source type to find:** Research paper (global, medium-high trust)

---

### GAP-09: Open-Source Credit Scoring Datasets and Benchmarks
**Missing:** Standard public datasets used for model benchmarking — German Credit, GIVE ME SOME CREDIT (Kaggle), Home Credit Default Risk, Taiwan Credit, LendingClub.
**Why important:** RAG system users will ask about benchmarking their models. Having dataset documentation and known benchmark AUC ranges enables concrete performance comparison guidance.
**Recommended action:** Document the standard datasets with their characteristics (size, default rate, features, known best model AUC).
**Source type to find:** Dataset documentation / Kaggle competition documentation (medium trust, global)

---

### GAP-10: MAS (Monetary Authority of Singapore) FEAT Principles
**Missing:** Singapore MAS Fairness, Ethics, Accountability, Transparency (FEAT) principles for financial services AI — the most advanced ASEAN AI governance framework.
**Why important:** Thailand may adopt similar principles; MAS FEAT is the regional benchmark. BOT often benchmarks against MAS policies.
**Recommended action:** Fetch https://www.mas.gov.sg/publications/monographs-or-information-papers/2018/feat
**Source type to find:** Central bank guidance (ASEAN regional context, high trust)

---

### GAP-11: Credit Scoring for Agricultural / Rural Segments (Thailand-specific)
**Missing:** Specialized credit scoring approaches for Thailand's agricultural sector (BAAC — Bank for Agriculture and Agricultural Cooperatives). Thailand's rural population remains largely served by BAAC.
**Why important:** A significant Thai credit market segment with unique risk factors (crop yields, weather, commodity prices) not covered by urban consumer credit models.
**Recommended action:** Search for BAAC credit assessment methodology, agricultural credit scoring literature.
**Source type to find:** Institutional publication or academic paper (Thai primary, medium-high trust)

---

### GAP-12: Python / MLflow Model Operations (MLOps) Guide for Credit Scoring
**Missing:** Technical MLOps implementation guide — model versioning, experiment tracking, deployment pipelines, A/B testing, shadow deployment for credit scoring models.
**Why important:** The RAG system will need to answer operational deployment questions. MLflow, BentoML, or similar MLOps tooling documentation would support this.
**Recommended action:** Document key MLOps frameworks: MLflow (model registry), Evidently AI (drift monitoring), Weights & Biases (experiment tracking), Great Expectations (data quality).
**Source type to find:** Technical documentation / library docs (medium trust, global)

---

## Nice-to-Have Gaps 🟢

### GAP-13: Peer-to-Peer (P2P) Lending Regulatory Framework Thailand
**Missing:** SEC / BOT regulatory framework for P2P lending platforms in Thailand. Emerging segment with alternative credit scoring implications.
**Recommended action:** Search SEC Thailand for P2P lending regulations.

---

### GAP-14: Buy Now Pay Later (BNPL) Credit Risk Assessment
**Missing:** BNPL-specific credit risk models. Lazada, Shopee Pay Later, and similar platforms increasingly relevant in Thai consumer credit market.
**Recommended action:** Search for BNPL credit risk academic papers (2022-2025), BOT BNPL regulatory notices.

---

### GAP-15: NLP / Text Mining for Credit Risk (Thai Language)
**Missing:** NLP approaches applied to Thai-language financial text (loan applications, company descriptions, news) for credit risk signals.
**Recommended action:** Search for Thai NLP credit risk papers, WangchanBERTa (Thai BERT) applications in finance.

---

### GAP-16: Privacy-Preserving Machine Learning for Credit Scoring
**Missing:** Federated learning, differential privacy, secure multi-party computation applied to credit scoring. Relevant for PDPA compliance in data-sharing scenarios.
**Recommended action:** Search for federated learning credit scoring papers (2023-2024), PySyft, TensorFlow Federated.

---

### GAP-17: Stress Testing Credit Models (Macro Scenarios)
**Missing:** Methodology for credit model stress testing under adverse macro scenarios (recession, rate shock). Relevant for BOT stress testing requirements.
**Recommended action:** Search for credit model stress testing methodologies, EBA stress testing guidelines.

---

## Coverage Assessment by Use Case

| Use Case | Coverage | Gap Level |
|----------|----------|-----------|
| Thai regulatory compliance (consumer) | ✅ Good | Minor |
| Thai regulatory compliance (SME) | ⚠️ Partial | GAP-01, GAP-04 |
| Traditional scorecard development | ✅ Good | Minor |
| ML model development (global) | ✅ Good | GAP-07, GAP-09 |
| ML model development (Thai data) | ⚠️ Partial | GAP-05, GAP-09 |
| Alternative data sourcing (Thailand) | ⚠️ Partial | GAP-03 |
| XAI / Adverse action explanations | ✅ Good | GAP-08 |
| Fairness and bias testing | ✅ Good | GAP-10 |
| Model governance and monitoring | ✅ Good | GAP-12 |
| SME lending (Thailand) | ⚠️ Partial | GAP-05, GAP-11 |
| Financial inclusion (Thailand) | ✅ Good | GAP-11 |
| TFRS 9 / ECL model building | ⚠️ Partial | GAP-06 |
| RAG system architecture | ✅ Minimal coverage | Needs expansion |

---

## Prioritized Next Search Actions

**Immediate (before Task B):**
1. 🔴 Search for BOT AI governance / model risk circular
2. 🔴 Search for BOT mortgage / home loan underwriting regulations
3. 🔴 Search for NCB Thailand technical score documentation
4. 🟡 Fetch MAS FEAT principles (Singapore)
5. 🟡 Search for reject inference methodology papers

**Before Task C (structured JSON):**
6. 🟡 Search for TFRS 9 practical implementation guide
7. 🟡 Document standard credit scoring benchmark datasets
8. 🟢 Search for BNPL credit risk literature
9. 🟢 Search for Thai agricultural credit scoring (BAAC)

---

## Recommended Source Shortlist: Best Sources per Sub-Topic

| Sub-Topic | Best Source | Reason |
|-----------|-------------|--------|
| Thai regulatory anchor | TH-BOT-001 (SorKorChor. 7/2566) | Directly binding, recently updated |
| Thai data privacy | TH-PDPA-001 | Binding legislation |
| Traditional scorecard | GL-WOE-PAPER-001 + GL-SCORECARD-001 | Academic rigor + practical guide |
| ML model selection | GL-ML-001 (PLOS ONE) | Peer-reviewed comparative benchmark |
| Alternative data | GL-ALTDATA-001 (World Bank) | Authoritative, comprehensive, global-with-SEA examples |
| XAI / explanations | GL-XAI-001 (SHAP docs) + GL-XAI-REVIEW-001 (hybrid) | Canonical source + credit-specific implementation |
| Fairness / bias | GL-FAIRML-001 (JMLR) + GL-FAIR-002 (CFPB) | Theory + regulatory enforcement practice |
| Global AI regulation | GL-FAIR-003 (EU AI Act Annex III) | Gold standard, Aug 2026 deadline |
| Model governance | GL-VALIDATION-001 (SR 11-7) | De facto global standard |
| Thai SME credit | TH-SME-001 (BOT) + TH-ADB-001 | Official database + empirical analysis |
| PD model building | GL-PDMD-001 (BoE PRA) + TH-TFRS9-001 | Technical depth + Thai accounting compliance |
| Model monitoring | GL-MONITORING-001 (PSI guide) | Industry-standard metric, practical thresholds |
