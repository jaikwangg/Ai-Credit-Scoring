# Topic Map: RAG-based AI Credit Scoring Assistant
## Knowledge Base Structure — Task A Output

**Project:** RAG-based AI Credit Scoring Assistant
**Geography Focus:** Thailand (primary) + Global (context)
**Compiled:** 2026-03-17
**Total Sources Mapped:** 38

---

## Domain Overview

The knowledge base covers 9 core domains arranged from foundational methodology → technical implementation → regulatory compliance → Thai market specifics.

```
┌─────────────────────────────────────────────────────────────────┐
│                    CREDIT SCORING KNOWLEDGE BASE                 │
├──────────────────┬──────────────────┬───────────────────────────┤
│  FOUNDATIONS     │  METHODS         │  GOVERNANCE               │
│                  │                  │                           │
│ Credit Score     │ Scorecard (WoE/  │ Regulatory Compliance     │
│ Fundamentals     │ LR)              │ (TH + Global)             │
│                  │                  │                           │
│ Basel IRB /      │ ML Models        │ Fairness & Bias           │
│ PD-LGD-EAD       │ (XGBoost etc.)   │                           │
│                  │                  │                           │
│ Financial        │ Alternative Data │ Model Governance          │
│ Inclusion        │                  │ & Monitoring              │
│                  │ Explainability   │                           │
│                  │ (XAI/SHAP)       │ RAG System Design         │
└──────────────────┴──────────────────┴───────────────────────────┘
```

---

## Domain 1: Credit Scoring Fundamentals

**Purpose in RAG:** Core definitional knowledge — what credit scoring is, how traditional scores work, what factors matter.

| Source ID | Title | Trust | Language |
|-----------|-------|-------|----------|
| GL-FICO-001 | FICO Score 5-Factor Model | Authoritative | EN |
| TH-NCB-001 | Thailand NCB Credit Score Overview | Authoritative | TH+EN |
| GL-BASEL-001 | Basel III IRB: PD/LGD/EAD Framework | Authoritative | EN |
| GL-PDMD-001 | Probability of Default Model Guide (BoE PRA) | High | EN |

**Key Concepts Covered:**
- FICO 5 factors and weights (Payment History 35%, Amounts Owed 30%, History 15%, Mix 10%, New Credit 10%)
- NCB Thailand score range (300–900) and factors
- PD, LGD, EAD definitions and relationships
- Through-the-cycle (TTC) vs point-in-time (PIT) PD
- Expected Loss = PD × LGD × EAD

---

## Domain 2: Scorecard Methodology (WoE / Logistic Regression)

**Purpose in RAG:** Detailed technical methodology for building traditional interpretable scorecards — the regulatory gold standard.

| Source ID | Title | Trust | Language |
|-----------|-------|-------|----------|
| GL-SCORECARD-001 | WoE and IV — listendata.com | Medium | EN |
| GL-SCORECARD-002 | Scorecard Development — Altair | Medium-High | EN |
| GL-WOE-PAPER-001 | Logistic Regression Scorecard — Tandfonline | High | EN |

**Key Concepts Covered:**
- WoE = ln(Events% / Non-Events%)
- IV thresholds: <0.02 useless → >0.3 strong predictor
- Fine vs coarse classing (binning optimization)
- Scorecard points = offset + factor × ln(odds)
- OOT validation as preferred credit model validation approach
- CSI (Characteristic Stability Index) per variable, PSI overall

---

## Domain 3: Machine Learning Models for Credit Scoring

**Purpose in RAG:** Modern ML approaches for credit scoring — performance benchmarks, implementation guidance, trade-offs.

| Source ID | Title | Trust | Language |
|-----------|-------|-------|----------|
| GL-ML-001 | ML Comparative Study — PLOS ONE | High | EN |
| GL-ML-002 | Deep Learning Review — Nature Sci. Reports 2024 | High | EN |
| GL-MLPAPER-001 | ML for Credit Default — arXiv 2023 | Medium-High | EN |
| GL-IMBALANCE-001 | SMOTE for Class Imbalance | High | EN |

**Key Concepts Covered:**
- XGBoost, LightGBM, Random Forest, CatBoost comparative performance
- AUC-ROC vs AUC-PR (PR curve preferred for imbalanced credit data)
- SMOTE + undersampling for class imbalance (typical default rate: 1–10%)
- Model calibration (Platt scaling, isotonic regression) for PD output
- LSTM/Transformer for sequential repayment data
- Graph Neural Networks for relationship-based credit (SME)
- Model stacking / ensemble approaches

---

## Domain 4: Alternative Data

**Purpose in RAG:** Non-traditional data sources to score thin-file and unbanked borrowers — the core innovation opportunity.

| Source ID | Title | Trust | Language |
|-----------|-------|-------|----------|
| GL-ALTDATA-001 | World Bank: Alternative Data for Credit Risk | High | EN |
| GL-ALTDATA-002 | Big Data/ML for Financial Inclusion — SSRN | Medium-High | EN |
| GL-ALTDATA-003 | AFI: Digital Financial Services & Alt Scoring | High | EN |
| TH-SME-001 | BOT SME Credit Risk Database + NaCGA | Authoritative | TH+EN |
| TH-ADB-001 | ADB: Access to Finance for Thai SMEs | High | EN |

**Alternative Data Taxonomy (by predictive power and risk):**

```
HIGH PREDICTIVE POWER, LOWER BIAS RISK:
├── Utility payment history (electricity, water, telecoms bills)
├── Telecom airtime/data usage patterns
├── Mobile money / e-wallet transaction history (PromptPay, TrueMoney)
├── E-commerce purchase and repayment history (Shopee, Lazada)
└── Tax filing and revenue records (SME)

MODERATE PREDICTIVE POWER, HIGHER BIAS RISK:
├── Rental payment history
├── Social media behavioral signals
└── Psychometric / willingness-to-pay scores

THAILAND-SPECIFIC OPPORTUNITIES:
├── PromptPay transaction frequency
├── Social Security contribution records
├── Revenue Department e-Tax records
└── NaCGA guarantee usage history
```

---

## Domain 5: Explainability (XAI)

**Purpose in RAG:** Methods for explaining ML model predictions to regulators, applicants, and auditors.

| Source ID | Title | Trust | Language |
|-----------|-------|-------|----------|
| GL-XAI-001 | SHAP — Original Paper + Documentation | Authoritative | EN |
| GL-XAI-002 | LIME — Original Paper | High | EN |
| GL-XAI-003 | XAI in Credit Scoring — CFA Institute | High | EN |
| GL-XAI-REVIEW-001 | Hybrid SHAP-Scorecard — Springer | High | EN |

**XAI Decision Framework:**

```
INTRINSICALLY INTERPRETABLE (no XAI needed):
└── Logistic Regression Scorecard → direct coefficient interpretation

POST-HOC GLOBAL XAI (understand model overall):
├── SHAP summary plots (feature importance across population)
└── Permutation importance

POST-HOC LOCAL XAI (explain individual decisions):
├── SHAP force plots / waterfall plots (RECOMMENDED — use for adverse action)
├── LIME local surrogate (secondary validation)
└── Counterfactual explanations ("what would change the decision?")

ADVERSE ACTION REASON CODES:
└── Top 3–5 SHAP values mapped to human-readable statements
    Example: "Low payment history score" / "High credit utilization" / "Short credit history"
```

---

## Domain 6: Fairness and Bias

**Purpose in RAG:** Bias detection, fairness measurement, and mitigation strategies for credit models.

| Source ID | Title | Trust | Language |
|-----------|-------|-------|----------|
| GL-FAIR-001 | CFPB: AI/ML Adverse Action Notice Guidance | Authoritative | EN |
| GL-FAIR-002 | CFPB Fair Lending Report 2023 | Authoritative | EN |
| GL-FAIR-003 | EU AI Act Annex III — High-Risk AI | Authoritative | EN |
| GL-FAIRML-001 | Fairness in ML — JMLR Survey | High | EN |

**Fairness Metrics Quick Reference:**

| Metric | Definition | When to Use |
|--------|-----------|-------------|
| Demographic Parity | Equal approval rates across groups | Policy requirement |
| Equalized Odds | Equal TPR and FPR across groups | Technical fairness |
| Equal Opportunity | Equal TPR across groups | Lending fairness standard |
| Individual Fairness | Similar individuals treated similarly | Specific case review |
| Disparate Impact | 80% rule: minority approval ≥ 80% of majority | US ECOA standard |

**Bias Mitigation Stages:**
1. Pre-processing: Resampling, re-weighting, feature removal
2. In-processing: Fairness-constrained training (adversarial debiasing)
3. Post-processing: Threshold adjustment per group, reject inference

---

## Domain 7: Regulatory Compliance

### 7A: Thailand Regulatory Framework

| Source ID | Title | Type | Effective |
|-----------|-------|------|-----------|
| TH-BOT-001 | Responsible Lending (SorKorChor. 7/2566) | Regulation | 2023, updated 2025 |
| TH-BOT-002 | DSR Macroprudential Policy | Policy | 2023 |
| TH-BOT-003 | Risk-Based Pricing Sandbox | Sandbox Framework | 2022 |
| TH-PDPA-001 | Personal Data Protection Act (PDPA) | Legislation | 2022 |
| TH-TFRS9-001 | TFRS 9 Expected Credit Loss | Accounting Standard | 2020 |
| TH-FINTECH-001 | BOT FinTech Sandbox (Digital Lending) | Regulatory Framework | 2022 |

**Thai Regulatory Compliance Checklist for AI Credit Scoring:**
- [ ] Lawful basis for PDPA data processing documented
- [ ] DSR calculation embedded as hard constraint (not just feature)
- [ ] Ability-to-repay assessment process documented (SorKorChor. 7/2566)
- [ ] Adverse action reason codes provided to rejected applicants
- [ ] Model documentation aligned with BOT model risk guidelines
- [ ] TFRS 9 PIT PD outputs calibrated for ECL provisioning
- [ ] FinTech sandbox application (if applicable)
- [ ] Risk-based pricing approval (if pricing differentiation used)

### 7B: Global Regulatory Framework

| Source ID | Title | Jurisdiction | Key Deadline |
|-----------|-------|-------------|--------------|
| GL-FAIR-001 | CFPB Adverse Action Guidance | USA | Active |
| GL-FAIR-002 | CFPB Fair Lending Report 2023 | USA | Active |
| GL-FAIR-003 | EU AI Act — Annex III | EU | Aug 2, 2026 |
| GL-BASEL-001 | Basel III IRB | Global | Active |
| GL-VALIDATION-001 | SR 11-7 Model Risk Management | USA (de facto global) | Active |

---

## Domain 8: Model Governance and Monitoring

**Purpose in RAG:** End-to-end model lifecycle management — development, validation, deployment, monitoring, retirement.

| Source ID | Title | Trust | Language |
|-----------|-------|-------|----------|
| GL-VALIDATION-001 | SR 11-7 Model Risk Management | Authoritative | EN |
| GL-MONITORING-001 | PSI and Model Monitoring | Medium | EN |
| GL-SCORECARD-002 | Scorecard Validation (Altair) | Medium-High | EN |

**Model Lifecycle Stages:**
```
1. DEVELOPMENT → 2. VALIDATION → 3. APPROVAL → 4. DEPLOYMENT → 5. MONITORING → 6. REVIEW/REBUILD

Key Metrics per Stage:
Development:  AUC, Gini, KS, F1, Precision/Recall, Calibration (Brier score)
Validation:   OOT AUC, OOT Gini, Benchmark comparison, Conceptual soundness review
Monitoring:   PSI (monthly), CSI (quarterly), Gini stability, Default rate by score band
Review:       Annual full revalidation; rebuild trigger if PSI > 0.2 or Gini drift > 5%
```

---

## Domain 9: SME Lending and Financial Inclusion

**Purpose in RAG:** Thailand and ASEAN context for SME credit scoring — the high-priority growth segment.

| Source ID | Title | Trust | Geography |
|-----------|-------|-------|-----------|
| TH-SME-001 | BOT SME Credit Risk Database + NaCGA | Authoritative | Thailand |
| TH-ADB-001 | ADB: SME Finance in Thailand | High | Thailand |
| TH-ASEAN-001 | ASEAN SME Financing Benchmark | High | ASEAN |
| TH-NCB-001 | NCB Thailand (thin-file context) | Authoritative | Thailand |
| GL-INCLUSION-001 | World Bank Global Findex 2021 | Authoritative | Global |
| GL-INCLUSION-002 | IFC Credit Infrastructure | High | Global |
| GL-ALTDATA-003 | AFI Digital Financial Services | High | SEA |

**Thailand SME Credit Profile:**
- SME credit penetration: ~40% of GDP (moderate vs ASEAN peers)
- Key barrier: lack of formal financial records and collateral
- Opportunity: Revenue Department records, social security, supply chain data
- NaCGA: partial guarantee scheme reduces lender risk
- Digital lending (FinTech): growing but not yet systemically significant

---

## Domain 10: RAG System Design

**Purpose in RAG:** Technical architecture knowledge for building the RAG system itself.

| Source ID | Title | Trust | Language |
|-----------|-------|-------|----------|
| GL-RAGTECH-001 | RAG Original Paper (NeurIPS 2020) | Authoritative | EN |

**Recommended RAG Architecture for Credit Scoring Knowledge Base:**
```
INGESTION PIPELINE:
Documents → Chunking (512–1024 tokens, overlap 128) → Embedding (text-embedding-3-large)
→ Vector Store (pgvector / Pinecone / Chroma)

RETRIEVAL PIPELINE:
Query → Hybrid Search (BM25 sparse + dense embedding)
→ Re-ranking (cross-encoder) → Top-K chunks (K=5–10)
→ Context assembly → LLM generation

METADATA FILTERING (by domain_topic, geography_relevance, trust_level):
→ Pre-filter: country=Thailand for regulatory queries
→ Pre-filter: trust_level=authoritative for compliance queries
→ Pre-filter: document_type for source type queries
```

---

## Cross-Domain Concept Index

| Concept | Primary Domains | Key Sources |
|---------|----------------|-------------|
| Adverse Action Reason Codes | XAI, Fairness, Regulatory | GL-FAIR-001, GL-XAI-001, GL-XAI-003 |
| DSR / Debt Service Ratio | Regulatory (TH), Fundamentals | TH-BOT-002, TH-BOT-001 |
| Thin File / No Credit History | Alternative Data, Financial Inclusion | GL-ALTDATA-001, TH-NCB-001, GL-INCLUSION-001 |
| PD (Probability of Default) | Fundamentals, ML, Regulatory | GL-BASEL-001, GL-PDMD-001, TH-TFRS9-001 |
| Model Monitoring / Drift | Governance, Scorecard | GL-MONITORING-001, GL-VALIDATION-001 |
| Class Imbalance | ML Models | GL-IMBALANCE-001, GL-ML-001, GL-MLPAPER-001 |
| SHAP / Feature Attribution | XAI, Fairness, Regulatory | GL-XAI-001, GL-XAI-REVIEW-001, GL-FAIR-001 |
| FinTech Alternative Lending | Alternative Data, Thai Market | GL-ALTDATA-002, TH-FINTECH-001, TH-ASEAN-001 |
| EU AI Act | Regulatory, Fairness | GL-FAIR-003 |
| PDPA Compliance | Regulatory (TH) | TH-PDPA-001 |
| TFRS 9 / ECL | Regulatory (TH), PD Models | TH-TFRS9-001, GL-PDMD-001 |
| WoE / IV Scorecard | Scorecard Methodology | GL-SCORECARD-001, GL-WOE-PAPER-001 |
| SME Credit Scoring Thailand | SME Lending | TH-SME-001, TH-ADB-001, TH-ASEAN-001 |
| Basel IRB | Fundamentals, Regulatory | GL-BASEL-001, GL-PDMD-001 |
| Less Discriminatory Alternative (LDA) | Fairness | GL-FAIR-002, GL-FAIRML-001 |

---

## Source Count by Domain

| Domain | # Sources | TH Primary | Global |
|--------|-----------|-----------|--------|
| Regulatory Compliance (TH) | 6 | 6 | 0 |
| Regulatory Compliance (Global) | 5 | 0 | 5 |
| Credit Scoring Fundamentals | 4 | 1 | 3 |
| Scorecard Methodology | 3 | 0 | 3 |
| Machine Learning Models | 4 | 0 | 4 |
| Alternative Data | 5 | 2 | 3 |
| Explainability (XAI) | 4 | 0 | 4 |
| Fairness & Bias | 4 | 0 | 4 |
| SME Lending / Financial Inclusion | 7 | 3 | 4 |
| Model Governance & Monitoring | 3 | 0 | 3 |
| RAG System Design | 1 | 0 | 1 |
| **TOTAL** | **46*** | **12** | **34** |

*Note: Some sources counted in multiple domains; unique sources = 38.
