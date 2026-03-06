# AI Credit Scoring

ระบบประเมินสินเชื่อบ้านอัตโนมัติ พร้อม RAG-backed Thai-language planning

## Architecture Overview

```
POST /api/v1/score/request
         │
         ▼
FeatureMergerService       ← ดึง credit_grade, outstanding, overdue ฯลฯ
         │
         ▼
ModelRunnerService         ← คำนวณ risk probability + SHAP values
         │                   (credit_score, credit_grade, outstanding,
         │                    overdue, loan_amount, loan_term, Salary ...)
         ▼
Planner + RAG              ← generate_response() เลือก mode อัตโนมัติ
         │
         ├─ approved_guidance  → เช็กลิสต์ + RAG sources (อนุมัติ)
         └─ improvement_plan  → แผน 3 ระยะ ภาษาไทย + RAG evidence (ปฏิเสธ)
         │
         ▼
ScoringResponse            ← JSON พร้อม advice.result_th
```

## Stack

| Layer | Technology |
|---|---|
| API | FastAPI + Pydantic |
| Vector Store | ChromaDB |
| Embeddings | BGE-M3 (HuggingFace) |
| LLM (default) | Ollama — qwen3:8b (local) |
| LLM (optional) | OpenAI GPT / Gemini 2 Flash |
| RAG Framework | LlamaIndex |
| Database | SQLite (SQLAlchemy) |

---

## Setup (Windows PowerShell)

### 1. Create and activate virtual environment

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
```

If script execution is blocked:

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

### 2. Install uv

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

### 3. Configure environment

```powershell
Copy-Item .env.example .env
```

Edit `.env`:

```env
# LLM — Ollama (default, local, no API key needed)
USE_OLLAMA=true
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=qwen3:8b

# Optional: switch to Gemini (see LLM Providers below)
USE_GEMINI=false
GEMINI_API_KEY=

# Optional
HF_TOKEN=hf_xxx
```

### 4. Install dependencies

```powershell
uv sync --extra dev
```

### 5. Build vector index

```powershell
uv run python -m src.ingest
```

### 6. Start API server

```powershell
uv run uvicorn src.api.main:app --reload
```

---

## LLM Providers

ระบบรองรับ 3 provider เปลี่ยนได้ผ่าน `.env` ไม่ต้องแก้โค้ด:

| Provider | ENV | Package |
|---|---|---|
| **Ollama** (default) | `USE_OLLAMA=true` | built-in |
| **Gemini 2/2.5 Flash** | `USE_GEMINI=true` + `GEMINI_API_KEY=...` | `pip install llama-index-llms-gemini` |
| **OpenAI** | `USE_OLLAMA=false` + `OPENAI_API_KEY=...` | built-in |

ลำดับความสำคัญ: `USE_GEMINI > USE_OLLAMA > OpenAI`

---

## API Usage

### Health check

```bash
GET /health
```

### Score request

```bash
POST /api/v1/score/request
Content-Type: application/json

{
  "request_id": "req-001",
  "customer_id": "cust-123",
  "demographics": {
    "age": 35,
    "employment_status": "Salaried_Employee",
    "education_level": "Bachelor",
    "marital_status": "Single"
  },
  "financials": {
    "monthly_income": 55000,
    "monthly_expenses": 20000,
    "existing_debt": 70000
  },
  "loan_details": {
    "loan_amount": 1100000,
    "loan_term_months": 324,
    "loan_purpose": "Mortgage"
  }
}
```

### Response structure

```json
{
  "request_id": "req-001",
  "approved": false,
  "probability_score": 0.52,
  "explanations": {
    "is_thin_file": false,
    "tree_shap_values": { "credit_score": -0.05, "loan_term": -0.05, ... }
  },
  "advice": {
    "mode": "improvement_plan",
    "result_th": "สรุปสั้น\n- สถานะปัจจุบัน: ยังมีความเสี่ยงไม่อนุมัติ...",
    "rag_sources": [{ "source_title": "...", "category": "hardship_support" }]
  }
}
```

---

## Commands

```powershell
# Ollama diagnostics
uv run python scripts\doctor_ollama.py

# Rebuild vector index
uv run python -m src.ingest

# Run all unit tests
uv run pytest tests -q

# Run planner quality tests (3 test cases × 4 layers)
uv run pytest tests/test_planner_quality.py -v

# Run RAG eval (requires Ollama running)
uv run pytest tests/test_rag_eval.py -v

# Demo: planner output for all 3 test cases (no Ollama needed)
python examples/planner/test_cases_demo.py

# Generate similarity report from retrieval logs
uv run python -m src.rag.report
```

---

## Test Cases

| Case | Input | Expected | Result |
|---|---|---|---|
| Low Risk | credit_score=700, grade=AA, no debt | approved + checklist | ✓ |
| High Risk | credit_score=652, grade=FF, 601k debt | rejected + credit plan | ✓ |
| Medium Risk | credit_score=700, grade=CC, 70k debt | borderline + loan plan | ✓ |

---

## Notes

- Ollama Desktop running locally is sufficient — CLI in `PATH` is not required.
- `probability_score` = risk/default probability (higher = riskier, threshold 0.50).
- Planner runs without RAG if Chroma is unavailable (graceful degradation).
- `FeatureMergerService` and `ModelRunnerService` are currently mocked — replace with real DB queries and ML model for production.
