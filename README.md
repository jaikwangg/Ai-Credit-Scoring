# AI Credit Scoring

## Windows PowerShell Setup (.venv), uv, and Ollama

This project is configured to use environment variables for Ollama:

- `OLLAMA_BASE_URL` (default: `http://localhost:11434`)
- `OLLAMA_MODEL` (default: `qwen3:8b`)

### 1) Create and activate a virtual environment

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
```

If script execution is blocked in PowerShell, run once:

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

### 2) Install uv

Option A: install globally (recommended)

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

Option B: install inside `.venv`

```powershell
python -m pip install uv
```

### 3) Configure environment

```powershell
Copy-Item .env.example .env
```

Edit `.env` if needed:

- `OLLAMA_BASE_URL=http://localhost:11434`
- `OLLAMA_MODEL=qwen3:8b`
- `HF_TOKEN=hf_xxx` (optional, to remove HF unauthenticated warning)

### 4) Sync dependencies with uv

```powershell
uv sync --extra dev
```

If `uv` is not in `PATH`, use:

```powershell
& .\.venv\Scripts\uv.exe sync --extra dev
```

## UV command cheatsheet

### Ollama diagnostics

```powershell
uv run python scripts\doctor_ollama.py
```

### Build/rebuild vector index (ingest)

```powershell
uv run python -m src.ingest
```

### Run full RAG smoke test (includes interactive mode)

```powershell
uv run python test_cimb_loans.py
```

Auto-exit interactive mode:

```powershell
'quit' | uv run python test_cimb_loans.py
```

### Unit tests (all)

```powershell
uv run pytest tests -q
```

### Unit tests via project script

```powershell
powershell -ExecutionPolicy Bypass -File scripts/run_rag_tests.ps1 -Mode unit -AllUnitTests
```

### Run a specific test file

```powershell
uv run pytest tests/test_scoring_api.py -q
```

### Generate similarity report from retrieval logs

```powershell
uv run python -m src.rag.report
```

## Notes

- The project does **not** require the `ollama` CLI in `PATH`.
- Ollama Desktop running locally is sufficient as long as `http://localhost:11434` is reachable.
