# AI Credit Scoring

## Windows PowerShell Setup (.venv) and Ollama

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

### 2) Install dependencies

```powershell
pip install -r requirements.txt
```

### 3) Configure environment

```powershell
Copy-Item .env.example .env
```

Edit `.env` if needed:

- `OLLAMA_BASE_URL=http://localhost:11434`
- `OLLAMA_MODEL=qwen3:8b`

### 4) Run Ollama diagnostics

```powershell
python scripts\doctor_ollama.py
```

The doctor script checks:

- Active Python + venv
- `import ollama`
- `GET /api/tags` connectivity and available models
- Minimal text generation via Ollama HTTP API
- Optional `ollama` Python client chat call

### Notes

- The project does **not** require the `ollama` CLI in `PATH`.
- Ollama Desktop running locally is sufficient as long as `http://localhost:11434` is reachable.
