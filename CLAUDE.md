# AI Credit Scoring — Personal Assistant

## Project Overview

A RAG-based AI personal assistant for credit scoring decisions, built on LlamaIndex + OpenAI. The system indexes internal credit policy documents, scoring model documentation, and rules, then answers queries with structured decisions (approve / decline / need_more_info / review) backed by evidence.

## Architecture

```
src/
  schema.py          # Pydantic response models (AssistantResponse, Reason, Evidence)
  indexer.py         # IndexManager — create, load, rebuild vector index
  query_engine.py    # QueryEngineManager — query + chat engine wrappers
  utils.py           # Logging, env validation, helpers
config/
  settings.py        # Central config (chunk size, top-k, model names, etc.)
examples/
  basic_query.py     # Single-turn query examples
  advanced_query.py  # Multi-config + performance analysis examples
  chat_engine.py     # Multi-turn chat session examples
data/
  documents/         # Source documents (PDF, DOCX, TXT, XLSX, CSV)
  index/             # Persisted vector index
logs/                # Runtime logs (llama_index.log)
```

## Key Models (`src/schema.py`)

- **`AssistantResponse`** — top-level structured output
  - `decision`: `"approve" | "decline" | "need_more_info" | "review"`
  - `summary`: short human-readable summary
  - `reasons`: list of `Reason` (type: rule/model/policy, text, evidence list)
  - `missing_info`: list of data gaps
  - `next_actions`: recommended follow-up steps
  - `customer_message_draft`: optional pre-written message to applicant
  - `risk_note`: optional risk annotation

## Environment Setup

```bash
# Windows (Git Bash / WSL)
source venv/Scripts/activate     # or venv/bin/activate on Linux/Mac
pip install -r Requirements.txt

# Required .env variables
OPENAI_API_KEY=sk-...
MODEL_NAME=gpt-3.5-turbo          # or gpt-4
EMBEDDING_MODEL=text-embedding-ada-002
VECTOR_STORE_TYPE=chroma           # chroma | faiss | simple
CHUNK_SIZE=1024
CHUNK_OVERLAP=20
SIMILARITY_TOP_K=4
RESPONSE_MODE=compact              # compact | refine | tree_summarize
LOG_LEVEL=INFO
```

## Common Commands

```bash
# Index credit documents
python index_documents.py

# Quick single query
python quick_query.py "Should we approve applicant with score 680?"

# Run examples
python examples/basic_query.py
python examples/advanced_query.py
python examples/chat_engine.py
```

## Development Notes

- All structured LLM responses should conform to `AssistantResponse` from `src/schema.py`
- Evidence must cite `doc_title`, `section`, and `page` where available
- The `decision` field drives downstream workflow routing — treat it as authoritative
- Vector store defaults to Chroma (persisted); use `simple` for fast local testing without persistence
- `data/index/` can be deleted and rebuilt with `python index_documents.py` at any time
- Credit policy documents belong in `data/documents/` — supported: PDF, TXT, DOCX, XLSX, CSV
