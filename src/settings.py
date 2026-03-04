import os
from dotenv import dotenv_values

# Read .env without mutating process environment so tests can monkeypatch
# os.environ deterministically.
_DOTENV = dotenv_values()


def _get_env(key: str, default: str) -> str:
    value = os.environ.get(key)
    if value is not None:
        return value
    file_value = _DOTENV.get(key)
    if file_value is not None:
        return str(file_value)
    return default

# NOTE: config/settings.py is the canonical source for ChromaDB settings (CHROMA_PERSIST_DIR, CHROMA_COLLECTION).
# All modules should import from config/settings.py to ensure consistent ChromaDB configuration.
# The settings below are kept for backward compatibility with legacy code only.

OLLAMA_BASE_URL = _get_env("OLLAMA_BASE_URL", "http://localhost:11434")
# Preferred variable is OLLAMA_MODEL; LLM_MODEL remains as backward-compatible fallback.
OLLAMA_MODEL = (
    os.environ.get("OLLAMA_MODEL")
    or os.environ.get("LLM_MODEL")
    or _DOTENV.get("OLLAMA_MODEL")
    or _DOTENV.get("LLM_MODEL")
    or "qwen3:8b"
)
LLM_MODEL = OLLAMA_MODEL
# BGE-M3: multilingual (incl. Thai), 1024 dim, 8192 token context
EMBED_MODEL = _get_env("EMBED_MODEL", "BAAI/bge-m3")

# Directory containing Thai PDFs and other documents
DATA_DIR = _get_env("DATA_DIR", "./data/documents")
INDEX_DIR = _get_env("INDEX_DIR", "./storage/index")
VECTOR_STORE_TYPE = _get_env("VECTOR_STORE_TYPE", "chroma").lower()

# DEPRECATED: Use config/settings.py for ChromaDB settings instead
# These are kept for backward compatibility only
CHROMA_PERSIST_DIR = _get_env("CHROMA_PERSIST_DIR", "./storage/chroma")
CHROMA_COLLECTION = _get_env("CHROMA_COLLECTION", "credit_policies")
