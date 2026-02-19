import os
from dotenv import load_dotenv

load_dotenv()

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
# Preferred variable is OLLAMA_MODEL; LLM_MODEL remains as backward-compatible fallback.
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", os.getenv("LLM_MODEL", "qwen3:8b"))
LLM_MODEL = OLLAMA_MODEL
EMBED_MODEL = os.getenv("EMBED_MODEL", "BAAI/bge-small-en-v1.5")

DATA_DIR = os.getenv("DATA_DIR", "./data/policies")
INDEX_DIR = os.getenv("INDEX_DIR", "./storage/index")
VECTOR_STORE_TYPE = os.getenv("VECTOR_STORE_TYPE", "chroma").lower()
CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "./storage/chroma")
CHROMA_COLLECTION = os.getenv("CHROMA_COLLECTION", "credit_policies")
