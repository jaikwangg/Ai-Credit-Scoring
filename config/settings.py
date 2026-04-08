"""
Configuration settings for LlamaIndex project
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class Settings:
    """Application settings"""
    
    # API Keys
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")

    # Ollama Settings (local, default)
    OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    OLLAMA_MODEL: str = os.getenv("OLLAMA_MODEL", "qwen3:8b")
    USE_OLLAMA: bool = os.getenv("USE_OLLAMA", "true").lower() == "true"
    # Cap Ollama context window so RAM usage is bounded on small machines.
    # Phi/Gemma default num_ctx can be 16k-128k → blows up RAM. Use 4096 to be safe.
    OLLAMA_NUM_CTX: int = int(os.getenv("OLLAMA_NUM_CTX", "4096"))
    OLLAMA_NUM_PREDICT: int = int(os.getenv("OLLAMA_NUM_PREDICT", "512"))

    # OpenAI Settings
    MODEL_NAME: str = os.getenv("MODEL_NAME", "gpt-3.5-turbo")

    # ---------------------------------------------------------------------------
    # Gemini Settings (future migration — not active until USE_GEMINI=true)
    # To enable: set USE_GEMINI=true, GEMINI_API_KEY=<your-key> in .env
    # Recommended model: gemini-2.0-flash  or  gemini-2.5-flash-preview-04-17
    # Required package:  pip install llama-index-llms-gemini
    # ---------------------------------------------------------------------------
    GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")
    GEMINI_MODEL: str = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")
    USE_GEMINI: bool = os.getenv("USE_GEMINI", "false").lower() == "true"
    # BGE-M3 for Thai/multilingual embeddings (1024 dim)
    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", os.getenv("EMBED_MODEL", "BAAI/bge-m3"))
    
    # Directory Paths
    PROJECT_ROOT: Path = Path(__file__).parent.parent

    # DOCUMENTS_DIR can be overridden via DATA_DIR env var (path is resolved
    # relative to PROJECT_ROOT when given as a relative path). Historically
    # this was hardcoded to data/documents/ which caused silent drift from
    # the .env file. Now it respects DATA_DIR env.
    @staticmethod
    def _resolve_documents_dir() -> Path:
        project_root = Path(__file__).parent.parent
        raw = os.getenv("DATA_DIR", "./data/documents").strip()
        p = Path(raw)
        if not p.is_absolute():
            p = (project_root / p).resolve()
        return p

    DATA_DIR: Path = PROJECT_ROOT / "data"
    DOCUMENTS_DIR: Path = _resolve_documents_dir()
    INDEX_DIR: Path = PROJECT_ROOT / "data" / "index"
    
    # Vector Store Settings
    VECTOR_STORE_TYPE: str = os.getenv("VECTOR_STORE_TYPE", "chroma")  # chroma, faiss, simple
    # ChromaDB settings - single source of truth for all modules
    CHROMA_PERSIST_DIR: str = os.getenv("CHROMA_PERSIST_DIR", "./storage/chroma")
    CHROMA_COLLECTION: str = os.getenv("CHROMA_COLLECTION", "credit_policies")
    
    # Index Settings
    CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "512"))  # Smaller chunks for better granularity
    CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "50"))  # More overlap
    
    # Query Settings
    SIMILARITY_TOP_K: int = int(os.getenv("SIMILARITY_TOP_K", "4"))
    SIMILARITY_CUTOFF: float = float(os.getenv("SIMILARITY_CUTOFF", "0.45"))
    RESPONSE_MODE: str = os.getenv("RESPONSE_MODE", "compact")

    # Database
    DATABASE_URL: str = os.getenv("DATABASE_URL", "sqlite:///./sql_app.db")

    # Ingestion safety
    # True => rebuild Chroma collection on ingest to avoid stale/mixed nodes
    RESET_CHROMA_COLLECTION_ON_INGEST: bool = (
        os.getenv("RESET_CHROMA_COLLECTION_ON_INGEST", "true").lower() == "true"
    )
    
    # Logging
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    
    @classmethod
    def validate(cls) -> bool:
        """Validate required settings"""
        if cls.USE_GEMINI and not cls.GEMINI_API_KEY:
            raise ValueError("GEMINI_API_KEY is required when USE_GEMINI=true.")
        if not cls.USE_OLLAMA and not cls.USE_GEMINI and not cls.OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY is required when not using Ollama or Gemini.")
        return True
    
    @classmethod
    def ensure_directories(cls):
        """Create necessary directories"""
        cls.DOCUMENTS_DIR.mkdir(parents=True, exist_ok=True)
        cls.INDEX_DIR.mkdir(parents=True, exist_ok=True)

# Global settings instance
settings = Settings()
