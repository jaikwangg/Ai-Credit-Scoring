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
    
    # Ollama Settings
    OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    OLLAMA_MODEL: str = os.getenv("OLLAMA_MODEL", "qwen3:8b")
    USE_OLLAMA: bool = os.getenv("USE_OLLAMA", "true").lower() == "true"
    
    # Model Settings
    MODEL_NAME: str = os.getenv("MODEL_NAME", "gpt-3.5-turbo")
    # BGE-M3 for Thai/multilingual embeddings (1024 dim)
    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", os.getenv("EMBED_MODEL", "BAAI/bge-m3"))
    
    # Directory Paths
    PROJECT_ROOT: Path = Path(__file__).parent.parent
    DATA_DIR: Path = PROJECT_ROOT / "data"
    DOCUMENTS_DIR: Path = DATA_DIR / "documents"
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
        if not cls.USE_OLLAMA and not cls.OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY is required when not using Ollama. Please set it in your .env file or set USE_OLLAMA=true.")
        return True
    
    @classmethod
    def ensure_directories(cls):
        """Create necessary directories"""
        cls.DOCUMENTS_DIR.mkdir(parents=True, exist_ok=True)
        cls.INDEX_DIR.mkdir(parents=True, exist_ok=True)

# Global settings instance
settings = Settings()
