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
    
    # Model Settings
    MODEL_NAME: str = os.getenv("MODEL_NAME", "gpt-3.5-turbo")
    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "text-embedding-ada-002")
    
    # Directory Paths
    PROJECT_ROOT: Path = Path(__file__).parent.parent
    DATA_DIR: Path = PROJECT_ROOT / "data"
    DOCUMENTS_DIR: Path = DATA_DIR / "documents"
    INDEX_DIR: Path = PROJECT_ROOT / "data" / "index"
    
    # Vector Store Settings
    VECTOR_STORE_TYPE: str = os.getenv("VECTOR_STORE_TYPE", "chroma")  # chroma, faiss, simple
    CHROMA_PERSIST_DIR: str = str(INDEX_DIR / "chroma")
    
    # Index Settings
    CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "1024"))
    CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "20"))
    
    # Query Settings
    SIMILARITY_TOP_K: int = int(os.getenv("SIMILARITY_TOP_K", "4"))
    RESPONSE_MODE: str = os.getenv("RESPONSE_MODE", "compact")
    
    # Logging
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    
    @classmethod
    def validate(cls) -> bool:
        """Validate required settings"""
        if not cls.OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY is required. Please set it in your .env file.")
        return True
    
    @classmethod
    def ensure_directories(cls):
        """Create necessary directories"""
        cls.DOCUMENTS_DIR.mkdir(parents=True, exist_ok=True)
        cls.INDEX_DIR.mkdir(parents=True, exist_ok=True)

# Global settings instance
settings = Settings()
