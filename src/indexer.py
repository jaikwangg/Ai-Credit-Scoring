"""
Vector index creation and management
"""

import logging
from pathlib import Path
from typing import Optional, List

from llama_index.core import VectorStoreIndex, StorageContext, load_index_from_storage
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.readers import SimpleDirectoryReader
from llama_index.core.settings import Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.vector_stores.faiss import FaissVectorStore
import chromadb
import faiss

from config.settings import settings
from src.data_loader import DataLoader

logger = logging.getLogger(__name__)

class IndexManager:
    """Manage vector index creation and operations"""
    
    def __init__(self):
        self.data_loader = DataLoader()
        self.index_dir = settings.INDEX_DIR
        self.vector_store_type = settings.VECTOR_STORE_TYPE
        
    def create_index(
        self, 
        documents: Optional[List] = None,
        persist: bool = True
    ) -> VectorStoreIndex:
        """
        Create a new vector index
        
        Args:
            documents: List of documents to index. If None, loads from documents directory
            persist: Whether to persist the index to disk
            
        Returns:
            VectorStoreIndex object
        """
        if documents is None:
            documents = self.data_loader.load_documents_from_directory()
        
        if not documents:
            logger.warning("No documents to index")
            return None
        
        # Add metadata to documents
        documents = self.data_loader.add_metadata_to_documents(documents)
        
        # Create nodes
        nodes = self.data_loader.create_nodes(documents)
        
        logger.info(f"Creating index with {len(nodes)} nodes using {self.vector_store_type}")
        
        # BGE-M3 embeddings for Thai/multilingual support (1024 dim)
        Settings.embed_model = HuggingFaceEmbedding(
            model_name=settings.EMBEDDING_MODEL,
            embed_batch_size=32,
        )
        
        # Create index based on vector store type
        if self.vector_store_type == "chroma":
            index = self._create_chroma_index(nodes)
        elif self.vector_store_type == "faiss":
            index = self._create_faiss_index(nodes)
        else:
            # Simple in-memory index
            index = VectorStoreIndex(nodes)
        
        if persist:
            self._persist_index(index)
        
        logger.info("Index created successfully")
        return index
    
    def _create_chroma_index(self, nodes: List) -> VectorStoreIndex:
        """Create index with Chroma vector store"""
        # Initialize Chroma client
        chroma_client = chromadb.PersistentClient(path=settings.CHROMA_PERSIST_DIR)
        chroma_collection = chroma_client.get_or_create_collection(settings.CHROMA_COLLECTION)
        
        # Create vector store
        vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        
        # Create index
        index = VectorStoreIndex(nodes=nodes, storage_context=storage_context)
        return index
    
    def _create_faiss_index(self, nodes: List) -> VectorStoreIndex:
        """Create index with FAISS vector store"""
        # Create FAISS index (1024 dim for BGE-M3)
        d = Settings.embed_model.get_text_embedding_dimension()
        faiss_index = faiss.IndexFlatL2(d)
        
        # Create vector store
        vector_store = FaissVectorStore(faiss_index=faiss_index)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        
        # Create index
        index = VectorStoreIndex(nodes=nodes, storage_context=storage_context)
        return index
    
    def _persist_index(self, index: VectorStoreIndex):
        """Persist index to disk"""
        try:
            self.index_dir.mkdir(parents=True, exist_ok=True)
            
            if self.vector_store_type in ("simple", "faiss"):
                # For simple vector store, use LlamaIndex's built-in persistence
                index.storage_context.persist(str(self.index_dir))
            
            logger.info(f"Index persisted to {self.index_dir}")
        except Exception as e:
            logger.error(f"Error persisting index: {e}")
    
    def load_index(self) -> Optional[VectorStoreIndex]:
        """
        Load existing index from disk
        
        Returns:
            VectorStoreIndex object or None if not found
        """
        if not self.index_dir.exists():
            logger.warning(f"Index directory {self.index_dir} does not exist")
            return None
        
        try:
            logger.info(f"Loading index from {self.index_dir}")
            
            if self.vector_store_type == "chroma":
                index = self._load_chroma_index()
            elif self.vector_store_type == "faiss":
                index = self._load_faiss_index()
            else:
                # Simple vector store
                storage_context = StorageContext.from_defaults(
                    persist_dir=str(self.index_dir)
                )
                index = load_index_from_storage(storage_context)
            
            logger.info("Index loaded successfully")
            return index
        except Exception as e:
            logger.error(f"Error loading index: {e}")
            return None
    
    def _load_chroma_index(self) -> VectorStoreIndex:
        """Load Chroma index"""
        # Must use same BGE-M3 embedding model for query encoding
        Settings.embed_model = HuggingFaceEmbedding(
            model_name=settings.EMBEDDING_MODEL,
            embed_batch_size=32,
        )
        chroma_client = chromadb.PersistentClient(path=settings.CHROMA_PERSIST_DIR)
        chroma_collection = chroma_client.get_or_create_collection(settings.CHROMA_COLLECTION)
        vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
        index = VectorStoreIndex.from_vector_store(vector_store)
        return index
    
    def _load_faiss_index(self) -> VectorStoreIndex:
        """Load FAISS index"""
        faiss_index = faiss.read_index(str(self.index_dir / "faiss.index"))
        vector_store = FaissVectorStore(faiss_index=faiss_index)
        index = VectorStoreIndex.from_vector_store(vector_store)
        return index
    
    def rebuild_index(self) -> VectorStoreIndex:
        """
        Rebuild the entire index from documents
        
        Returns:
            New VectorStoreIndex object
        """
        logger.info("Rebuilding index...")
        
        # Clear existing index directory
        if self.index_dir.exists():
            import shutil
            shutil.rmtree(self.index_dir)
        
        return self.create_index()
    
    def get_index_stats(self, index: VectorStoreIndex) -> dict:
        """
        Get statistics about the index
        
        Args:
            index: VectorStoreIndex object
            
        Returns:
            Dictionary with index statistics
        """
        try:
            doc_store = index.doc_store
            index_struct = index.index_struct
            
            stats = {
                "total_docs": len(doc_store.docs) if doc_store else 0,
                "vector_store_type": self.vector_store_type,
                "index_type": type(index_struct).__name__,
                "index_dir": str(self.index_dir)
            }
            
            return stats
        except Exception as e:
            logger.error(f"Error getting index stats: {e}")
            return {}
