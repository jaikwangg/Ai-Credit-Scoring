"""
Vector index creation and management
"""

import logging
from pathlib import Path
from typing import Optional, List

from llama_index.core import VectorStoreIndex, StorageContext, load_index_from_storage
from llama_index.core.settings import Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.vector_stores.faiss import FaissVectorStore
import chromadb
import faiss

from config.settings import settings
from src.data_loader import DataLoader
from src.document_parser import CLEANING_VERSION

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
        persist: bool = True,
        reset_chroma_collection: Optional[bool] = None,
    ) -> VectorStoreIndex:
        """
        Create a new vector index
        
        Args:
            documents: List of documents to index. If None, loads from documents directory
            persist: Whether to persist the index to disk
            reset_chroma_collection: Whether to clear existing Chroma collection
                before indexing. If None, uses settings default.
            
        Returns:
            VectorStoreIndex object
        """
        if documents is None:
            documents = self.data_loader.load_documents_from_directory()
        
        if not documents:
            raise ValueError(
                "No documents to index. Add files to the documents directory and retry."
            )
        
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
            if reset_chroma_collection is None:
                reset_chroma_collection = settings.RESET_CHROMA_COLLECTION_ON_INGEST
            index = self._create_chroma_index(
                nodes,
                reset_collection=bool(reset_chroma_collection),
            )
        elif self.vector_store_type == "faiss":
            index = self._create_faiss_index(nodes)
        else:
            # Simple in-memory index
            index = VectorStoreIndex(nodes)
        
        if persist:
            self._persist_index(index)
        
        logger.info("Index created successfully")
        if not self._verify_cleaning_fingerprint(index):
            logger.warning(
                "Parser changes not reflected in index. Check DATA_DIR / parser pipeline."
            )
        else:
            logger.info("Fingerprint sanity check passed: CLEANING_VERSION=%s", CLEANING_VERSION)
        return index
    
    def _create_chroma_index(self, nodes: List, reset_collection: bool = False) -> VectorStoreIndex:
        """Create index with Chroma vector store"""
        # Initialize Chroma client
        chroma_client = chromadb.PersistentClient(path=settings.CHROMA_PERSIST_DIR)
        if reset_collection:
            self._reset_chroma_collection(chroma_client)
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
        if self.vector_store_type == "chroma":
            chroma_dir = Path(settings.CHROMA_PERSIST_DIR)
            if not chroma_dir.exists():
                logger.warning(f"Chroma directory {chroma_dir} does not exist")
                return None
        elif not self.index_dir.exists():
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

        # Clear existing Chroma collection when using Chroma vector store
        if (
            self.vector_store_type == "chroma"
            and settings.RESET_CHROMA_COLLECTION_ON_INGEST
        ):
            chroma_client = chromadb.PersistentClient(path=settings.CHROMA_PERSIST_DIR)
            self._reset_chroma_collection(chroma_client)

        # Clear existing index directory
        if self.index_dir.exists():
            import shutil
            shutil.rmtree(self.index_dir)

        return self.create_index()

    def _reset_chroma_collection(self, chroma_client: chromadb.PersistentClient) -> None:
        """Delete existing Chroma collection if present."""
        try:
            chroma_client.delete_collection(settings.CHROMA_COLLECTION)
            logger.info(
                "Deleted existing Chroma collection '%s' before rebuild",
                settings.CHROMA_COLLECTION,
            )
        except Exception:
            # Collection may not exist yet, which is fine.
            logger.debug(
                "Chroma collection '%s' did not exist before rebuild",
                settings.CHROMA_COLLECTION,
            )
    
    def get_index_stats(self, index: VectorStoreIndex) -> dict:
        """
        Get statistics about the index
        
        Args:
            index: VectorStoreIndex object
            
        Returns:
            Dictionary with index statistics
        """
        try:
            total_docs = 0
            docstore = getattr(index, "docstore", None)
            docs = getattr(docstore, "docs", None)
            if isinstance(docs, (dict, list, tuple, set)):
                total_docs = len(docs)

            stats = {
                "total_docs": total_docs,
                "vector_store_type": self.vector_store_type,
                "index_type": type(index).__name__,
                "index_dir": str(self.index_dir)
            }
            
            return stats
        except Exception as e:
            logger.error(f"Error getting index stats: {e}")
            return {
                "total_docs": "Unknown",
                "vector_store_type": self.vector_store_type,
                "index_dir": str(self.index_dir)
            }

    def _verify_cleaning_fingerprint(self, index: VectorStoreIndex) -> bool:
        """Run a quick retrieval sanity query to ensure parser fingerprint was indexed."""
        try:
            retriever = index.as_retriever(similarity_top_k=5)
            nodes = retriever.retrieve("CLEANING_VERSION")
        except Exception as exc:
            logger.warning("Fingerprint sanity query failed: %s", exc)
            return False

        expected_line = f"CLEANING_VERSION: {CLEANING_VERSION}"
        for node in nodes or []:
            text = getattr(node, "text", None)
            if not isinstance(text, str):
                inner = getattr(node, "node", None)
                text = getattr(inner, "text", "") if inner is not None else ""
            metadata = getattr(node, "metadata", None)
            if not isinstance(metadata, dict):
                inner = getattr(node, "node", None)
                metadata = getattr(inner, "metadata", {}) if inner is not None else {}

            if expected_line in str(text) or str(metadata.get("cleaning_version", "")) == CLEANING_VERSION:
                return True
        return False
