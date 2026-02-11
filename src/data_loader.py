"""
Data loader for processing various document types
"""

import logging
from pathlib import Path
from typing import List, Optional

from llama_index.core import Document
from llama_index.core.readers import PDFReader, SimpleDirectoryReader
from llama_index.core.node_parser import SentenceSplitter

from config.settings import settings

logger = logging.getLogger(__name__)

class DataLoader:
    """Handle loading and processing of documents"""
    
    def __init__(self):
        self.chunk_size = settings.CHUNK_SIZE
        self.chunk_overlap = settings.CHUNK_OVERLAP
        self.node_parser = SentenceSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap
        )
    
    def load_documents_from_directory(
        self, 
        directory: Optional[Path] = None,
        recursive: bool = True
    ) -> List[Document]:
        """
        Load all documents from a directory
        
        Args:
            directory: Path to directory containing documents
            recursive: Whether to search subdirectories
            
        Returns:
            List of Document objects
        """
        if directory is None:
            directory = settings.DOCUMENTS_DIR
        
        if not directory.exists():
            logger.warning(f"Directory {directory} does not exist")
            return []
        
        logger.info(f"Loading documents from {directory}")
        
        try:
            reader = SimpleDirectoryReader(
                input_dir=str(directory),
                recursive=recursive,
                required_exts=[".pdf", ".txt", ".docx", ".xlsx", ".csv"]
            )
            documents = reader.load_data()
            logger.info(f"Loaded {len(documents)} documents")
            return documents
        except Exception as e:
            logger.error(f"Error loading documents: {e}")
            return []
    
    def load_single_document(self, file_path: Path) -> Optional[Document]:
        """
        Load a single document
        
        Args:
            file_path: Path to the document file
            
        Returns:
            Document object or None if failed
        """
        if not file_path.exists():
            logger.warning(f"File {file_path} does not exist")
            return None
        
        try:
            if file_path.suffix.lower() == ".pdf":
                reader = PDFReader()
                documents = reader.load_data(file_path)
            else:
                reader = SimpleDirectoryReader(input_files=[str(file_path)])
                documents = reader.load_data()
            
            if documents:
                logger.info(f"Loaded document: {file_path.name}")
                return documents[0]
            return None
        except Exception as e:
            logger.error(f"Error loading document {file_path}: {e}")
            return None
    
    def create_nodes(self, documents: List[Document]) -> List[Document]:
        """
        Parse documents into nodes (chunks)
        
        Args:
            documents: List of Document objects
            
        Returns:
            List of chunked Document objects
        """
        if not documents:
            return []
        
        logger.info(f"Creating nodes from {len(documents)} documents")
        nodes = self.node_parser.get_nodes_from_documents(documents)
        logger.info(f"Created {len(nodes)} nodes")
        return nodes
    
    def add_metadata_to_documents(self, documents: List[Document]) -> List[Document]:
        """
        Add metadata to documents
        
        Args:
            documents: List of Document objects
            
        Returns:
            List of Document objects with metadata
        """
        for i, doc in enumerate(documents):
            if not doc.metadata:
                doc.metadata = {}
            
            # Add basic metadata
            doc.metadata.update({
                "document_id": i,
                "source": getattr(doc, 'file_path', 'unknown'),
                "chunk_size": self.chunk_size,
                "chunk_overlap": self.chunk_overlap
            })
        
        return documents
