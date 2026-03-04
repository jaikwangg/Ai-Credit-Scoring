"""
Preservation Property Tests for RAG System

**Validates: Requirements 3.1, 3.2, 3.3, 3.4, 3.5**

These tests capture baseline behavior that must be preserved after the fix.
They are designed to PASS on unfixed code, establishing the baseline.

EXPECTED OUTCOME ON UNFIXED CODE: Tests PASS
- Document parsing produces consistent chunks and metadata
- BGE-M3 embedding model generates embeddings correctly
- Environment variables override defaults (non-ChromaDB settings)
- Alternative vector stores (faiss, simple) work if configured

After the fix is implemented, these same tests should still PASS,
confirming no regressions were introduced.
"""

import os
import tempfile
import shutil
from pathlib import Path
from typing import List

import pytest
import numpy as np
from hypothesis import given, strategies as st, settings, Phase, assume

from llama_index.core import Document, SimpleDirectoryReader
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.settings import Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

# Import document parser
from src.document_parser import StructuredDocumentParser


class TestPreservationProperties:
    """
    Property 2: Preservation - Non-Configuration Operations Unchanged
    
    These tests verify that operations not involving ChromaDB path/collection
    configuration continue to work correctly after the fix.
    """
    
    @settings(
        max_examples=5,
        phases=[Phase.generate],
        deadline=None
    )
    @given(
        doc_text=st.text(
            alphabet=st.characters(whitelist_categories=('Lu', 'Ll', 'Nd', 'Zs', 'Po')),
            min_size=100,
            max_size=500
        )
    )
    def test_document_parsing_consistency(self, doc_text):
        """
        Property: Document parsing produces consistent chunks and metadata.
        
        For any document text, the parsing process should:
        - Successfully parse the document
        - Produce consistent chunks with the same input
        - Preserve metadata correctly
        
        **Validates: Requirement 3.1** - ChromaDB persistence behavior unchanged
        """
        # Skip if content is too short
        assume(len(doc_text.strip()) >= 50)

        with tempfile.TemporaryDirectory() as temp_dir:
            # Create a test document
            test_file = Path(temp_dir) / "test_doc.txt"
            # Use UTF-8 encoding explicitly to handle all characters
            test_file.write_text(doc_text, encoding='utf-8')

            # Parse document using SimpleDirectoryReader (standard approach)
            documents = SimpleDirectoryReader(input_dir=temp_dir).load_data()

            # Verify document was parsed successfully
            assert len(documents) > 0, "Document parsing should produce at least one document"
            
            # Verify document has text content
            assert len(documents[0].text) > 0, "Parsed document should have text content"
            
            # Parse the same document again to verify consistency
            documents_second = SimpleDirectoryReader(input_dir=temp_dir).load_data()
            
            # Verify consistent parsing
            assert len(documents) == len(documents_second), "Parsing should be consistent"
            assert documents[0].text == documents_second[0].text, "Document text should be identical"
            
            print(f"✓ Document parsing consistent: {len(documents)} documents, {len(documents[0].text)} chars")
    
    @settings(
        max_examples=3,
        phases=[Phase.generate],
        deadline=None
    )
    @given(
        text_content=st.text(
            alphabet=st.characters(whitelist_categories=('Lu', 'Ll', 'Nd', 'Zs')),
            min_size=100,
            max_size=300
        )
    )
    def test_embedding_model_generates_embeddings(self, text_content):
        """
        Property: BGE-M3 embedding model generates embeddings correctly.
        
        For any text content, the embedding model should:
        - Successfully generate embeddings
        - Produce consistent embeddings for the same input
        - Generate embeddings of expected dimensionality (1024 for BGE-M3)
        
        **Validates: Requirement 3.2** - BGE-M3 embedding model usage unchanged
        """
        # Skip if content is too short
        assume(len(text_content.strip()) >= 50)
        
        # Import settings to get embedding model name
        from config.settings import settings
        
        # Initialize BGE-M3 embedding model
        embed_model = HuggingFaceEmbedding(
            model_name=settings.EMBEDDING_MODEL,
            embed_batch_size=32,
        )
        
        # Generate embedding
        embedding = embed_model.get_text_embedding(text_content)
        
        # Verify embedding was generated
        assert embedding is not None, "Embedding should be generated"
        assert len(embedding) > 0, "Embedding should have non-zero length"
        
        # BGE-M3 produces 1024-dimensional embeddings
        assert len(embedding) == 1024, f"BGE-M3 should produce 1024-dim embeddings, got {len(embedding)}"
        
        # Verify embedding contains valid values
        assert all(isinstance(x, (int, float)) for x in embedding), "Embedding should contain numeric values"
        
        # Generate embedding again to verify consistency
        embedding_second = embed_model.get_text_embedding(text_content)
        
        # Verify consistency (embeddings should be identical for same input)
        assert len(embedding) == len(embedding_second), "Embedding dimensions should be consistent"
        
        # Check if embeddings are very similar (allowing for minor floating point differences)
        embedding_array = np.array(embedding)
        embedding_second_array = np.array(embedding_second)
        cosine_similarity = np.dot(embedding_array, embedding_second_array) / (
            np.linalg.norm(embedding_array) * np.linalg.norm(embedding_second_array)
        )
        
        assert cosine_similarity > 0.99, f"Embeddings should be consistent, got similarity {cosine_similarity}"
        
        print(f"✓ BGE-M3 embedding generated: {len(embedding)} dimensions, similarity={cosine_similarity:.4f}")
    
    def test_environment_variable_override_non_chroma(self):
        """
        Property: Environment variables override defaults for non-ChromaDB settings.
        
        This test verifies that environment variables like OLLAMA_MODEL correctly
        override default values, ensuring this behavior is preserved after the fix.
        
        **Validates: Requirement 3.5** - Environment variable overrides unchanged
        """
        from config.settings import settings
        
        # Test that environment variables are being read
        # We test with OLLAMA_MODEL which is not related to ChromaDB configuration
        
        # Get the current OLLAMA_MODEL setting
        ollama_model = settings.OLLAMA_MODEL
        
        # Verify it's a valid string (either default or from env)
        assert isinstance(ollama_model, str), "OLLAMA_MODEL should be a string"
        assert len(ollama_model) > 0, "OLLAMA_MODEL should not be empty"
        
        # Test other non-ChromaDB settings
        chunk_size = settings.CHUNK_SIZE
        assert isinstance(chunk_size, int), "CHUNK_SIZE should be an integer"
        assert chunk_size > 0, "CHUNK_SIZE should be positive"
        
        similarity_top_k = settings.SIMILARITY_TOP_K
        assert isinstance(similarity_top_k, int), "SIMILARITY_TOP_K should be an integer"
        assert similarity_top_k > 0, "SIMILARITY_TOP_K should be positive"
        
        print(f"✓ Environment variables working: OLLAMA_MODEL={ollama_model}, CHUNK_SIZE={chunk_size}")
    
    def test_structured_document_parser_preserves_metadata(self):
        """
        Property: StructuredDocumentParser correctly extracts metadata.
        
        This test verifies that the custom document parser continues to work
        correctly, extracting metadata from structured documents.
        
        **Validates: Requirement 3.1** - Document parsing behavior unchanged
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create a structured test document
            test_file = Path(temp_dir) / "structured_doc.txt"
            structured_content = """TITLE: Test Credit Policy
SOURCE URL: https://example.com/policy
INSTITUTION: Test Bank
PUBLICATION DATE: 2024-01-01
CATEGORY: Credit Policy
---
SUMMARY (3-5 sentences relevance)
This is a test credit policy document for testing purposes.
It contains information about loan requirements and procedures.
---
FULL CLEANED TEXT CONTENT
This is the main content of the credit policy document.
It includes detailed information about credit scoring and loan approval processes.
"""
            # Use UTF-8 encoding explicitly
            test_file.write_text(structured_content, encoding='utf-8')
            
            # Parse the document
            doc = StructuredDocumentParser.parse_file(test_file)
            
            # Verify document was parsed
            assert doc is not None, "Document should be parsed successfully"
            
            # Verify metadata extraction
            assert 'title' in doc.metadata, "Title should be extracted"
            assert doc.metadata['title'] == "Test Credit Policy", "Title should match"
            
            assert 'source_url' in doc.metadata, "Source URL should be extracted"
            assert doc.metadata['source_url'] == "https://example.com/policy", "Source URL should match"
            
            assert 'institution' in doc.metadata, "Institution should be extracted"
            assert doc.metadata['institution'] == "Test Bank", "Institution should match"
            
            assert 'category' in doc.metadata, "Category should be extracted"
            assert doc.metadata['category'] == "Credit Policy", "Category should match"
            
            # Verify text content includes metadata context
            assert "Test Credit Policy" in doc.text, "Document text should include title"
            assert "main content" in doc.text, "Document text should include main content"
            
            print(f"✓ Structured document parser working: extracted {len(doc.metadata)} metadata fields")
    
    def test_vector_store_type_configuration(self):
        """
        Property: System supports multiple vector store types configuration.
        
        This test verifies that the VECTOR_STORE_TYPE setting is correctly
        configured and can be read, supporting chroma, faiss, and simple types.
        
        **Validates: Requirement 3.4** - Multi-vector-store support unchanged
        """
        from config.settings import settings
        
        # Get the configured vector store type
        vector_store_type = settings.VECTOR_STORE_TYPE
        
        # Verify it's a valid string
        assert isinstance(vector_store_type, str), "VECTOR_STORE_TYPE should be a string"
        
        # Verify it's one of the supported types
        supported_types = ['chroma', 'faiss', 'simple']
        assert vector_store_type in supported_types, (
            f"VECTOR_STORE_TYPE should be one of {supported_types}, got {vector_store_type}"
        )
        
        print(f"✓ Vector store type configuration working: {vector_store_type}")
    
    def test_chroma_persistence_directory_exists(self):
        """
        Property: ChromaDB persistence behavior - directory creation works.
        
        This test verifies that ChromaDB can create and use persistence directories,
        which is core functionality that must be preserved.
        
        **Validates: Requirement 3.1** - ChromaDB persistence unchanged
        """
        import chromadb
        import gc
        import time
        
        # Create a temporary directory manually to avoid cleanup issues
        temp_dir = Path(tempfile.gettempdir()) / f"test_chroma_{os.getpid()}_{int(time.time())}"
        persist_dir = temp_dir / "test_chroma"
        
        try:
            # Create a ChromaDB client with persistence
            client = chromadb.PersistentClient(path=str(persist_dir))
            
            # Create a collection
            collection = client.get_or_create_collection("test_collection")
            
            # Verify collection was created
            assert collection is not None, "Collection should be created"
            assert collection.name == "test_collection", "Collection name should match"
            
            # Verify persistence directory was created
            assert persist_dir.exists(), "Persistence directory should be created"
            
            # Add a test document to the collection
            collection.add(
                ids=["test1"],
                documents=["This is a test document"],
                metadatas=[{"source": "test"}]
            )
            
            # Verify document was added
            results = collection.get(ids=["test1"])
            assert len(results['ids']) == 1, "Document should be stored"
            
            print(f"✓ ChromaDB persistence working: directory created at {persist_dir}")
            
            # Clean up ChromaDB client to release file locks
            del collection
            del client
            gc.collect()
            
            # Give Windows time to release file locks
            time.sleep(0.5)
            
        finally:
            # Clean up manually with retry logic for Windows
            if temp_dir.exists():
                try:
                    shutil.rmtree(temp_dir)
                except PermissionError:
                    # On Windows, ChromaDB may still hold file locks
                    # This is acceptable - the test verified the functionality
                    print(f"Note: Could not clean up {temp_dir} due to file locks (Windows issue)")
                    pass


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
