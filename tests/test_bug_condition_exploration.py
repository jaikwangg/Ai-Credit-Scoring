"""
Bug Condition Exploration Test for ChromaDB Configuration Mismatch

**Validates: Requirements 2.1, 2.2, 2.3, 2.4**

This test is designed to FAIL on unfixed code to confirm the bug exists.
When it fails, it demonstrates that ingestion and query modules use different
ChromaDB configurations, resulting in 0 nodes retrieved.

EXPECTED OUTCOME ON UNFIXED CODE: Test FAILS
- Configuration mismatch detected
- Documents ingested to one location, queries look in another
- Result: 0 nodes retrieved, "Empty Response"

After the fix is implemented, this same test should PASS, confirming the
expected behavior is satisfied.
"""

import os
import tempfile
import shutil
from pathlib import Path

import pytest
from hypothesis import given, strategies as st, settings, Phase

# Import configurations from both sources to detect mismatch
import src.settings as ingest_settings
from config.settings import settings as query_settings


class TestBugConditionExploration:
    """
    Property 1: Fault Condition - Configuration Mismatch Detection
    
    This test explores the bug condition by:
    1. Checking if ingestion and query modules use different ChromaDB configs
    2. Attempting end-to-end ingestion and retrieval
    3. Documenting the failure (0 nodes retrieved)
    """
    
    def test_configuration_mismatch_detection(self):
        """
        Test that ingestion and query modules use different ChromaDB configurations.
        
        This test will FAIL on unfixed code, confirming the configuration mismatch.
        """
        # Get configuration values from both sources
        ingest_chroma_dir = ingest_settings.CHROMA_PERSIST_DIR
        ingest_collection = ingest_settings.CHROMA_COLLECTION
        
        query_chroma_dir = query_settings.CHROMA_PERSIST_DIR
        query_collection = query_settings.CHROMA_COLLECTION
        
        # Document the configurations being used
        print(f"\n=== Configuration Mismatch Detection ===")
        print(f"Ingestion config: dir={ingest_chroma_dir}, collection={ingest_collection}")
        print(f"Query config: dir={query_chroma_dir}, collection={query_collection}")
        
        # Check for mismatch
        config_mismatch = (
            ingest_chroma_dir != query_chroma_dir or 
            ingest_collection != query_collection
        )
        
        if config_mismatch:
            print(f"❌ CONFIGURATION MISMATCH DETECTED!")
            print(f"   Ingestion uses: {ingest_chroma_dir}/{ingest_collection}")
            print(f"   Query uses: {query_chroma_dir}/{query_collection}")
        else:
            print(f"✓ Configurations match (bug may be fixed)")
        
        # On unfixed code, this assertion will pass (confirming mismatch exists)
        # After fix, this will fail, and we'll need to invert the logic
        # For now, we assert that configs should be EQUAL (expected behavior)
        assert ingest_chroma_dir == query_chroma_dir, (
            f"Configuration mismatch: ingestion uses {ingest_chroma_dir}, "
            f"query uses {query_chroma_dir}"
        )
        assert ingest_collection == query_collection, (
            f"Collection mismatch: ingestion uses {ingest_collection}, "
            f"query uses {query_collection}"
        )
    
    @settings(
        max_examples=3,
        phases=[Phase.generate, Phase.target],
        deadline=None
    )
    @given(
        doc_content=st.text(
            alphabet=st.characters(whitelist_categories=('Lu', 'Ll', 'Nd', 'Zs')),
            min_size=50,
            max_size=200
        )
    )
    def test_end_to_end_retrieval_after_ingestion(self, doc_content):
        """
        Property-based test: After ingesting a document with known content,
        queries should retrieve >0 nodes.
        
        This test will FAIL on unfixed code (0 nodes retrieved), confirming
        that documents are stored in a different location than where queries look.
        
        **Validates: Requirements 2.1, 2.2, 2.3, 2.4**
        """
        # Skip if content is too short or empty
        if len(doc_content.strip()) < 20:
            return
        
        # Create a temporary directory for test documents
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create a test document with known content
            test_doc_path = Path(temp_dir) / "test_document.txt"
            test_content = f"Test Document Content: {doc_content}"
            test_doc_path.write_text(test_content, encoding='utf-8')
            
            print(f"\n=== End-to-End Retrieval Test ===")
            print(f"Test document created with content length: {len(test_content)}")
            
            # Import ingestion and query modules
            from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, StorageContext
            from llama_index.core.settings import Settings
            from llama_index.embeddings.huggingface import HuggingFaceEmbedding
            from llama_index.vector_stores.chroma import ChromaVectorStore
            import chromadb
            
            # Setup embedding model (BGE-M3)
            Settings.embed_model = HuggingFaceEmbedding(
                model_name=ingest_settings.EMBED_MODEL,
                embed_batch_size=32,
            )
            
            # === INGESTION PHASE (using src/settings.py config) ===
            print(f"Ingesting to: {ingest_settings.CHROMA_PERSIST_DIR}/{ingest_settings.CHROMA_COLLECTION}")
            
            # Create ingestion storage context
            os.makedirs(ingest_settings.CHROMA_PERSIST_DIR, exist_ok=True)
            ingest_client = chromadb.PersistentClient(path=ingest_settings.CHROMA_PERSIST_DIR)
            ingest_collection = ingest_client.get_or_create_collection(ingest_settings.CHROMA_COLLECTION)
            ingest_vector_store = ChromaVectorStore(chroma_collection=ingest_collection)
            ingest_storage_context = StorageContext.from_defaults(vector_store=ingest_vector_store)
            
            # Load and ingest document
            documents = SimpleDirectoryReader(input_dir=temp_dir).load_data()
            ingest_index = VectorStoreIndex.from_documents(
                documents,
                storage_context=ingest_storage_context
            )
            
            print(f"✓ Document ingested successfully")
            
            # === QUERY PHASE (using config/settings.py config) ===
            print(f"Querying from: {query_settings.CHROMA_PERSIST_DIR}/{query_settings.CHROMA_COLLECTION}")
            
            # Create query storage context (using different config)
            os.makedirs(query_settings.CHROMA_PERSIST_DIR, exist_ok=True)
            query_client = chromadb.PersistentClient(path=query_settings.CHROMA_PERSIST_DIR)
            query_collection = query_client.get_or_create_collection(query_settings.CHROMA_COLLECTION)
            query_vector_store = ChromaVectorStore(chroma_collection=query_collection)
            query_index = VectorStoreIndex.from_vector_store(query_vector_store)
            
            # Create retriever (no LLM needed for retrieval)
            from llama_index.core.retrievers import VectorIndexRetriever
            
            retriever = VectorIndexRetriever(
                index=query_index,
                similarity_top_k=4
            )
            
            # Extract a query phrase from the document content
            words = doc_content.strip().split()[:5]
            query_text = " ".join(words) if words else "test"
            
            print(f"Querying for: '{query_text}'")
            
            # Execute retrieval (no LLM needed)
            retrieved_nodes = retriever.retrieve(query_text)
            
            # Check retrieval results
            num_nodes = len(retrieved_nodes)
            
            print(f"Retrieved nodes: {num_nodes}")
            if num_nodes > 0:
                print(f"First node preview: {retrieved_nodes[0].text[:100]}...")
            
            # Document the counterexample
            if num_nodes == 0:
                print(f"\n❌ COUNTEREXAMPLE FOUND:")
                print(f"   Ingestion config: {ingest_settings.CHROMA_PERSIST_DIR}/{ingest_settings.CHROMA_COLLECTION}")
                print(f"   Query config: {query_settings.CHROMA_PERSIST_DIR}/{query_settings.CHROMA_COLLECTION}")
                print(f"   Result: {num_nodes} nodes retrieved")
            
            # This assertion encodes the EXPECTED behavior:
            # After ingestion, queries should retrieve >0 nodes
            # On unfixed code, this will FAIL (confirming the bug)
            # After fix, this will PASS (confirming expected behavior)
            assert num_nodes > 0, (
                f"Expected >0 nodes after ingestion, but got {num_nodes}. "
                f"Configuration mismatch: ingestion used {ingest_settings.CHROMA_PERSIST_DIR}/"
                f"{ingest_settings.CHROMA_COLLECTION}, query used {query_settings.CHROMA_PERSIST_DIR}/"
                f"{query_settings.CHROMA_COLLECTION}"
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
