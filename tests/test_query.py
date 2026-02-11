"""
Unit tests for query functionality
"""

import unittest
import sys
from pathlib import Path
from unittest.mock import Mock, patch

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.indexer import IndexManager
from src.query_engine import QueryEngineManager
from src.data_loader import DataLoader
from src.utils import validate_environment, format_response
from config.settings import settings

class TestDataLoader(unittest.TestCase):
    """Test data loader functionality"""
    
    def setUp(self):
        self.data_loader = DataLoader()
    
    def test_chunk_size_setting(self):
        """Test chunk size configuration"""
        self.assertEqual(self.data_loader.chunk_size, settings.CHUNK_SIZE)
        self.assertEqual(self.data_loader.chunk_overlap, settings.CHUNK_OVERLAP)
    
    def test_add_metadata_to_documents(self):
        """Test metadata addition to documents"""
        from llama_index.core import Document
        
        # Create mock document
        doc = Mock(spec=Document)
        doc.text = "Test document content"
        doc.metadata = {}
        
        documents = [doc]
        result = self.data_loader.add_metadata_to_documents(documents)
        
        # Check metadata was added
        self.assertIn("document_id", result[0].metadata)
        self.assertIn("chunk_size", result[0].metadata)
        self.assertEqual(result[0].metadata["document_id"], 0)

class TestIndexManager(unittest.TestCase):
    """Test index manager functionality"""
    
    def setUp(self):
        self.index_manager = IndexManager()
    
    def test_initialization(self):
        """Test index manager initialization"""
        self.assertIsNotNone(self.index_manager.data_loader)
        self.assertEqual(self.index_manager.vector_store_type, settings.VECTOR_STORE_TYPE)
    
    @patch('src.indexer.DataLoader.load_documents_from_directory')
    def test_create_index_no_documents(self, mock_load):
        """Test creating index with no documents"""
        mock_load.return_value = []
        
        index = self.index_manager.create_index()
        self.assertIsNone(index)
    
    def test_get_index_stats_empty(self):
        """Test getting stats for empty index"""
        mock_index = Mock()
        mock_index.doc_store = None
        mock_index.index_struct = Mock()
        mock_index.index_struct.__class__.__name__ = "MockIndexStruct"
        
        stats = self.index_manager.get_index_stats(mock_index)
        self.assertEqual(stats["total_docs"], 0)

class TestQueryEngineManager(unittest.TestCase):
    """Test query engine manager functionality"""
    
    def setUp(self):
        # Create mock index
        self.mock_index = Mock()
        self.query_manager = QueryEngineManager(self.mock_index)
    
    def test_initialization(self):
        """Test query engine manager initialization"""
        self.assertEqual(self.query_manager.index, self.mock_index)
        self.assertEqual(self.query_manager.similarity_top_k, settings.SIMILARITY_TOP_K)
        self.assertEqual(self.query_manager.response_mode, settings.RESPONSE_MODE)
    
    @patch('src.query_engine.VectorIndexRetriever')
    @patch('src.query_engine.get_response_synthesizer')
    def test_create_query_engine(self, mock_synthesizer, mock_retriever):
        """Test query engine creation"""
        # Mock the dependencies
        mock_retriever_instance = Mock()
        mock_retriever.return_value = mock_retriever_instance
        
        mock_synthesizer_instance = Mock()
        mock_synthesizer.return_value = mock_synthesizer_instance
        
        # Create query engine
        query_engine = self.query_manager.create_query_engine(
            similarity_top_k=3,
            response_mode="compact"
        )
        
        # Verify the mocks were called with correct parameters
        mock_retriever.assert_called_once_with(
            index=self.mock_index,
            similarity_top_k=3
        )
        mock_synthesizer.assert_called_once_with(
            response_mode="compact",
            llm=self.query_manager.llm
        )
    
    def test_query_response_format(self):
        """Test query response format"""
        # Mock query engine and response
        mock_query_engine = Mock()
        mock_response = Mock()
        mock_response.__str__ = Mock(return_value="Test answer")
        mock_response.source_nodes = []
        
        mock_query_engine.query.return_value = mock_response
        
        with patch.object(self.query_manager, 'create_query_engine', return_value=mock_query_engine):
            result = self.query_manager.query("Test question")
            
            # Check response format
            self.assertIn("question", result)
            self.assertIn("answer", result)
            self.assertIn("response", result)
            self.assertEqual(result["question"], "Test question")
            self.assertEqual(result["answer"], "Test answer")

class TestUtils(unittest.TestCase):
    """Test utility functions"""
    
    def test_format_response(self):
        """Test response formatting"""
        response = {
            "answer": "This is a test answer",
            "sources": [
                {
                    "content": "Test source content",
                    "metadata": {"source": "test.txt"},
                    "score": 0.85
                }
            ]
        }
        
        formatted = format_response(response)
        
        self.assertIn("This is a test answer", formatted)
        self.assertIn("Test source content", formatted)
        self.assertIn("test.txt", formatted)
    
    def test_format_response_no_sources(self):
        """Test response formatting without sources"""
        response = {
            "answer": "This is a test answer"
        }
        
        formatted = format_response(response)
        
        self.assertIn("This is a test answer", formatted)
        self.assertNotIn("Sources", formatted)
    
    def test_get_document_summary_empty(self):
        """Test document summary with empty list"""
        from src.utils import get_document_summary
        
        summary = get_document_summary([])
        self.assertEqual(summary["total_documents"], 0)
    
    def test_get_document_summary_with_docs(self):
        """Test document summary with documents"""
        from src.utils import get_document_summary
        from llama_index.core import Document
        
        # Create mock documents
        doc1 = Mock(spec=Document)
        doc1.text = "This is document one with some words."
        doc1.file_path = "test1.txt"
        
        doc2 = Mock(spec=Document)
        doc2.text = "This is document two with more words."
        doc2.file_path = "test2.txt"
        
        summary = get_document_summary([doc1, doc2])
        
        self.assertEqual(summary["total_documents"], 2)
        self.assertGreater(summary["total_characters"], 0)
        self.assertGreater(summary["total_words"], 0)
        self.assertGreater(summary["avg_chars_per_doc"], 0)
        self.assertGreater(summary["avg_words_per_doc"], 0)

class TestSettings(unittest.TestCase):
    """Test settings configuration"""
    
    def test_settings_attributes(self):
        """Test that settings have required attributes"""
        self.assertTrue(hasattr(settings, 'OPENAI_API_KEY'))
        self.assertTrue(hasattr(settings, 'MODEL_NAME'))
        self.assertTrue(hasattr(settings, 'EMBEDDING_MODEL'))
        self.assertTrue(hasattr(settings, 'CHUNK_SIZE'))
        self.assertTrue(hasattr(settings, 'SIMILARITY_TOP_K'))
    
    def test_directory_paths(self):
        """Test directory path settings"""
        self.assertTrue(settings.PROJECT_ROOT.exists())
        self.assertIsInstance(settings.DATA_DIR, Path)
        self.assertIsInstance(settings.DOCUMENTS_DIR, Path)
        self.assertIsInstance(settings.INDEX_DIR, Path)

def run_integration_tests():
    """Run integration tests (require actual API key)"""
    print("Running integration tests...")
    
    # Skip if no API key
    if not settings.OPENAI_API_KEY:
        print("Skipping integration tests - no OpenAI API key found")
        return
    
    try:
        # Test actual document loading and indexing
        data_loader = DataLoader()
        documents = data_loader.load_documents_from_directory()
        
        if documents:
            print(f"✓ Loaded {len(documents)} documents")
            
            # Test indexing
            index_manager = IndexManager()
            index = index_manager.create_index(documents, persist=False)
            
            if index:
                print("✓ Created index successfully")
                
                # Test querying
                query_manager = QueryEngineManager(index)
                result = query_manager.query("What is this document about?")
                
                if result and result.get("answer"):
                    print("✓ Query executed successfully")
                    print(f"Sample answer: {result['answer'][:100]}...")
                else:
                    print("✗ Query failed")
            else:
                print("✗ Index creation failed")
        else:
            print("No documents found for integration testing")
            
    except Exception as e:
        print(f"Integration test failed: {e}")

if __name__ == "__main__":
    # Run unit tests
    print("Running unit tests...")
    unittest.main(verbosity=2, exit=False)
    
    # Run integration tests
    print("\n" + "=" * 50)
    run_integration_tests()
