"""
Unit tests for metadata formatting in query results
Tests for task 2.2: Integrate metadata formatting into QueryEngineManager.query() method
"""

import unittest
import sys
from pathlib import Path
from unittest.mock import Mock, patch

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.query_engine import QueryEngineManager, format_source_display


class TestMetadataFormatting(unittest.TestCase):
    """Test metadata formatting in query results"""
    
    def test_format_source_display_with_all_fields(self):
        """Test format_source_display with all metadata fields present"""
        metadata = {
            "title": "Test Document",
            "category": "Research",
            "source_url": "https://example.com/doc",
            "institution": "Test University",
            "publication_date": "2024-01-01",
            "file_name": "test.txt"
        }
        
        result = format_source_display(metadata)
        
        self.assertEqual(result["title"], "Test Document")
        self.assertEqual(result["category"], "Research")
        self.assertEqual(result["source_url"], "https://example.com/doc")
        self.assertEqual(result["institution"], "Test University")
        self.assertEqual(result["publication_date"], "2024-01-01")
    
    def test_format_source_display_with_missing_title(self):
        """Test format_source_display falls back to file_name when title is missing"""
        metadata = {
            "category": "Research",
            "file_name": "test_document.txt"
        }
        
        result = format_source_display(metadata)
        
        self.assertEqual(result["title"], "test_document.txt")
        self.assertEqual(result["category"], "Research")
    
    def test_format_source_display_with_missing_title_and_filename(self):
        """Test format_source_display falls back to 'Unknown' when both title and file_name are missing"""
        metadata = {
            "category": "Research"
        }
        
        result = format_source_display(metadata)
        
        self.assertEqual(result["title"], "Unknown")
        self.assertEqual(result["category"], "Research")
    
    def test_format_source_display_with_missing_category(self):
        """Test format_source_display falls back to 'Uncategorized' when category is missing"""
        metadata = {
            "title": "Test Document"
        }
        
        result = format_source_display(metadata)
        
        self.assertEqual(result["title"], "Test Document")
        self.assertEqual(result["category"], "Uncategorized")
    
    def test_format_source_display_with_empty_metadata(self):
        """Test format_source_display with empty metadata dict"""
        metadata = {}
        
        result = format_source_display(metadata)
        
        self.assertEqual(result["title"], "Unknown")
        self.assertEqual(result["category"], "Uncategorized")
        self.assertIsNone(result["source_url"])
        self.assertIsNone(result["institution"])
        self.assertIsNone(result["publication_date"])
    
    def test_query_method_uses_formatted_metadata(self):
        """Test that QueryEngineManager.query() uses format_source_display for metadata"""
        # Create mock index
        mock_index = Mock()
        query_manager = QueryEngineManager(mock_index)
        
        # Create mock source node with metadata
        mock_node = Mock()
        mock_node.text = "This is test content from a document."
        mock_node.metadata = {
            "title": "Integration Test Doc",
            "category": "Testing",
            "source_url": "https://test.com",
            "file_name": "test.txt"
        }
        mock_node.score = 0.95
        
        # Create mock response with source nodes
        mock_response = Mock()
        mock_response.__str__ = Mock(return_value="Test answer")
        mock_response.source_nodes = [mock_node]
        
        # Create mock query engine
        mock_query_engine = Mock()
        mock_query_engine.query.return_value = mock_response
        
        # Patch create_query_engine to return our mock
        with patch.object(query_manager, 'create_query_engine', return_value=mock_query_engine):
            result = query_manager.query("Test question")
            
            # Verify the result structure
            self.assertIn("sources", result)
            self.assertEqual(len(result["sources"]), 1)
            
            # Verify metadata is formatted
            source = result["sources"][0]
            self.assertIn("metadata", source)
            
            # Check that formatted metadata has the expected structure
            metadata = source["metadata"]
            self.assertEqual(metadata["title"], "Integration Test Doc")
            self.assertEqual(metadata["category"], "Testing")
            self.assertEqual(metadata["source_url"], "https://test.com")
            
            # Verify content and score are preserved
            self.assertIn("content", source)
            self.assertEqual(source["score"], 0.95)
    
    def test_query_method_with_missing_metadata_fields(self):
        """Test that QueryEngineManager.query() handles missing metadata fields with fallbacks"""
        # Create mock index
        mock_index = Mock()
        query_manager = QueryEngineManager(mock_index)
        
        # Create mock source node with partial metadata
        mock_node = Mock()
        mock_node.text = "Content without full metadata."
        mock_node.metadata = {
            "file_name": "partial_metadata.txt"
            # Missing title and category
        }
        mock_node.score = 0.80
        
        # Create mock response
        mock_response = Mock()
        mock_response.__str__ = Mock(return_value="Test answer")
        mock_response.source_nodes = [mock_node]
        
        # Create mock query engine
        mock_query_engine = Mock()
        mock_query_engine.query.return_value = mock_response
        
        # Patch create_query_engine
        with patch.object(query_manager, 'create_query_engine', return_value=mock_query_engine):
            result = query_manager.query("Test question")
            
            # Verify fallbacks are applied
            source = result["sources"][0]
            metadata = source["metadata"]
            
            # Title should fall back to file_name
            self.assertEqual(metadata["title"], "partial_metadata.txt")
            # Category should fall back to "Uncategorized"
            self.assertEqual(metadata["category"], "Uncategorized")
    
    def test_chat_method_uses_formatted_metadata(self):
        """Test that QueryEngineManager.chat() also uses format_source_display for metadata"""
        # Create mock index
        mock_index = Mock()
        query_manager = QueryEngineManager(mock_index)
        
        # Create mock source node
        mock_node = Mock()
        mock_node.text = "Chat response content."
        mock_node.metadata = {
            "title": "Chat Test Doc",
            "category": "Conversation"
        }
        mock_node.score = 0.88
        
        # Create mock response
        mock_response = Mock()
        mock_response.__str__ = Mock(return_value="Chat answer")
        mock_response.source_nodes = [mock_node]
        
        # Create mock chat engine
        mock_chat_engine = Mock()
        mock_chat_engine.chat.return_value = mock_response
        
        # Patch create_chat_engine
        with patch.object(query_manager, 'create_chat_engine', return_value=mock_chat_engine):
            result = query_manager.chat("Test message")
            
            # Verify metadata is formatted
            self.assertIn("sources", result)
            source = result["sources"][0]
            metadata = source["metadata"]
            
            self.assertEqual(metadata["title"], "Chat Test Doc")
            self.assertEqual(metadata["category"], "Conversation")


if __name__ == "__main__":
    unittest.main(verbosity=2)
