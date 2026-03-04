"""
Unit tests for Chroma collection reset behavior during indexing.
"""

import sys
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from config.settings import settings
from src.indexer import IndexManager


class TestIndexerChromaReset(unittest.TestCase):
    """Test Chroma reset logic for deterministic re-ingest behavior."""

    @patch("src.indexer.VectorStoreIndex")
    @patch("src.indexer.StorageContext")
    @patch("src.indexer.ChromaVectorStore")
    @patch("src.indexer.chromadb.PersistentClient")
    def test_create_chroma_index_resets_collection_when_requested(
        self,
        mock_client_ctor,
        _mock_vector_store,
        mock_storage_context,
        _mock_vector_index,
    ):
        manager = IndexManager()
        mock_client = Mock()
        mock_client_ctor.return_value = mock_client
        mock_client.get_or_create_collection.return_value = Mock()
        mock_storage_context.from_defaults.return_value = Mock()

        manager._create_chroma_index(nodes=[Mock()], reset_collection=True)

        mock_client.delete_collection.assert_called_once_with(
            settings.CHROMA_COLLECTION
        )
        mock_client.get_or_create_collection.assert_called_once_with(
            settings.CHROMA_COLLECTION
        )

    @patch("src.indexer.VectorStoreIndex")
    @patch("src.indexer.StorageContext")
    @patch("src.indexer.ChromaVectorStore")
    @patch("src.indexer.chromadb.PersistentClient")
    def test_create_chroma_index_does_not_reset_when_not_requested(
        self,
        mock_client_ctor,
        _mock_vector_store,
        mock_storage_context,
        _mock_vector_index,
    ):
        manager = IndexManager()
        mock_client = Mock()
        mock_client_ctor.return_value = mock_client
        mock_client.get_or_create_collection.return_value = Mock()
        mock_storage_context.from_defaults.return_value = Mock()

        manager._create_chroma_index(nodes=[Mock()], reset_collection=False)

        mock_client.delete_collection.assert_not_called()
        mock_client.get_or_create_collection.assert_called_once_with(
            settings.CHROMA_COLLECTION
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
