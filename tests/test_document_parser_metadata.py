"""
Unit tests for structured parser metadata behavior.
"""

import sys
import tempfile
import unittest
from pathlib import Path

from llama_index.core.schema import MetadataMode

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.document_parser import CLEANING_VERSION, StructuredDocumentParser


class TestStructuredParserMetadata(unittest.TestCase):
    def test_summary_not_in_metadata_and_metadata_excluded_from_split(self):
        content = """TITLE: Test Doc
SOURCE URL: https://example.com/really/long/path/that/should/not/bloat/metadata/for/chunking
INSTITUTION: Test Bank
PUBLICATION DATE: 2026-01-01
CATEGORY: bank_policy
---
SUMMARY (3-5 sentences relevance)
This summary should stay out of metadata used by splitter.
---
FULL CLEANED TEXT CONTENT
Main body text for retrieval.
"""
        with tempfile.TemporaryDirectory() as tmp_dir:
            file_path = Path(tmp_dir) / "doc.txt"
            file_path.write_text(content, encoding="utf-8")

            doc = StructuredDocumentParser.parse_file(file_path)
            self.assertIsNotNone(doc)

            # Prevent long summary text from inflating metadata.
            self.assertNotIn("summary", doc.metadata)

            # Metadata is excluded from splitter context to avoid chunk-size errors.
            self.assertEqual(doc.get_metadata_str(mode=MetadataMode.EMBED), "")
            self.assertEqual(doc.get_metadata_str(mode=MetadataMode.LLM), "")

            # Fingerprint should exist in both text and metadata to verify ingest lineage.
            self.assertIn(f"CLEANING_VERSION: {CLEANING_VERSION}", doc.text)
            self.assertEqual(doc.metadata.get("cleaning_version"), CLEANING_VERSION)


if __name__ == "__main__":
    unittest.main(verbosity=2)
