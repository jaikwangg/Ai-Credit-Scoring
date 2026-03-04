# Implementation Plan: Metadata Extraction and Ingestion

## Overview

This implementation modifies the RAG system's ingestion pipeline to extract and preserve structured metadata from documents. The core change replaces `SimpleDirectoryReader` with the existing `StructuredDocumentParser` in the ingestion pipeline, and updates query result formatting to display extracted metadata. The implementation leverages existing components and maintains backward compatibility with unstructured documents.

## Tasks

- [ ] 1. Modify ingestion pipeline to use StructuredDocumentParser
  - [x] 1.1 Replace SimpleDirectoryReader with StructuredDocumentParser in src/ingest.py
    - Import `StructuredDocumentParser` from `src.document_parser`
    - Replace `SimpleDirectoryReader(input_dir=DATA_DIR, recursive=True).load_data()` with `StructuredDocumentParser.parse_directory(Path(DATA_DIR))`
    - Add `from pathlib import Path` import
    - Verify the rest of the pipeline (SentenceSplitter, VectorStoreIndex) remains unchanged
    - _Requirements: 2.1, 2.2, 2.3, 2.4_

  - [ ]* 1.2 Write property test for metadata extraction completeness
    - **Property 1: Metadata Extraction Completeness**
    - **Validates: Requirements 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 2.3**
    - Generate random documents with all five metadata fields (title, category, source_url, institution, publication_date)
    - Verify all fields are correctly extracted into Document.metadata
    - Test with 100+ iterations using hypothesis

  - [ ]* 1.3 Write property test for recursive directory processing
    - **Property 2: Recursive Directory Processing**
    - **Validates: Requirements 2.4**
    - Generate random directory structures with .txt files at various depths
    - Verify all .txt files are discovered and parsed regardless of nesting level
    - Test with 100+ iterations using hypothesis

- [ ] 2. Update query result formatting to display metadata
  - [x] 2.1 Add metadata display helper function to src/query_engine.py
    - Create `format_source_display(metadata: dict) -> dict` function
    - Implement fallback logic: title defaults to file_name then "Unknown", category defaults to "Uncategorized"
    - Return formatted dict with title, category, source_url, institution, publication_date
    - _Requirements: 4.1, 4.2, 4.3, 4.4_

  - [x] 2.2 Integrate metadata formatting into QueryEngineManager.query() method
    - Update source_info construction to use `format_source_display()`
    - Ensure formatted metadata is included in the response
    - Maintain existing score and content fields
    - _Requirements: 4.1, 4.2_

  - [ ]* 2.3 Write property test for metadata display in query results
    - **Property 4: Metadata Display in Query Results**
    - **Validates: Requirements 4.1, 4.2**
    - Generate documents with random title and category values
    - Ingest into vector store and perform queries
    - Verify query results display actual metadata values (not "N/A")
    - Test with both ChromaDB and FAISS backends

- [x] 3. Checkpoint - Verify basic metadata flow
  - Ensure all tests pass, ask the user if questions arise.

- [ ] 4. Add comprehensive property tests for metadata persistence
  - [ ]* 4.1 Write property test for metadata persistence round-trip
    - **Property 3: Metadata Persistence Round-Trip**
    - **Validates: Requirements 3.1, 3.2, 3.3, 3.4**
    - Generate documents with random metadata
    - Ingest into vector store and retrieve
    - Verify metadata survives round-trip with identical values
    - Test with both ChromaDB and FAISS backends

  - [ ]* 4.2 Write property test for unstructured document processing
    - **Property 5: Unstructured Document Processing**
    - **Validates: Requirements 5.1, 5.4**
    - Generate documents without metadata headers (plain text)
    - Verify parser creates Document objects with full text preserved
    - Verify metadata fields are empty but document is not rejected
    - Test with 100+ iterations using hypothesis

  - [ ]* 4.3 Write property test for partial failure isolation
    - **Property 6: Partial Failure Isolation**
    - **Validates: Requirements 5.2, 5.3**
    - Generate mix of valid and invalid documents (some with parse errors)
    - Verify all parseable documents are successfully ingested
    - Verify individual failures don't prevent other documents from processing
    - Test with 100+ iterations using hypothesis

  - [ ]* 4.4 Write property test for enhanced text format structure
    - **Property 7: Enhanced Text Format Structure**
    - **Validates: Requirements 6.1, 6.2, 6.3, 6.4**
    - Generate documents with random metadata values
    - Verify Document.text contains metadata preamble followed by content
    - Verify Document.metadata dict contains same fields separately accessible
    - Verify metadata appears at beginning of enhanced text
    - Test with 100+ iterations using hypothesis

- [ ] 5. Add unit tests for edge cases and error handling
  - [ ]* 5.1 Write unit tests for parser edge cases
    - Test empty metadata values (headers present but empty)
    - Test missing metadata headers (partial metadata)
    - Test malformed headers (unusual spacing, formatting)
    - Test special characters in metadata (unicode, newlines)
    - Test empty files (zero-byte, whitespace-only)
    - _Requirements: 5.1, 5.4_

  - [ ]* 5.2 Write unit tests for error handling
    - Test file read errors (permission denied, file not found)
    - Test encoding errors (invalid UTF-8)
    - Test parser error logging behavior
    - Verify errors are logged and processing continues
    - _Requirements: 5.2, 5.3_

  - [ ]* 5.3 Write integration tests for end-to-end metadata flow
    - Create test documents with known metadata
    - Run full ingestion pipeline
    - Query the index
    - Verify metadata appears correctly in results
    - Test with both ChromaDB and FAISS backends
    - _Requirements: 1.1, 3.1, 4.1_

  - [ ]* 5.4 Write backward compatibility tests
    - Ingest mix of structured and unstructured documents
    - Verify all documents are processed successfully
    - Verify structured documents show actual metadata
    - Verify unstructured documents show fallback values
    - _Requirements: 5.1, 5.4, 4.3, 4.4_

- [x] 6. Final checkpoint - Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

## Notes

- Tasks marked with `*` are optional and can be skipped for faster MVP
- The existing `StructuredDocumentParser` in `src/document_parser.py` already implements all required metadata extraction functionality - no changes needed
- Property tests require `hypothesis>=6.0.0` to be added to requirements.txt
- All property tests should run minimum 100 iterations for comprehensive coverage
- Metadata automatically flows through LlamaIndex's pipeline - no custom serialization needed
- Enhanced text format improves retrieval quality by including metadata context in embeddings
- Implementation maintains backward compatibility with documents lacking structured metadata
