# Implementation Plan

- [x] 1. Write bug condition exploration test
  - **Property 1: Fault Condition** - Configuration Mismatch Detection
  - **CRITICAL**: This test MUST FAIL on unfixed code - failure confirms the bug exists
  - **DO NOT attempt to fix the test or the code when it fails**
  - **NOTE**: This test encodes the expected behavior - it will validate the fix when it passes after implementation
  - **GOAL**: Surface counterexamples that demonstrate the configuration mismatch
  - **Scoped PBT Approach**: Scope the property to the concrete failing case - ingestion and query modules using different ChromaDB configurations
  - Test that ingestion module (src/ingest.py) and query modules (src/indexer.py, src/query_engine.py) use different CHROMA_PERSIST_DIR and/or CHROMA_COLLECTION values
  - Test end-to-end: ingest a document with known content, query for that content, assert retrieval returns >0 nodes
  - Run test on UNFIXED code
  - **EXPECTED OUTCOME**: Test FAILS (configuration mismatch detected, 0 nodes retrieved)
  - Document counterexamples found (e.g., "ingestion uses ./storage/chroma/credit_policies, query uses data/index/chroma/documents, result: 0 nodes")
  - Mark task complete when test is written, run, and failure is documented
  - _Requirements: 2.1, 2.2, 2.3, 2.4_

- [x] 2. Write preservation property tests (BEFORE implementing fix)
  - **Property 2: Preservation** - Non-Configuration Operations Unchanged
  - **IMPORTANT**: Follow observation-first methodology
  - Observe behavior on UNFIXED code for operations not involving ChromaDB path/collection configuration
  - Observe: Document parsing produces consistent chunks and metadata
  - Observe: BGE-M3 embedding model generates embeddings correctly
  - Observe: Environment variables (non-ChromaDB settings like OLLAMA_MODEL) override defaults
  - Observe: Alternative vector stores (faiss, simple) work if configured
  - Write property-based tests capturing observed behavior patterns from Preservation Requirements
  - Property-based testing generates many test cases for stronger guarantees
  - Run tests on UNFIXED code
  - **EXPECTED OUTCOME**: Tests PASS (confirms baseline behavior to preserve)
  - Mark task complete when tests are written, run, and passing on unfixed code
  - _Requirements: 3.1, 3.2, 3.3, 3.4, 3.5_

- [x] 3. Fix for ChromaDB configuration mismatch

  - [x] 3.1 Consolidate to single configuration source
    - Choose config/settings.py as the single source of truth (recommended for better structure)
    - Update src/ingest.py to import from config/settings.py instead of src/settings.py
    - Update src/indexer.py to ensure it imports from config/settings.py
    - Update src/query_engine.py to ensure it imports from config/settings.py
    - Set consistent defaults in config/settings.py (recommend CHROMA_PERSIST_DIR="./storage/chroma", CHROMA_COLLECTION="credit_policies")
    - Add comments to src/settings.py indicating config/settings.py is the canonical source for ChromaDB settings
    - Verify environment variables CHROMA_PERSIST_DIR and CHROMA_COLLECTION properly override defaults
    - _Bug_Condition: isBugCondition(operation) where ingestion_chroma_dir != query_chroma_dir OR ingestion_collection != query_collection_
    - _Expected_Behavior: All modules use same CHROMA_PERSIST_DIR and CHROMA_COLLECTION from single configuration source_
    - _Preservation: ChromaDB persistence, BGE-M3 embedding model, environment variable overrides, multi-vector-store support, document parsing, query processing_
    - _Requirements: 2.1, 2.2, 2.3, 2.4, 3.1, 3.2, 3.3, 3.4, 3.5_

  - [x] 3.2 Verify bug condition exploration test now passes
    - **Property 1: Expected Behavior** - Unified ChromaDB Configuration
    - **IMPORTANT**: Re-run the SAME test from task 1 - do NOT write a new test
    - The test from task 1 encodes the expected behavior
    - When this test passes, it confirms the expected behavior is satisfied
    - Run bug condition exploration test from step 1
    - **EXPECTED OUTCOME**: Test PASSES (configuration is now unified, documents are retrievable)
    - _Requirements: 2.1, 2.2, 2.3, 2.4_

  - [x] 3.3 Verify preservation tests still pass
    - **Property 2: Preservation** - Existing Functionality Unchanged
    - **IMPORTANT**: Re-run the SAME tests from task 2 - do NOT write new tests
    - Run preservation property tests from step 2
    - **EXPECTED OUTCOME**: Tests PASS (confirms no regressions)
    - Confirm all tests still pass after fix (document parsing, embedding model, environment variables, alternative vector stores)

- [x] 4. Checkpoint - Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.
