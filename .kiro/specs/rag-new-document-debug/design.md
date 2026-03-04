# RAG New Document Debug Bugfix Design

## Overview

The RAG system has a configuration mismatch causing newly ingested documents to be inaccessible during queries. The ingestion module (`src/ingest.py`) uses `src/settings.py` which stores vectors in `./storage/chroma` with collection `credit_policies`, while the query modules (`src/indexer.py`, `src/query_engine.py`) use `config/settings.py` which looks for vectors in `data/index/chroma` with collection `documents`. This results in 0 nodes retrieved for all queries despite successful ingestion.

The fix will unify the configuration by making all modules use a single source of truth for ChromaDB settings, ensuring ingestion and query operations access the same storage location and collection.

## Glossary

- **Bug_Condition (C)**: The condition where ingestion and query modules use different ChromaDB configurations (different persist directories or collection names)
- **Property (P)**: The desired behavior where both ingestion and query modules use the same ChromaDB configuration, enabling successful retrieval of ingested documents
- **Preservation**: Existing functionality that must remain unchanged - ChromaDB persistence, BGE-M3 embedding model usage, environment variable overrides, multi-vector-store support
- **src/settings.py**: Configuration module used by `src/ingest.py` - currently specifies `./storage/chroma` and collection `credit_policies`
- **config/settings.py**: Configuration module used by `src/indexer.py` and `src/query_engine.py` - currently specifies `data/index/chroma` and collection `documents`
- **CHROMA_PERSIST_DIR**: Environment variable/setting that specifies the filesystem path where ChromaDB persists vector data
- **CHROMA_COLLECTION**: Environment variable/setting that specifies the ChromaDB collection name for storing/retrieving vectors

## Bug Details

### Fault Condition

The bug manifests when the ingestion module and query modules use different configuration sources, resulting in mismatched ChromaDB persist directories and/or collection names. The system successfully ingests documents but stores them in a location that the query system never checks.

**Formal Specification:**
```
FUNCTION isBugCondition(operation)
  INPUT: operation of type {ingestion_config, query_config}
  OUTPUT: boolean
  
  LET ingestion_chroma_dir = src/settings.py::CHROMA_PERSIST_DIR
  LET ingestion_collection = src/settings.py::CHROMA_COLLECTION
  LET query_chroma_dir = config/settings.py::CHROMA_PERSIST_DIR
  LET query_collection = config/settings.py::CHROMA_COLLECTION
  
  RETURN (ingestion_chroma_dir != query_chroma_dir) 
         OR (ingestion_collection != query_collection)
END FUNCTION
```

### Examples

- **Ingestion stores at**: `./storage/chroma` with collection `credit_policies`
  **Query looks at**: `data/index/chroma` with collection `documents`
  **Result**: 0 nodes retrieved, "Empty Response" for all queries

- **After ingesting loan policy PDF**: Document successfully processed and stored in `./storage/chroma/credit_policies`
  **When querying "What are the loan requirements?"**: System searches `data/index/chroma/documents`, finds nothing, returns "Empty Response"

- **Environment variable override**: User sets `CHROMA_PERSIST_DIR=./my_storage` in `.env`
  **Expected**: Both ingestion and query use `./my_storage`
  **Actual (buggy)**: Only ingestion respects it if using `src/settings.py`, query still uses hardcoded path from `config/settings.py`

- **Edge case - both configs accidentally match**: If both configuration files happen to specify the same paths, the bug doesn't manifest (but the dual configuration source remains a latent issue)

## Expected Behavior

### Preservation Requirements

**Unchanged Behaviors:**
- ChromaDB must continue to persist vectors to disk for reuse across sessions
- BGE-M3 embedding model must continue to be used for both ingestion and query operations
- Ingestion process must continue to display confirmation messages with storage location and collection name
- System must continue to support multiple vector store types (chroma, faiss, simple) as configured
- Environment variables in `.env` must continue to override default configuration values

**Scope:**
All operations that do NOT involve ChromaDB configuration path/collection settings should be completely unaffected by this fix. This includes:
- Document parsing and chunking logic
- Embedding model selection and usage
- Query processing and response generation
- LLM model configuration (Ollama settings)
- Logging and error handling
- Other vector store types (faiss, simple)

## Hypothesized Root Cause

Based on the bug description and code analysis, the root cause is:

1. **Duplicate Configuration Sources**: The codebase has two separate configuration files (`src/settings.py` and `config/settings.py`) that both define ChromaDB settings independently
   - `src/ingest.py` imports from `src/settings.py`
   - `src/indexer.py` and `src/query_engine.py` import from `config/settings.py`
   - No shared configuration module exists

2. **Hardcoded Default Values**: Each configuration file has different hardcoded defaults
   - `src/settings.py`: `CHROMA_PERSIST_DIR = "./storage/chroma"`, `CHROMA_COLLECTION = "credit_policies"`
   - `config/settings.py`: `CHROMA_PERSIST_DIR = str(INDEX_DIR / "chroma")` (resolves to `data/index/chroma`), `CHROMA_COLLECTION = "documents"`

3. **Inconsistent Environment Variable Usage**: While both files read from environment variables, if those variables are not set, they fall back to different defaults, causing the mismatch

4. **No Validation**: There is no startup validation that checks whether ingestion and query configurations are aligned

## Correctness Properties

Property 1: Fault Condition - Unified ChromaDB Configuration

_For any_ operation (ingestion or query) that accesses ChromaDB, the system SHALL use the same CHROMA_PERSIST_DIR and CHROMA_COLLECTION values from a single configuration source, ensuring that documents ingested are stored in the same location where queries will retrieve them.

**Validates: Requirements 2.1, 2.2, 2.3, 2.4**

Property 2: Preservation - Existing Functionality Unchanged

_For any_ operation that does NOT involve ChromaDB path/collection configuration (document parsing, embedding model usage, LLM queries, environment variable overrides, alternative vector stores), the fixed code SHALL produce exactly the same behavior as the original code, preserving all existing functionality.

**Validates: Requirements 3.1, 3.2, 3.3, 3.4, 3.5**

## Fix Implementation

### Changes Required

Assuming our root cause analysis is correct:

**Primary Approach: Consolidate to Single Configuration Source**

**File**: `src/ingest.py`, `src/indexer.py`, `src/query_engine.py`

**Specific Changes**:
1. **Standardize Configuration Import**: Modify all modules to import from a single configuration source
   - Option A: Make all modules import from `config/settings.py` (recommended - more structured)
   - Option B: Make all modules import from `src/settings.py`
   - Ensure consistent import statements across all affected files

2. **Update Configuration Defaults**: In the chosen configuration file, set consistent defaults
   - Decide on standard location (recommend `./storage/chroma` for clarity)
   - Decide on standard collection name (recommend `credit_policies` or `documents`)
   - Update hardcoded defaults to match

3. **Remove Duplicate Configuration**: Deprecate or remove ChromaDB settings from the non-chosen configuration file
   - Add comments indicating the canonical configuration location
   - Or remove the duplicate settings entirely to prevent future confusion

4. **Verify Environment Variable Handling**: Ensure the chosen configuration file properly reads and respects environment variables
   - `CHROMA_PERSIST_DIR` should override the default
   - `CHROMA_COLLECTION` should override the default
   - Test that `.env` file values are correctly loaded

5. **Add Configuration Validation** (optional but recommended): Add a validation function that logs the active ChromaDB configuration at startup
   - Log the persist directory and collection name being used
   - This helps with debugging and confirms the fix is working

## Testing Strategy

### Validation Approach

The testing strategy follows a two-phase approach: first, surface counterexamples that demonstrate the bug on unfixed code by showing the configuration mismatch, then verify the fix works correctly by confirming unified configuration and successful retrieval.

### Exploratory Fault Condition Checking

**Goal**: Surface counterexamples that demonstrate the bug BEFORE implementing the fix. Confirm that ingestion and query modules use different ChromaDB configurations, leading to 0 nodes retrieved.

**Test Plan**: Write tests that inspect the configuration values used by ingestion and query modules, then attempt to ingest a document and query it. Run these tests on the UNFIXED code to observe the configuration mismatch and retrieval failure.

**Test Cases**:
1. **Configuration Mismatch Test**: Import settings from both `src/settings.py` and `config/settings.py`, assert that `CHROMA_PERSIST_DIR` and/or `CHROMA_COLLECTION` differ (will fail on unfixed code showing the mismatch)
2. **Ingestion Storage Location Test**: Ingest a test document, verify it's stored in `./storage/chroma/credit_policies` (will succeed on unfixed code)
3. **Query Retrieval Location Test**: Attempt to query the ingested document, observe that query system looks in `data/index/chroma/documents` (will fail to find documents on unfixed code)
4. **End-to-End Retrieval Test**: Ingest a document with known content, query for that content, assert retrieval returns >0 nodes (will fail on unfixed code with 0 nodes)

**Expected Counterexamples**:
- Configuration values differ between ingestion and query modules
- Documents successfully ingested but queries return 0 nodes
- Possible causes: duplicate configuration files, different hardcoded defaults, inconsistent imports

### Fix Checking

**Goal**: Verify that for all operations where the bug condition previously held (ingestion and query using different configs), the fixed system uses unified configuration and successfully retrieves ingested documents.

**Pseudocode:**
```
FOR ALL operation IN {ingest, query} DO
  config := getChromaConfig(operation)
  ASSERT config.persist_dir = UNIFIED_PERSIST_DIR
  ASSERT config.collection = UNIFIED_COLLECTION
END FOR

// End-to-end test
ingestDocument(test_doc)
results := queryEngine.query("content from test_doc")
ASSERT results.nodes.length > 0
ASSERT results.response != "Empty Response"
```

### Preservation Checking

**Goal**: Verify that for all operations where the bug condition does NOT hold (operations not involving ChromaDB path/collection configuration), the fixed system produces the same result as the original system.

**Pseudocode:**
```
FOR ALL operation WHERE NOT isChromaConfigOperation(operation) DO
  ASSERT fixedSystem(operation) = originalSystem(operation)
END FOR
```

**Testing Approach**: Property-based testing is recommended for preservation checking because:
- It generates many test cases automatically across the input domain
- It catches edge cases that manual unit tests might miss
- It provides strong guarantees that behavior is unchanged for all non-config operations

**Test Plan**: Observe behavior on UNFIXED code first for document parsing, embedding generation, and query processing (excluding retrieval), then write property-based tests capturing that behavior.

**Test Cases**:
1. **Document Parsing Preservation**: Observe that document parsing produces the same chunks and metadata on unfixed code, verify this continues after fix
2. **Embedding Model Preservation**: Observe that BGE-M3 embeddings are generated correctly on unfixed code, verify this continues after fix
3. **Environment Variable Preservation**: Observe that `.env` overrides work on unfixed code, verify this continues after fix (test with non-ChromaDB settings like `OLLAMA_MODEL`)
4. **Alternative Vector Store Preservation**: If system supports faiss or simple vector stores, verify those continue to work unchanged

### Unit Tests

- Test that all modules import configuration from the same source
- Test that ChromaDB persist directory is consistent across ingestion and query
- Test that ChromaDB collection name is consistent across ingestion and query
- Test that environment variables override default configuration values
- Test edge case where `.env` is missing (should use consistent defaults)

### Property-Based Tests

- Generate random document content, ingest it, verify queries can retrieve it (tests unified configuration)
- Generate random environment variable combinations, verify both ingestion and query respect them consistently
- Generate random query strings after ingestion, verify all return >0 nodes (not "Empty Response")

### Integration Tests

- Test full ingestion-to-query flow: ingest document, query it, verify meaningful response
- Test configuration validation: start system, verify logged configuration shows unified settings
- Test environment variable override: set `CHROMA_PERSIST_DIR` and `CHROMA_COLLECTION` in `.env`, verify both ingestion and query use those values
- Test persistence across sessions: ingest documents, restart system, verify queries still work (tests that persist directory is truly unified)
