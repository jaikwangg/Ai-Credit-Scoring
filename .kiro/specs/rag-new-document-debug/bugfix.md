# Bugfix Requirements Document

## Introduction

The RAG system fails to retrieve and answer questions from newly ingested documents due to a configuration mismatch between the ingestion module (`src/ingest.py`) and the query modules (`src/indexer.py`, `src/query_engine.py`). Documents are successfully ingested into ChromaDB but stored in a different location and collection than where the query system looks for them, resulting in 0 nodes retrieved for all queries.

## Bug Analysis

### Current Behavior (Defect)

1.1 WHEN new documents are ingested using `python -m src.ingest` THEN the system stores vectors in ChromaDB at `./storage/chroma` with collection name `credit_policies` (from `src/settings.py`)

1.2 WHEN queries are executed using `debug_index.py`, `test_cimb_loans.py`, or any module using `src/indexer.py` THEN the system looks for vectors in ChromaDB at `data/index/chroma` with collection name `documents` (from `config/settings.py`)

1.3 WHEN the query system loads the index from a different ChromaDB path/collection than where documents were ingested THEN retrieval returns 0 nodes for all queries

1.4 WHEN retrieval returns 0 nodes THEN the query engine returns "Empty Response" for all questions, even though documents were successfully ingested

### Expected Behavior (Correct)

2.1 WHEN new documents are ingested using `python -m src.ingest` THEN the system SHALL store vectors in the same ChromaDB location and collection that the query system uses

2.2 WHEN queries are executed after ingestion THEN the system SHALL retrieve relevant nodes from the same ChromaDB location and collection where documents were stored

2.3 WHEN both ingestion and query modules access ChromaDB THEN they SHALL use a single, consistent configuration source for `CHROMA_PERSIST_DIR` and `CHROMA_COLLECTION`

2.4 WHEN documents are successfully ingested THEN subsequent queries SHALL retrieve relevant nodes and return meaningful answers based on the ingested content

### Unchanged Behavior (Regression Prevention)

3.1 WHEN the system uses ChromaDB as the vector store THEN it SHALL CONTINUE TO persist vectors to disk for reuse across sessions

3.2 WHEN documents are ingested with the BGE-M3 embedding model THEN queries SHALL CONTINUE TO use the same BGE-M3 model for encoding query embeddings

3.3 WHEN the ingestion process completes successfully THEN it SHALL CONTINUE TO display confirmation messages indicating the storage location and collection name

3.4 WHEN the system supports multiple vector store types (chroma, faiss, simple) THEN it SHALL CONTINUE TO support all configured vector store types

3.5 WHEN environment variables are set in `.env` THEN the system SHALL CONTINUE TO respect those configuration overrides
