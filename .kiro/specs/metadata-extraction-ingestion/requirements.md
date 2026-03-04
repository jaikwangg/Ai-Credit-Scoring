# Requirements Document

## Introduction

The RAG system currently displays "Source: N/A" and "Category: N/A" when showing query results because the document ingestion process uses SimpleDirectoryReader, which ignores structured metadata headers present in the documents. This feature will modify the ingestion pipeline to extract and preserve metadata (title, category, source URL, institution, publication date) from structured documents, enabling the query system to display accurate source information to users.

## Glossary

- **Ingestion_Pipeline**: The process in `src/ingest.py` that loads documents from the filesystem and creates vector embeddings for the RAG system
- **StructuredDocumentParser**: The existing parser class in `src/document_parser.py` that extracts metadata from documents with structured headers
- **SimpleDirectoryReader**: The LlamaIndex component currently used in ingestion that loads document text but ignores metadata headers
- **Document_Metadata**: Structured information extracted from document headers including title, category, source_url, institution, and publication_date
- **Vector_Store**: The storage system (ChromaDB or FAISS) that persists document embeddings and associated metadata
- **Query_Result**: The response object returned to users containing the answer and source information

## Requirements

### Requirement 1: Extract Metadata During Ingestion

**User Story:** As a system administrator, I want the ingestion pipeline to extract structured metadata from documents, so that source information is available for query results.

#### Acceptance Criteria

1. WHEN the Ingestion_Pipeline processes a document with structured headers, THE Ingestion_Pipeline SHALL extract title, category, source_url, institution, and publication_date metadata fields
2. WHEN a document contains a TITLE header field, THE Ingestion_Pipeline SHALL store the title value in Document_Metadata
3. WHEN a document contains a CATEGORY header field, THE Ingestion_Pipeline SHALL store the category value in Document_Metadata
4. WHEN a document contains a SOURCE URL header field, THE Ingestion_Pipeline SHALL store the source_url value in Document_Metadata
5. WHEN a document contains an INSTITUTION header field, THE Ingestion_Pipeline SHALL store the institution value in Document_Metadata
6. WHEN a document contains a PUBLICATION DATE header field, THE Ingestion_Pipeline SHALL store the publication_date value in Document_Metadata

### Requirement 2: Use StructuredDocumentParser for Document Loading

**User Story:** As a developer, I want the ingestion pipeline to use StructuredDocumentParser instead of SimpleDirectoryReader, so that metadata extraction is performed automatically.

#### Acceptance Criteria

1. THE Ingestion_Pipeline SHALL use StructuredDocumentParser to load documents from the data directory
2. THE Ingestion_Pipeline SHALL NOT use SimpleDirectoryReader for loading structured documents
3. WHEN StructuredDocumentParser loads a document, THE Ingestion_Pipeline SHALL receive a Document object with populated metadata fields
4. THE Ingestion_Pipeline SHALL process all .txt files in the data/documents directory recursively

### Requirement 3: Persist Metadata in Vector Store

**User Story:** As a system operator, I want document metadata to be stored in the vector store, so that it remains available after ingestion completes.

#### Acceptance Criteria

1. WHEN the Ingestion_Pipeline creates vector embeddings, THE Vector_Store SHALL persist the Document_Metadata alongside the embeddings
2. WHEN using ChromaDB as the Vector_Store, THE Ingestion_Pipeline SHALL store metadata in the ChromaDB collection
3. WHEN using FAISS as the Vector_Store, THE Ingestion_Pipeline SHALL store metadata in the FAISS index storage context
4. FOR ALL stored documents, retrieving the document from the Vector_Store SHALL return the associated Document_Metadata

### Requirement 4: Display Metadata in Query Results

**User Story:** As an end user, I want to see the actual document title and category in query results, so that I can identify the source of information.

#### Acceptance Criteria

1. WHEN a query returns results, THE Query_Result SHALL include the title from Document_Metadata instead of "N/A"
2. WHEN a query returns results, THE Query_Result SHALL include the category from Document_Metadata instead of "N/A"
3. IF a document has no title metadata, THEN THE Query_Result SHALL display the file_name as a fallback
4. IF a document has no category metadata, THEN THE Query_Result SHALL display "Uncategorized" as a fallback

### Requirement 5: Maintain Backward Compatibility

**User Story:** As a system maintainer, I want the ingestion pipeline to handle documents without structured metadata, so that the system remains robust.

#### Acceptance Criteria

1. WHEN StructuredDocumentParser encounters a document without metadata headers, THE Ingestion_Pipeline SHALL process the document using the full text content
2. IF metadata extraction fails for a document, THEN THE Ingestion_Pipeline SHALL log a warning and continue processing remaining documents
3. THE Ingestion_Pipeline SHALL NOT fail completely if individual documents cannot be parsed
4. WHEN a document lacks structured format, THE StructuredDocumentParser SHALL create a Document object with empty metadata fields and the raw text content

### Requirement 6: Preserve Enhanced Text Format

**User Story:** As a RAG system operator, I want document text to include metadata context, so that retrieval quality is improved.

#### Acceptance Criteria

1. WHEN StructuredDocumentParser creates a Document object, THE Document object SHALL include an enhanced text format containing title, category, institution, summary, and main content
2. THE enhanced text format SHALL place metadata fields at the beginning of the document text
3. THE enhanced text format SHALL maintain the original content structure after the metadata preamble
4. FOR ALL documents, the enhanced text SHALL be used for embedding generation while metadata fields remain separately accessible

