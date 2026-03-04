import os
from pathlib import Path

import numpy as _np
if not hasattr(_np, "float_"):
    _np.float_ = _np.float64  # type: ignore

from llama_index.core import StorageContext, VectorStoreIndex
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.settings import Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore

from src.document_parser import CLEANING_VERSION, StructuredDocumentParser

try:
    from config.settings import settings
    CHROMA_COLLECTION = settings.CHROMA_COLLECTION
    CHROMA_PERSIST_DIR = settings.CHROMA_PERSIST_DIR
    DATA_DIR = str(settings.DOCUMENTS_DIR)
    EMBED_MODEL = settings.EMBEDDING_MODEL
    INDEX_DIR = str(settings.INDEX_DIR)
    CHUNK_SIZE = settings.CHUNK_SIZE
    CHUNK_OVERLAP = settings.CHUNK_OVERLAP
    RESET_CHROMA_COLLECTION_ON_INGEST = settings.RESET_CHROMA_COLLECTION_ON_INGEST
    OLLAMA_BASE_URL = settings.OLLAMA_BASE_URL
    OLLAMA_MODEL = settings.OLLAMA_MODEL
    VECTOR_STORE_TYPE = settings.VECTOR_STORE_TYPE
except ImportError:  # pragma: no cover - script execution fallback
    from src.settings import (
        CHROMA_COLLECTION,
        CHROMA_PERSIST_DIR,
        DATA_DIR,
        EMBED_MODEL,
        INDEX_DIR,
        OLLAMA_BASE_URL,
        OLLAMA_MODEL,
        VECTOR_STORE_TYPE,
    )
    CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "512"))
    CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "50"))
    RESET_CHROMA_COLLECTION_ON_INGEST = (
        os.getenv("RESET_CHROMA_COLLECTION_ON_INGEST", "true").lower() == "true"
    )


def _get_storage_context() -> StorageContext:
    if VECTOR_STORE_TYPE == "chroma":
        import chromadb
        os.makedirs(CHROMA_PERSIST_DIR, exist_ok=True)
        client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)

        if RESET_CHROMA_COLLECTION_ON_INGEST:
            try:
                client.delete_collection(CHROMA_COLLECTION)
                print(f"Deleted existing Chroma collection: {CHROMA_COLLECTION}")
            except Exception:
                pass

        collection = client.get_or_create_collection(CHROMA_COLLECTION)
        vector_store = ChromaVectorStore(chroma_collection=collection)
        return StorageContext.from_defaults(vector_store=vector_store)

    # Default to FAISS for backward compatibility when not using Chroma.
    from llama_index.vector_stores.faiss import FaissVectorStore
    import faiss
    os.makedirs(INDEX_DIR, exist_ok=True)
    dim = Settings.embed_model.get_text_embedding_dimension()
    faiss_index = faiss.IndexFlatIP(dim)
    vector_store = FaissVectorStore(faiss_index=faiss_index)
    return StorageContext.from_defaults(vector_store=vector_store)


def _verify_cleaning_fingerprint(index: VectorStoreIndex) -> bool:
    try:
        retriever = index.as_retriever(similarity_top_k=5)
        nodes = retriever.retrieve("CLEANING_VERSION")
    except Exception as exc:
        print(f"WARNING: Fingerprint sanity query failed: {exc}")
        return False

    fingerprint_line = f"CLEANING_VERSION: {CLEANING_VERSION}"
    for node in nodes or []:
        text = getattr(node, "text", None)
        if not isinstance(text, str):
            inner = getattr(node, "node", None)
            text = getattr(inner, "text", "") if inner is not None else ""
        metadata = getattr(node, "metadata", None)
        if not isinstance(metadata, dict):
            inner = getattr(node, "node", None)
            metadata = getattr(inner, "metadata", {}) if inner is not None else {}

        if fingerprint_line in str(text) or str(metadata.get("cleaning_version", "")) == CLEANING_VERSION:
            return True
    return False


def build_index() -> None:
    os.makedirs(INDEX_DIR, exist_ok=True)
    print(f"Using Ollama config: model={OLLAMA_MODEL}, base_url={OLLAMA_BASE_URL}")
    print(f"Vector store: {VECTOR_STORE_TYPE}")
    print(f"Embedding model: {EMBED_MODEL}")
    print(
        "Reset Chroma collection on ingest: "
        f"{RESET_CHROMA_COLLECTION_ON_INGEST}"
    )

    docs = StructuredDocumentParser.parse_directory(Path(DATA_DIR))
    if not docs:
        raise RuntimeError(f"No documents found in {DATA_DIR}")

    parse_report = StructuredDocumentParser.get_last_parse_report()
    total_docs = int(parse_report.get("total_docs", len(docs)))
    indexed_docs = int(parse_report.get("indexed_docs", len(docs)))
    quarantined_docs = int(parse_report.get("quarantined_docs", 0))

    print(f"Documents parsed (total): {total_docs}")
    print(f"Documents indexed: {indexed_docs}")
    print(f"Documents quarantined: {quarantined_docs}")

    quarantined_examples = parse_report.get("quarantined_examples", [])
    if isinstance(quarantined_examples, list) and quarantined_examples:
        print("Top quarantined documents:")
        for item in quarantined_examples[:10]:
            if not isinstance(item, dict):
                continue
            title = str(item.get("title", "unknown"))
            reason = str(item.get("reason", "unknown"))
            print(f"  - {title} | reason={reason}")

    splitter = SentenceSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    Settings.embed_model = HuggingFaceEmbedding(
        model_name=EMBED_MODEL,
        embed_batch_size=32,
    )
    Settings.node_parser = splitter

    storage_context = _get_storage_context()

    index = VectorStoreIndex.from_documents(
        docs,
        storage_context=storage_context,
    )

    # Keep a local persisted context for non-Chroma stores.
    if VECTOR_STORE_TYPE != "chroma":
        index.storage_context.persist(persist_dir=INDEX_DIR)
        print(f"Index built and saved to: {INDEX_DIR}")
    else:
        print(
            f"Index built and stored in ChromaDB at: {CHROMA_PERSIST_DIR} "
            f"(collection: {CHROMA_COLLECTION})"
        )

    fingerprint_found = _verify_cleaning_fingerprint(index)
    if fingerprint_found:
        print(f"Fingerprint sanity check passed: CLEANING_VERSION={CLEANING_VERSION}")
    else:
        print(
            "WARNING: Parser changes not reflected in index. "
            "Check DATA_DIR / parser pipeline."
        )


if __name__ == "__main__":
    build_index()
