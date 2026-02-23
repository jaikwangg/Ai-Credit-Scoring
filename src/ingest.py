import os

import numpy as _np
if not hasattr(_np, "float_"):
    _np.float_ = _np.float64  # type: ignore

from llama_index.core import SimpleDirectoryReader, StorageContext, VectorStoreIndex
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.settings import Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore

try:
    from .settings import (
        CHROMA_COLLECTION,
        CHROMA_PERSIST_DIR,
        DATA_DIR,
        EMBED_MODEL,
        INDEX_DIR,
        OLLAMA_BASE_URL,
        OLLAMA_MODEL,
        VECTOR_STORE_TYPE,
    )
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


def _get_storage_context() -> StorageContext:
    if VECTOR_STORE_TYPE == "chroma":
        import chromadb
        os.makedirs(CHROMA_PERSIST_DIR, exist_ok=True)
        client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
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


def build_index() -> None:
    os.makedirs(INDEX_DIR, exist_ok=True)
    print(f"Using Ollama config: model={OLLAMA_MODEL}, base_url={OLLAMA_BASE_URL}")
    print(f"Vector store: {VECTOR_STORE_TYPE}")
    print(f"Embedding model: {EMBED_MODEL}")

    docs = SimpleDirectoryReader(input_dir=DATA_DIR, recursive=True).load_data()

    splitter = SentenceSplitter(chunk_size=512, chunk_overlap=80)
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


if __name__ == "__main__":
    build_index()
