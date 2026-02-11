import os
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, StorageContext
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.settings import Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.faiss import FaissVectorStore

import faiss

from .settings import DATA_DIR, INDEX_DIR, EMBED_MODEL

def build_index():
    os.makedirs(INDEX_DIR, exist_ok=True)

    # 1) Load docs
    docs = SimpleDirectoryReader(
        input_dir=DATA_DIR,
        recursive=True
    ).load_data()

    # 2) Chunker
    splitter = SentenceSplitter(chunk_size=512, chunk_overlap=80)

    # 3) Embeddings
    Settings.embed_model = HuggingFaceEmbedding(model_name=EMBED_MODEL)
    Settings.node_parser = splitter

    # 4) Vector store (FAISS)
    dim = Settings.embed_model.get_text_embedding_dimension()
    faiss_index = faiss.IndexFlatIP(dim)  # cosine-ish if embeddings normalized (bge usually ok)
    vector_store = FaissVectorStore(faiss_index=faiss_index)

    storage_context = StorageContext.from_defaults(
        vector_store=vector_store
    )

    # 5) Build index
    index = VectorStoreIndex.from_documents(
        docs,
        storage_context=storage_context
    )

    # 6) Persist
    index.storage_context.persist(persist_dir=INDEX_DIR)
    print(f"✅ Index built & saved to: {INDEX_DIR}")

if __name__ == "__main__":
    build_index()
