#!/usr/bin/env python3
"""
Add docs to ChromaDB and run a retrieval query to verify the collection works.
Run: python scripts/test_query_collection.py
"""
# ChromaDB + NumPy 2.0 workaround
import numpy as np
if not hasattr(np, "float_"):
    np.float_ = np.float64  # type: ignore

import os
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
os.chdir(project_root)

DATA_DIR = project_root / "data" / "documents"
CHROMA_DIR = project_root / "storage" / "chroma"
COLLECTION = "credit_policies"


def main():
    # Ensure docs exist
    if not DATA_DIR.exists():
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        print(f"Created {DATA_DIR} - add .txt or .pdf files and run again.")
        return 1

    txt_files = list(DATA_DIR.glob("*.txt")) + list(DATA_DIR.glob("*.pdf"))
    if not txt_files:
        print(f"No .txt or .pdf files in {DATA_DIR}. Add documents first.")
        return 1

    print(f"Found {len(txt_files)} documents: {[f.name for f in txt_files]}")

    # Step 1: Build index (add docs to ChromaDB)
    print("\n--- Building index (ingesting into ChromaDB) ---")
    os.environ["DATA_DIR"] = str(DATA_DIR)
    os.environ["CHROMA_PERSIST_DIR"] = str(CHROMA_DIR)
    os.environ["VECTOR_STORE_TYPE"] = "chroma"
    os.environ["CHROMA_COLLECTION"] = COLLECTION

    from src.ingest import build_index
    build_index()

    # Step 2: Query retrieval (no LLM - just vector search)
    print("\n--- Querying collection ---")
    from llama_index.core import VectorStoreIndex
    from llama_index.vector_stores.chroma import ChromaVectorStore
    from llama_index.embeddings.huggingface import HuggingFaceEmbedding
    from llama_index.core.settings import Settings

    embed_model_name = os.getenv("EMBED_MODEL", "BAAI/bge-m3")
    embed_model = HuggingFaceEmbedding(
        model_name=embed_model_name,
        embed_batch_size=32,
    )
    Settings.embed_model = embed_model

    import chromadb
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    coll = client.get_collection(COLLECTION)
    print(f"Collection '{COLLECTION}' has {coll.count()} chunks")

    vector_store = ChromaVectorStore(chroma_collection=coll)
    index = VectorStoreIndex.from_vector_store(vector_store)
    retriever = index.as_retriever(similarity_top_k=3)

    queries = [
        "ต้องทำงานมานานเท่าไร",
        "ต้องมีคะแนนเครดิตเท่าไร"
    ]
    for q in queries:
        print(f"\nQ: {q}")
        nodes = retriever.retrieve(q)
        for i, n in enumerate(nodes, 1):
            text = n.get_content()[:150].replace("\n", " ")
            print(f"  [{i}] {text}...")

        # ChromaDB native query - raw result
        # query_embedding = embed_model.get_query_embedding(q)
        # result = coll.query(
        #     query_embeddings=[query_embedding],
        #     n_results=3,
        #     include=["documents", "metadatas", "distances"],
        # )
        # print("\n collection.query result =", result)

    print("\n Query from collection successful")
    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
