#!/usr/bin/env python3
"""
Debug index to see what's loaded
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.indexer import IndexManager
from src.query_engine import QueryEngineManager
from llama_index.core.retrievers import VectorIndexRetriever

def main():
    print("🔍 Debugging Index")
    print("=" * 60)
    
    # Load index
    index_manager = IndexManager()
    index = index_manager.load_index()
    
    if index is None:
        print("❌ No index found!")
        return
    
    print("✅ Index loaded")
    
    # Test retrieval
    print("\n1. Testing retrieval...")
    print("-" * 60)
    
    retriever = VectorIndexRetriever(index=index, similarity_top_k=5)
    
    test_queries = [
        "อัตราดอกเบี้ย",
        "สินเชื่อบ้าน",
        "interest rate",
        "home loan"
    ]
    
    for query in test_queries:
        print(f"\nQuery: {query}")
        nodes = retriever.retrieve(query)
        print(f"Retrieved {len(nodes)} nodes")
        
        if nodes:
            for i, node in enumerate(nodes[:2], 1):
                print(f"\n  Node {i}:")
                print(f"    Text preview: {node.text[:150]}...")
                print(f"    Score: {node.score if hasattr(node, 'score') else 'N/A'}")
                if hasattr(node, 'metadata'):
                    print(f"    Metadata: {node.metadata}")
        else:
            print("  ⚠️ No nodes retrieved!")
    
    # Test query engine
    print("\n2. Testing query engine...")
    print("-" * 60)
    
    query_manager = QueryEngineManager(index)
    
    result = query_manager.query("อัตราดอกเบี้ยเท่าไหร่")
    print(f"Answer: {result['answer']}")
    print(f"Response type: {type(result['response'])}")
    
    if hasattr(result['response'], '__dict__'):
        print(f"Response dict: {result['response'].__dict__}")

if __name__ == "__main__":
    main()
