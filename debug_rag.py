#!/usr/bin/env python3
"""
Debug RAG pipeline
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.indexer import IndexManager
from src.query_engine import QueryEngineManager

def main():
    print("🔍 Debugging RAG Pipeline")
    print("=" * 50)
    
    # Load index
    index_manager = IndexManager()
    index = index_manager.load_index()
    
    if index is None:
        print("No index found!")
        return
    
    print("Index loaded successfully!")
    
    # Test retrieval
    print("\n1. Testing retrieval...")
    print("-" * 50)
    
    from llama_index.core.retrievers import VectorIndexRetriever
    retriever = VectorIndexRetriever(index=index, similarity_top_k=3)
    
    query = "คะแนนเครดิต"
    print(f"Query: {query}")
    
    nodes = retriever.retrieve(query)
    print(f"Retrieved {len(nodes)} nodes")
    
    for i, node in enumerate(nodes, 1):
        print(f"\nNode {i}:")
        print(f"  Text: {node.text[:200]}...")
        print(f"  Score: {node.score if hasattr(node, 'score') else 'N/A'}")
    
    # Test LLM directly with context
    print("\n2. Testing LLM with context...")
    print("-" * 50)
    
    query_manager = QueryEngineManager(index)
    
    context = nodes[0].text if nodes else "No context"
    prompt = f"""Based on this context:
{context}

Question: คะแนนเครดิตขั้นต่ำคือเท่าไหร่

Answer in Thai:"""
    
    print(f"Prompt:\n{prompt}\n")
    
    response = query_manager.llm.complete(prompt)
    print(f"LLM Response: {response}")
    
    # Test full query engine
    print("\n3. Testing full query engine...")
    print("-" * 50)
    
    result = query_manager.query("คะแนนเครดิตขั้นต่ำคือเท่าไหร่")
    print(f"Answer: {result['answer']}")
    print(f"Response object: {result['response']}")
    print(f"Response type: {type(result['response'])}")
    
    # Check if response has any attributes
    if hasattr(result['response'], '__dict__'):
        print(f"Response attributes: {result['response'].__dict__}")

if __name__ == "__main__":
    main()
