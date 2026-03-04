#!/usr/bin/env python3
"""
Simple test for Thai credit policy RAG
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.indexer import IndexManager
from src.query_engine import QueryEngineManager

def main():
    print("🔍 Testing Thai Credit Policy RAG")
    print("=" * 50)
    
    # Initialize index manager
    index_manager = IndexManager()
    
    # Load index
    index = index_manager.load_index()
    if index is None:
        print("Creating new index...")
        index = index_manager.create_index()
    
    print("Index loaded successfully!")
    print()
    
    # Initialize query engine
    query_manager = QueryEngineManager(index)
    
    # Thai queries about credit policy
    queries = [
        "อัตราส่วนหนี้ต่อรายได้ต้องไม่เกินเท่าไหร่",
        "คะแนนเครดิตขั้นต่ำคือเท่าไหร่",
        "ต้องใช้เอกสารอะไรบ้างในการสมัคร",
        "ถ้าชำระล่าช้าจะเกิดอะไรขึ้น"
    ]
    
    print("Testing Thai queries:")
    print("-" * 50)
    
    for i, query in enumerate(queries, 1):
        print(f"\nQuery {i}: {query}")
        print("-" * 30)
        
        try:
            result = query_manager.query(query)
            print(f"Answer: {result['answer']}")
            
            if result.get('sources'):
                print(f"\nSources ({len(result['sources'])}):")
                for j, source in enumerate(result['sources'][:2], 1):
                    print(f"  {j}. {source['content'][:100]}...")
                    if source.get('score'):
                        print(f"     Score: {source['score']:.4f}")
        except Exception as e:
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()
        
        print()

if __name__ == "__main__":
    main()
