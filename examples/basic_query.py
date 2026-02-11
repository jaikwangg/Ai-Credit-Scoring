#!/usr/bin/env python3
"""
Basic query example for LlamaIndex
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.indexer import IndexManager
from src.query_engine import QueryEngineManager
from src.utils import setup_logging, validate_environment, format_response

def main():
    """Basic query example"""
    # Setup logging
    setup_logging()
    
    # Validate environment
    if not validate_environment():
        print("Environment validation failed. Please check your configuration.")
        return
    
    print("🔍 Basic Query Example")
    print("=" * 50)
    
    # Initialize index manager
    index_manager = IndexManager()
    
    # Try to load existing index, or create new one
    index = index_manager.load_index()
    if index is None:
        print("No existing index found. Creating new index...")
        index = index_manager.create_index()
        if index is None:
            print("Failed to create index. Please check your documents.")
            return
    
    # Get index statistics
    stats = index_manager.get_index_stats(index)
    print(f"Index loaded: {stats['total_docs']} documents")
    print()
    
    # Initialize query engine manager
    query_manager = QueryEngineManager(index)
    
    # Example queries
    queries = [
        "What is artificial intelligence?",
        "How does machine learning work?",
        "What are the applications of AI?",
        "Explain natural language processing"
    ]
    
    print("Running example queries:")
    print("-" * 50)
    
    for i, query in enumerate(queries, 1):
        print(f"\nQuery {i}: {query}")
        print("-" * 30)
        
        try:
            result = query_manager.query(query)
            print(format_response(result))
        except Exception as e:
            print(f"Error: {e}")
        
        print("\n" + "=" * 50)
    
    # Interactive mode
    print("\nWould you like to try interactive mode? (y/n): ", end="")
    choice = input().strip().lower()
    
    if choice == 'y':
        from src.utils import interactive_query_loop
        interactive_query_loop(query_manager)

if __name__ == "__main__":
    main()
