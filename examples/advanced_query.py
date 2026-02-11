#!/usr/bin/env python3
"""
Advanced query example for LlamaIndex
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.indexer import IndexManager
from src.query_engine import QueryEngineManager
from src.utils import setup_logging, validate_environment, format_response, measure_performance

def main():
    """Advanced query example"""
    # Setup logging
    setup_logging()
    
    # Validate environment
    if not validate_environment():
        print("Environment validation failed. Please check your configuration.")
        return
    
    print("🚀 Advanced Query Example")
    print("=" * 50)
    
    # Initialize index manager
    index_manager = IndexManager()
    
    # Load or create index
    index = index_manager.load_index()
    if index is None:
        print("No existing index found. Creating new index...")
        index = index_manager.create_index()
        if index is None:
            print("Failed to create index. Please check your documents.")
            return
    
    # Initialize query engine manager
    query_manager = QueryEngineManager(index)
    
    # Advanced query configurations
    configurations = [
        {
            "name": "High Precision (top_k=2)",
            "similarity_top_k": 2,
            "response_mode": "compact"
        },
        {
            "name": "Balanced (top_k=5)",
            "similarity_top_k": 5,
            "response_mode": "refine"
        },
        {
            "name": "High Recall (top_k=10)",
            "similarity_top_k": 10,
            "response_mode": "tree_summarize"
        }
    ]
    
    # Test query
    test_query = "What are the main applications of artificial intelligence in healthcare?"
    
    print(f"Testing query: '{test_query}'")
    print("=" * 50)
    
    for config in configurations:
        print(f"\nConfiguration: {config['name']}")
        print("-" * 40)
        
        try:
            # Measure performance
            @measure_performance
            def run_query():
                return query_manager.query(
                    test_query,
                    similarity_top_k=config['similarity_top_k'],
                    response_mode=config['response_mode']
                )
            
            result = run_query()
            
            print(f"Answer: {result['answer']}")
            print(f"Sources used: {len(result.get('sources', []))}")
            
            # Show source details
            if result.get('sources'):
                print("\nSource details:")
                for i, source in enumerate(result['sources'][:3], 1):  # Show top 3 sources
                    print(f"  {i}. Score: {source.get('score', 'N/A'):.4f}")
                    print(f"     Content: {source['content'][:100]}...")
                    print()
            
        except Exception as e:
            print(f"Error: {e}")
        
        print("=" * 50)
    
    # Query suggestions
    print("\n" + "=" * 50)
    print("Query Suggestions")
    print("=" * 50)
    
    topics = ["machine learning", "neural networks", "data science", "AI ethics"]
    
    for topic in topics:
        print(f"\nSuggestions for '{topic}':")
        suggestions = query_manager.get_query_suggestions(topic, num_suggestions=3)
        for i, suggestion in enumerate(suggestions, 1):
            print(f"  {i}. {suggestion}")
    
    # Comparison test
    print("\n" + "=" * 50)
    print("Response Mode Comparison")
    print("=" * 50)
    
    comparison_query = "Explain the difference between AI and machine learning"
    response_modes = ["compact", "refine", "tree_summarize"]
    
    print(f"Query: '{comparison_query}'")
    print()
    
    for mode in response_modes:
        print(f"Response Mode: {mode}")
        print("-" * 30)
        
        try:
            result = query_manager.query(
                comparison_query,
                response_mode=mode,
                similarity_top_k=3
            )
            print(f"Answer: {result['answer'][:200]}...")
            print()
        except Exception as e:
            print(f"Error: {e}")
    
    # Performance analysis
    print("\n" + "=" * 50)
    print("Performance Analysis")
    print("=" * 50)
    
    performance_queries = [
        "What is deep learning?",
        "How do neural networks work?",
        "What are the types of machine learning?"
    ]
    
    for query in performance_queries:
        print(f"\nQuery: '{query}'")
        
        try:
            @measure_performance
            def timed_query():
                return query_manager.query(query, similarity_top_k=5)
            
            result = timed_query()
            print(f"Answer length: {len(result['answer'])} characters")
            print(f"Sources used: {len(result.get('sources', []))}")
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    main()
