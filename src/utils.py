"""
Utility functions for LlamaIndex project
"""

import logging
import os
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional
import json
import time

from config.settings import settings

logger = logging.getLogger(__name__)

def setup_logging(log_level: str = None) -> None:
    """
    Setup logging configuration
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    """
    log_level = log_level or settings.LOG_LEVEL
    
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('llama_index.log')
        ]
    )

def validate_environment() -> bool:
    """
    Validate that the environment is properly configured
    
    Returns:
        True if environment is valid, False otherwise
    """
    try:
        # Validate settings
        settings.validate()
        
        # Check directories
        settings.ensure_directories()
        
        # Check OpenAI API key
        if not settings.OPENAI_API_KEY:
            logger.error("OPENAI_API_KEY not found in environment variables")
            return False
        
        logger.info("Environment validation passed")
        return True
    except Exception as e:
        logger.error(f"Environment validation failed: {e}")
        return False

def format_response(response: Dict[str, Any]) -> str:
    """
    Format query response for display
    
    Args:
        response: Response dictionary
        
    Returns:
        Formatted response string
    """
    output = []
    
    # Add answer
    if "answer" in response:
        output.append(f"**Answer:** {response['answer']}")
        output.append("")
    
    # Add sources if available
    if "sources" in response and response["sources"]:
        output.append("**Sources:**")
        for i, source in enumerate(response["sources"], 1):
            output.append(f"{i}. {source['content']}")
            if source.get("metadata"):
                metadata_str = ", ".join([f"{k}: {v}" for k, v in source["metadata"].items()])
                output.append(f"   Metadata: {metadata_str}")
            if source.get("score"):
                output.append(f"   Score: {source['score']:.4f}")
            output.append("")
    
    return "\n".join(output)

def save_chat_history(history: List[Dict[str, str]], filename: str = "chat_history.json") -> None:
    """
    Save chat history to file
    
    Args:
        history: Chat history list
        filename: Output filename
    """
    try:
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(history, f, ensure_ascii=False, indent=2)
        logger.info(f"Chat history saved to {filename}")
    except Exception as e:
        logger.error(f"Error saving chat history: {e}")

def load_chat_history(filename: str = "chat_history.json") -> List[Dict[str, str]]:
    """
    Load chat history from file
    
    Args:
        filename: Input filename
        
    Returns:
        Chat history list
    """
    try:
        if Path(filename).exists():
            with open(filename, 'r', encoding='utf-8') as f:
                history = json.load(f)
            logger.info(f"Chat history loaded from {filename}")
            return history
        return []
    except Exception as e:
        logger.error(f"Error loading chat history: {e}")
        return []

def measure_performance(func):
    """
    Decorator to measure function performance
    
    Args:
        func: Function to measure
        
    Returns:
        Wrapped function
    """
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        
        logger.info(f"{func.__name__} executed in {end_time - start_time:.2f} seconds")
        return result
    
    return wrapper

def get_document_summary(documents: List) -> Dict[str, Any]:
    """
    Get summary statistics about documents
    
    Args:
        documents: List of Document objects
        
    Returns:
        Dictionary with document statistics
    """
    if not documents:
        return {"total_documents": 0}
    
    total_chars = sum(len(doc.text) for doc in documents)
    total_words = sum(len(doc.text.split()) for doc in documents)
    
    file_types = {}
    for doc in documents:
        if hasattr(doc, 'file_path'):
            ext = Path(doc.file_path).suffix.lower()
            file_types[ext] = file_types.get(ext, 0) + 1
    
    return {
        "total_documents": len(documents),
        "total_characters": total_chars,
        "total_words": total_words,
        "avg_chars_per_doc": total_chars / len(documents),
        "avg_words_per_doc": total_words / len(documents),
        "file_types": file_types
    }

def create_sample_documents() -> None:
    """
    Create sample documents for testing
    """
    documents_dir = settings.DOCUMENTS_DIR
    
    # Sample text document
    sample_text = """
    Artificial Intelligence and Machine Learning

    Artificial Intelligence (AI) is a branch of computer science that aims to create intelligent machines 
    that can simulate human thinking capability and behavior. AI encompasses various subfields including 
    machine learning, natural language processing, computer vision, and robotics.

    Machine Learning (ML) is a subset of AI that enables systems to learn and improve from experience 
    without being explicitly programmed. ML algorithms build mathematical models based on training data 
    to make predictions or decisions.

    Deep Learning is a subset of machine learning that uses neural networks with multiple layers to 
    progressively extract higher-level features from raw input. Deep learning has revolutionized fields 
    like computer vision, speech recognition, and natural language processing.

    Applications of AI and ML include:
    - Healthcare: Disease diagnosis, drug discovery, personalized medicine
    - Finance: Fraud detection, algorithmic trading, risk assessment
    - Transportation: Autonomous vehicles, traffic optimization
    - Retail: Recommendation systems, inventory management
    - Customer service: Chatbots, virtual assistants
    """
    
    with open(documents_dir / "ai_overview.txt", "w", encoding="utf-8") as f:
        f.write(sample_text)
    
    # Another sample document
    sample_text2 = """
    Natural Language Processing

    Natural Language Processing (NLP) is a branch of artificial intelligence that helps computers 
    understand, interpret and manipulate human language. NLP draws from many disciplines, including 
    computer science and computational linguistics, in its pursuit to fill the gap between human 
    communication and computer understanding.

    Key NLP tasks include:
    - Text classification: Categorizing text into predefined categories
    - Named Entity Recognition: Identifying entities like names, organizations, locations
    - Sentiment Analysis: Determining emotional tone of text
    - Machine Translation: Translating text from one language to another
    - Question Answering: Answering questions based on text context
    - Text Summarization: Creating concise summaries of longer texts

    Modern NLP heavily relies on transformer models like BERT, GPT, and T5, which have achieved 
    state-of-the-art performance across various NLP tasks.
    """
    
    with open(documents_dir / "nlp_basics.txt", "w", encoding="utf-8") as f:
        f.write(sample_text2)
    
    logger.info(f"Sample documents created in {documents_dir}")

def print_index_info(index_stats: Dict[str, Any]) -> None:
    """
    Print index information in a formatted way
    
    Args:
        index_stats: Index statistics dictionary
    """
    print("\n" + "="*50)
    print("INDEX INFORMATION")
    print("="*50)
    
    for key, value in index_stats.items():
        print(f"{key.replace('_', ' ').title()}: {value}")
    
    print("="*50)

def interactive_query_loop(query_engine_manager) -> None:
    """
    Interactive query loop for testing
    
    Args:
        query_engine_manager: QueryEngineManager instance
    """
    print("\n" + "="*50)
    print("INTERACTIVE QUERY MODE")
    print("="*50)
    print("Type 'quit' or 'exit' to end the session")
    print("Type 'help' for available commands")
    print("="*50)
    
    while True:
        user_input = input("\nEnter your query: ").strip()
        
        if user_input.lower() in ['quit', 'exit']:
            print("Goodbye!")
            break
        elif user_input.lower() == 'help':
            print("\nAvailable commands:")
            print("- Type any question to query the documents")
            print("- 'stats' - Show index statistics")
            print("- 'suggestions <topic>' - Get query suggestions")
            print("- 'quit' or 'exit' - End session")
            continue
        elif user_input.lower() == 'stats':
            # You would need to pass the index to get stats
            print("Index statistics not available in this mode")
            continue
        elif user_input.lower().startswith('suggestions'):
            topic = user_input[12:].strip()
            if topic:
                suggestions = query_engine_manager.get_query_suggestions(topic)
                print(f"\nQuery suggestions for '{topic}':")
                for i, suggestion in enumerate(suggestions, 1):
                    print(f"{i}. {suggestion}")
            else:
                print("Please provide a topic for suggestions")
            continue
        
        # Process query
        try:
            result = query_engine_manager.query(user_input)
            print("\n" + format_response(result))
        except Exception as e:
            print(f"Error processing query: {e}")
