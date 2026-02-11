#!/usr/bin/env python3
"""
Chat engine example for LlamaIndex
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.indexer import IndexManager
from src.query_engine import QueryEngineManager
from src.utils import setup_logging, validate_environment, save_chat_history, load_chat_history

def main():
    """Chat engine example"""
    # Setup logging
    setup_logging()
    
    # Validate environment
    if not validate_environment():
        print("Environment validation failed. Please check your configuration.")
        return
    
    print("💬 Chat Engine Example")
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
    
    # Chat modes to test
    chat_modes = ["condense_question", "simple"]
    
    print("Testing different chat modes:")
    print("=" * 50)
    
    for mode in chat_modes:
        print(f"\nTesting chat mode: {mode}")
        print("-" * 40)
        
        # Create chat engine
        chat_engine = query_manager.create_chat_engine(chat_mode=mode)
        
        # Test conversation
        test_conversation = [
            "Hello! Can you help me learn about artificial intelligence?",
            "What are the main types of machine learning?",
            "Can you explain deep learning in more detail?",
            "How is AI used in healthcare?",
            "Thank you for the information!"
        ]
        
        conversation_history = []
        
        for message in test_conversation:
            print(f"User: {message}")
            
            try:
                response = chat_engine.chat(message)
                print(f"Assistant: {response}")
                
                # Store conversation
                conversation_history.append({
                    "user": message,
                    "assistant": str(response),
                    "mode": mode
                })
                
            except Exception as e:
                print(f"Error: {e}")
            
            print()
        
        # Save conversation
        filename = f"chat_history_{mode}.json"
        save_chat_history(conversation_history, filename)
        print(f"Conversation saved to {filename}")
        
        print("=" * 50)
    
    # Interactive chat session
    print("\n" + "=" * 50)
    print("Interactive Chat Session")
    print("=" * 50)
    print("Choose chat mode:")
    print("1. condense_question (recommended for document-based chat)")
    print("2. simple (basic chat without document context)")
    
    while True:
        choice = input("\nEnter your choice (1 or 2, or 'quit' to exit): ").strip()
        
        if choice.lower() == 'quit':
            break
        elif choice == '1':
            chat_mode = "condense_question"
            break
        elif choice == '2':
            chat_mode = "simple"
            break
        else:
            print("Invalid choice. Please try again.")
    
    if choice.lower() != 'quit':
        print(f"\nStarting interactive chat with mode: {chat_mode}")
        print("Type 'quit' or 'exit' to end the session")
        print("Type 'save' to save the conversation")
        print("Type 'clear' to clear the conversation history")
        print("-" * 50)
        
        # Create chat engine
        chat_engine = query_manager.create_chat_engine(chat_mode=chat_mode)
        conversation_history = []
        
        while True:
            user_input = input("\nYou: ").strip()
            
            if user_input.lower() in ['quit', 'exit']:
                print("Goodbye!")
                break
            elif user_input.lower() == 'save':
                filename = f"interactive_chat_{chat_mode}_{len(conversation_history)}_messages.json"
                save_chat_history(conversation_history, filename)
                print(f"Conversation saved to {filename}")
                continue
            elif user_input.lower() == 'clear':
                conversation_history = []
                print("Conversation history cleared.")
                continue
            elif not user_input:
                continue
            
            try:
                response = chat_engine.chat(user_input)
                print(f"Assistant: {response}")
                
                # Store in history
                conversation_history.append({
                    "user": user_input,
                    "assistant": str(response),
                    "timestamp": str(Path(__file__).stat().st_mtime)
                })
                
            except Exception as e:
                print(f"Error: {e}")
        
        # Auto-save conversation
        if conversation_history:
            filename = f"final_chat_{chat_mode}_{len(conversation_history)}_messages.json"
            save_chat_history(conversation_history, filename)
            print(f"Final conversation saved to {filename}")

def demo_contextual_chat():
    """Demonstrate contextual chat capabilities"""
    print("\n" + "=" * 50)
    print("Contextual Chat Demonstration")
    print("=" * 50)
    
    # Initialize components
    index_manager = IndexManager()
    index = index_manager.load_index() or index_manager.create_index()
    query_manager = QueryEngineManager(index)
    
    # Create contextual chat engine
    chat_engine = query_manager.create_chat_engine(
        chat_mode="condense_question",
        similarity_top_k=5
    )
    
    # Demonstrate context retention
    contextual_demo = [
        "What is artificial intelligence?",
        "How does it relate to machine learning?",
        "Can you give me some specific examples?",
        "What are the ethical concerns with these examples?",
        "How can we address these concerns?"
    ]
    
    print("Demonstrating context retention across multiple messages:")
    print("-" * 50)
    
    for i, message in enumerate(contextual_demo, 1):
        print(f"\nMessage {i}: {message}")
        print("-" * 30)
        
        try:
            response = chat_engine.chat(message)
            print(f"Response: {response}")
        except Exception as e:
            print(f"Error: {e}")
    
    print("\n" + "=" * 50)
    print("Notice how the assistant maintains context across messages!")
    print("=" * 50)

if __name__ == "__main__":
    main()
    
    # Optional: Run contextual demo
    run_demo = input("\nWould you like to see the contextual chat demo? (y/n): ").strip().lower()
    if run_demo == 'y':
        demo_contextual_chat()
