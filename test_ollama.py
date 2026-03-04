#!/usr/bin/env python3
"""
Test Ollama connection
"""

import requests
from llama_index.llms.ollama import Ollama

def test_ollama_direct():
    """Test Ollama API directly"""
    print("Testing Ollama API directly...")
    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": "llama3:8b",
                "prompt": "Say hello in one sentence.",
                "stream": False
            },
            timeout=30
        )
        print(f"Status: {response.status_code}")
        if response.status_code == 200:
            print(f"Response: {response.json().get('response', 'No response')}")
        else:
            print(f"Error: {response.text}")
    except Exception as e:
        print(f"Error: {e}")

def test_ollama_llama_index():
    """Test Ollama via LlamaIndex"""
    print("\nTesting Ollama via LlamaIndex...")
    try:
        llm = Ollama(
            model="llama3:8b",
            base_url="http://localhost:11434",
            request_timeout=60.0
        )
        
        response = llm.complete("Say hello in one sentence.")
        print(f"Response: {response}")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_ollama_direct()
    test_ollama_llama_index()
