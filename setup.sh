#!/bin/bash

# LlamaIndex Project Setup Script
# This script sets up the environment and creates initial files

set -e  # Exit on any error

echo "🚀 Setting up LlamaIndex Project..."
echo "=================================="

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed. Please install Python 3.8 or higher."
    exit 1
fi

# Check Python version
python_version=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
required_version="3.8"

if [ "$(printf '%s\n' "$required_version" "$python_version" | sort -V | head -n1)" != "$required_version" ]; then
    echo "❌ Python version $python_version is too old. Please upgrade to Python 3.8 or higher."
    exit 1
fi

echo "✅ Python version $python_version detected"

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
    echo "✅ Virtual environment created"
else
    echo "✅ Virtual environment already exists"
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "⬆️  Upgrading pip..."
pip install --upgrade pip

# Install requirements
echo "📚 Installing requirements..."
if [ -f "Requirements.txt" ]; then
    pip install -r Requirements.txt
    echo "✅ Requirements installed"
else
    echo "❌ Requirements.txt not found"
    exit 1
fi

# Create necessary directories
echo "📁 Creating directories..."
mkdir -p data/documents
mkdir -p data/index
mkdir -p logs
echo "✅ Directories created"

# Create .env file if it doesn't exist
if [ ! -f ".env" ]; then
    echo "🔐 Creating .env file template..."
    cat > .env << EOF
# OpenAI API Configuration
OPENAI_API_KEY=your_openai_api_key_here

# Model Settings
MODEL_NAME=gpt-3.5-turbo
EMBEDDING_MODEL=text-embedding-ada-002

# Vector Store Settings
VECTOR_STORE_TYPE=chroma
# Options: chroma, faiss, simple

# Index Settings
CHUNK_SIZE=1024
CHUNK_OVERLAP=20

# Query Settings
SIMILARITY_TOP_K=4
RESPONSE_MODE=compact
# Options: compact, refine, tree_summarize

# Logging
LOG_LEVEL=INFO
# Options: DEBUG, INFO, WARNING, ERROR, CRITICAL
EOF
    echo "✅ .env file created"
    echo "⚠️  Please edit .env file and add your OpenAI API key"
else
    echo "✅ .env file already exists"
fi

# Create sample documents
echo "📄 Creating sample documents..."
python3 -c "
import sys
sys.path.append('.')
from src.utils import create_sample_documents
create_sample_documents()
print('✅ Sample documents created')
"

# Run tests to verify installation
echo "🧪 Running tests..."
python3 tests/test_query.py

# Create main indexer script
echo "📝 Creating main indexer script..."
cat > index_documents.py << 'EOF'
#!/usr/bin/env python3
"""
Main script to index documents
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.indexer import IndexManager
from src.utils import setup_logging, validate_environment, print_index_info

def main():
    """Main indexing function"""
    setup_logging()
    
    if not validate_environment():
        print("Environment validation failed. Please check your configuration.")
        return
    
    print("📚 Indexing Documents")
    print("=" * 50)
    
    index_manager = IndexManager()
    
    # Create new index
    index = index_manager.create_index()
    
    if index:
        # Show index statistics
        stats = index_manager.get_index_stats(index)
        print_index_info(stats)
        print("✅ Indexing completed successfully!")
    else:
        print("❌ Indexing failed!")

if __name__ == "__main__":
    main()
EOF

chmod +x index_documents.py
echo "✅ Main indexer script created"

# Create quick query script
echo "📝 Creating quick query script..."
cat > quick_query.py << 'EOF'
#!/usr/bin/env python3
"""
Quick query script for testing
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.indexer import IndexManager
from src.query_engine import QueryEngineManager
from src.utils import setup_logging, validate_environment

def main():
    """Quick query function"""
    setup_logging()
    
    if not validate_environment():
        print("Environment validation failed. Please check your configuration.")
        return
    
    if len(sys.argv) < 2:
        print("Usage: python quick_query.py 'your question here'")
        print("Example: python quick_query.py 'What is artificial intelligence?'")
        return
    
    query = " ".join(sys.argv[1:])
    
    print(f"🔍 Query: {query}")
    print("-" * 50)
    
    # Load index
    index_manager = IndexManager()
    index = index_manager.load_index()
    
    if not index:
        print("❌ No index found. Please run 'python index_documents.py' first.")
        return
    
    # Query
    query_manager = QueryEngineManager(index)
    result = query_manager.query(query)
    
    print(f"Answer: {result['answer']}")
    
    if result.get('sources'):
        print(f"\nSources ({len(result['sources'])}):")
        for i, source in enumerate(result['sources'], 1):
            print(f"  {i}. {source['content'][:100]}...")

if __name__ == "__main__":
    main()
EOF

chmod +x quick_query.py
echo "✅ Quick query script created"

echo ""
echo "🎉 Setup completed successfully!"
echo "=================================="
echo ""
echo "Next steps:"
echo "1. Edit .env file and add your OpenAI API key"
echo "2. Add your documents to data/documents/ directory"
echo "3. Run 'python index_documents.py' to create index"
echo "4. Run 'python quick_query.py \"your question\"' to test"
echo "5. Try examples: python examples/basic_query.py"
echo ""
echo "For more information, see QUICKSTART.md"
echo ""

# Deactivate virtual environment
deactivate
