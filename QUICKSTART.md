# LlamaIndex Project - Quick Start Guide

## 🚀 การเริ่มต้นใช้งาน

ยินดีต้อนรับสู่โปรเจค LlamaIndex! คู่มือนี้จะช่วยคุณเริ่มต้นใช้งานระบบค้นหาเอกสารอัจฉริยะด้วย LlamaIndex

## 📋 ข้อกำหนดเบื้องต้น

- Python 3.8 ขึ้นไป
- OpenAI API Key
- เอกสารที่ต้องการค้นหา (PDF, TXT, DOCX, XLSX, CSV)

## 🔧 การติดตั้ง

### 1. ใช้ Setup Script (แนะนำ)

```bash
# รัน setup script
chmod +x setup.sh
./setup.sh
```

Script นี้จะ:
- ✅ สร้าง virtual environment
- ✅ ติดตั้ง dependencies ทั้งหมด
- ✅ สร้างโครงสร้างโฟลเดอร์
- ✅ สร้างไฟล์ .env template
- ✅ สร้างเอกสารตัวอย่าง
- ✅ สร้าง scripts หลัก

### 2. ติดตั้งแบบ Manual

```bash
# สร้าง virtual environment
python3 -m venv venv
source venv/bin/activate

# ติดตั้ง dependencies
pip install -r Requirements.txt

# สร้างโฟลเดอร์
mkdir -p data/documents data/index logs
```

## ⚙️ การตั้งค่า

### 1. ตั้งค่า OpenAI API Key

แก้ไขไฟล์ `.env`:

```bash
# เปิดไฟล์ .env
nano .env
```

เพิ่ม API key ของคุณ:

```env
OPENAI_API_KEY=sk-your-openai-api-key-here
MODEL_NAME=gpt-3.5-turbo
EMBEDDING_MODEL=text-embedding-ada-002
```

### 2. การตั้งค่าอื่นๆ (Optional)

```env
# Vector Store Type
VECTOR_STORE_TYPE=chroma  # หรือ faiss, simple

# Index Settings
CHUNK_SIZE=1024
CHUNK_OVERLAP=20

# Query Settings
SIMILARITY_TOP_K=4
RESPONSE_MODE=compact
```

## 📚 การเตรียมเอกสาร

### 1. วางเอกสารในโฟลเดอร์

```bash
# วางเอกสารของคุณที่นี่
data/documents/
├── document1.pdf
├── document2.txt
├── report.docx
└── data.xlsx
```

### 2. รูปแบบที่รองรับ

- **PDF** (.pdf) - ใช้ PyPDF2
- **Text** (.txt) - ข้อความธรรมดา
- **Word** (.docx) - ใช้ python-docx
- **Excel** (.xlsx, .csv) - ใช้ openpyxl, pandas

## 🚀 การใช้งาน

### 1. สร้าง Index

```bash
# Activate virtual environment
source venv/bin/activate

# สร้าง index จากเอกสารทั้งหมด
python index_documents.py
```

### 2. ทดสอบการค้นหา

```bash
# ค้นหาแบบรวดเร็ว
python quick_query.py "What is artificial intelligence?"

# ตัวอย่างผลลัพธ์:
# Answer: Artificial Intelligence is a branch of computer science...
# Sources:
#   1. Artificial Intelligence (AI) is a branch of computer science...
```

### 3. ใช้งานแบบ Interactive

```bash
# Basic query example
python examples/basic_query.py

# Advanced query with different configurations
python examples/advanced_query.py

# Chat engine
python examples/chat_engine.py
```

## 🎯 ตัวอย่างการใช้งาน

### Basic Query

```python
from src.indexer import IndexManager
from src.query_engine import QueryEngineManager

# Load index
index_manager = IndexManager()
index = index_manager.load_index()

# Create query engine
query_manager = QueryEngineManager(index)

# Query
result = query_manager.query("What are the applications of AI?")
print(result['answer'])
```

### Chat Engine

```python
# Create chat engine
chat_engine = query_manager.create_chat_engine(chat_mode="condense_question")

# Chat with context
response = chat_engine.chat("Tell me more about machine learning")
print(response)
```

## 📊 การปรับแต่ง

### 1. Vector Store Options

```env
# Chroma (แนะนำ) - สำหรับ production
VECTOR_STORE_TYPE=chroma

# FAISS - เร็วกว่า แต่ใช้ RAM มาก
VECTOR_STORE_TYPE=faiss

# Simple - ง่าย แต่ไม่ persist
VECTOR_STORE_TYPE=simple
```

### 2. Query Configurations

```python
# High precision
result = query_manager.query(
    question="What is AI?",
    similarity_top_k=2,
    response_mode="compact"
)

# High recall
result = query_manager.query(
    question="What is AI?",
    similarity_top_k=10,
    response_mode="tree_summarize"
)
```

### 3. Index Settings

```python
# ใน config/settings.py
CHUNK_SIZE = 512        # ขนาด chunk ที่เล็กลง
CHUNK_OVERLAP = 50     #  overlap ที่มากขึ้น
SIMILARITY_TOP_K = 8    # ค้นหาเอกสารมากขึ้น
```

## 🔍 การค้นหาขั้นสูง

### 1. ค้นหาด้วย Sources

```python
result = query_manager.query(
    "Explain neural networks",
    include_sources=True
)

# ดู sources
for source in result['sources']:
    print(f"Source: {source['content'][:100]}...")
    print(f"Score: {source['score']}")
    print(f"Metadata: {source['metadata']}")
```

### 2. Query Suggestions

```python
suggestions = query_manager.get_query_suggestions("machine learning")
print(suggestions)
# ['What are the types of machine learning?', 'How do neural networks work?', ...]
```

### 3. Performance Analysis

```python
from src.utils import measure_performance

@measure_performance
def run_complex_query():
    return query_manager.query(complex_question)

result = run_complex_query()
```

## 🛠️ การดูแลระบบ

### 1. ตรวจสอบ Index Stats

```python
stats = index_manager.get_index_stats(index)
print(f"Total documents: {stats['total_docs']}")
print(f"Vector store: {stats['vector_store_type']}")
```

### 2. Rebuild Index

```python
# สร้าง index ใหม่ทั้งหมด
index = index_manager.rebuild_index()
```

### 3. Logging

```bash
# ดู logs
tail -f llama_index.log
```

## 🧪 การทดสอบ

```bash
# รัน tests ทั้งหมด
python tests/test_query.py

# รันด้วย unittest
python -m pytest tests/ -v
```

## ❓ คำถามที่พบบ่อย

### Q: OpenAI API key ใส่ไว้ที่ไหน?
A: ในไฟล์ `.env` ใน variable `OPENAI_API_KEY`

### Q: ใส่เอกสารได้กี่ไฟล์?
A: ขึ้นอยู่กับ RAM แต่แนะนำไม่เกิน 1000 ไฟล์สำหรับเริ่มต้น

### Q: ทำไม index สร้างนาน?
A: ขึ้นอยู่กับขนาดเอกสารและประเภท vector store

### Q: เปลี่ยน model ได้ไหม?
A: ได้ ใน `.env` เปลี่ยน `MODEL_NAME` เช่น `gpt-4`

### Q: ใช้งาน offline ได้ไหม?
A: ไม่ได้ ต้องมีการเชื่อมต่อ internet สำหรับ OpenAI API

## 🐛 การแก้ไขปัญหา

### 1. Environment Issues

```bash
# ตรวจสอบ Python version
python3 --version

# ตรวจสอบ virtual environment
which python
```

### 2. API Issues

```bash
# ทดสอบ API key
python -c "import openai; print('API key works' if openai.api_key else 'No API key')"
```

### 3. Index Issues

```bash
# ล้าง index และสร้างใหม่
rm -rf data/index/*
python index_documents.py
```

## 📚 แหล่งข้อมูลเพิ่มเติม

- [LlamaIndex Documentation](https://docs.llamaindex.ai/)
- [OpenAI API Documentation](https://platform.openai.com/docs)
- [Project Examples](examples/)

## 🤝 การสนับสนุน

หากพบปัญหา:
1. ตรวจสอบ logs ใน `llama_index.log`
2. รัน tests ด้วย `python tests/test_query.py`
3. ตรวจสอบ configuration ใน `.env`

---

🎉 **เริ่มต้นสร้างระบบค้นหาเอกสารอัจฉริยะของคุณวันนี้!**

สำหรับคำถามเพิ่มเติม ดูได้ที่:
- `examples/` - ตัวอย่างการใช้งาน
- `tests/` - การทดสอบระบบ
- `src/` - ซอร์สโค้ดหลัก
