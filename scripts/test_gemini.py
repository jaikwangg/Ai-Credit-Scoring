import sys, io, os
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config.settings import settings
from src.query_engine import _build_llm

print(f"Provider : {'Gemini' if settings.USE_GEMINI else 'Ollama'}")
print(f"Model    : {settings.GEMINI_MODEL if settings.USE_GEMINI else settings.OLLAMA_MODEL}")

llm = _build_llm()
resp = llm.complete("สวัสดี ตอบสั้นๆ ว่าพร้อมใช้งานไหม")
print(f"Response : {str(resp)}")
