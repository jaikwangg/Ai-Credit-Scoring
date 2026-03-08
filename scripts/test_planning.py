"""
Test script for LLM-synthesized planning output.
Initializes LLM (Gemini/Ollama) then runs generate_response with mock RAG.

Usage:
    uv run python scripts/test_planning.py 2>/dev/null
"""
import sys, io, os, json
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from llama_index.core.settings import Settings
from config.settings import settings as cfg
from src.query_engine import _build_llm
from src.planner.planning import NO_ANSWER_SENTINEL, generate_response

# --- Init LLM (same as query_engine does at startup) ---
print(f"Provider : {'Gemini' if cfg.USE_GEMINI else 'Ollama'}")
print(f"Model    : {cfg.GEMINI_MODEL if cfg.USE_GEMINI else cfg.OLLAMA_MODEL}")
Settings.llm = _build_llm()
print("LLM ready.\n")

# --- Mock RAG (same as mock_demo.py) ---
_KB = {
    "เอกสารที่ต้องใช้สมัครสินเชื่อบ้านมีอะไรบ้าง": ("ใช้บัตรประชาชน ทะเบียนบ้าน และเอกสารแสดงรายได้ตามประเภทอาชีพ", "โฮมโลนฟอร์ยู | สินเชื่อบ้าน | CIMB TH", "policy_requirement", 0.91),
    "ต้องมีคุณสมบัติอย่างไรถึงจะกู้บ้านได้": ("ต้องมีสัญชาติไทยและมีรายได้สม่ำเสมอตามเกณฑ์", "โฮมโลนฟอร์ยู | สินเชื่อบ้าน | CIMB TH", "policy_requirement", 0.89),
    "รายได้ขั้นต่ำเท่าไหร่ถึงจะกู้บ้านได้": ("รายได้ขั้นต่ำเป็นไปตามเงื่อนไขผลิตภัณฑ์และประเภทผู้กู้", "โฮมโลนฟอร์ยู | สินเชื่อบ้าน | CIMB TH", "policy_requirement", 0.84),
    "ผ่อนไม่ไหวต้องทำอย่างไร ปรับโครงสร้างหนี้": ("สามารถยื่นคำขอปรับโครงสร้างหนี้และขยายงวดผ่อนภายใต้เงื่อนไขสถาบันการเงิน", "ใบคำขอปรับปรุงโครงสร้างหนี้ (ขยายระยะเวลาผ่อน)", "hardship_support", 0.87),
    "ขอขยายระยะเวลาผ่อนได้ไหม": ("สามารถขอขยายระยะเวลาผ่อนได้ โดยพิจารณาตามความสามารถชำระ", "ใบคำขอปรับปรุงโครงสร้างหนี้ (ขยายระยะเวลาผ่อน)", "hardship_support", 0.86),
    "มีมาตรการช่วยเหลือลูกหนี้อะไรบ้าง": ("มีมาตรการช่วยเหลือลูกหนี้เป็นระยะตามประกาศของสถาบันการเงิน", "มาตรการช่วยเหลือลูกหนี้ระยะที่ 2", "hardship_support", 0.83),
    "อัตราดอกเบี้ยสินเชื่อบ้านเท่าไหร่": ("อัตราดอกเบี้ยขึ้นกับแผนสินเชื่อและช่วงเวลาโปรโมชัน", "อัตราดอกเบี้ยสินเชื่อบ้านใหม่ (Generic) ปี 2568/2569", "interest_structure", 0.88),
    "มี fixed rate หรือ floating rate บ้าง": ("มีทั้ง fixed rate และ floating rate ตามแผนสินเชื่อ", "loan-interest-rates-th.txt", "interest_structure", 0.81),
    "เครดิตบูโรสำคัญอย่างไร": (NO_ANSWER_SENTINEL, "", "", 0.0),
}

def mock_rag(query: str) -> dict:
    item = _KB.get(query)
    if not item:
        return {"answer": NO_ANSWER_SENTINEL, "sources": []}
    answer, title, category, score = item
    if answer == NO_ANSWER_SENTINEL:
        return {"answer": answer, "sources": []}
    return {"answer": answer, "sources": [{"title": title, "category": category, "score": score}]}

# --- Load test fixtures ---
base = os.path.join(os.path.dirname(__file__), "..", "examples", "planner")
def load(name): return json.load(open(os.path.join(base, name), encoding="utf-8"))

user = load("user.json")
shap = load("shap.json")
model_rejected = load("model_rejected.json")
model_approved = load("model_approved.json")

# --- Run ---
print("=" * 60)
print("REJECTED CASE — LLM synthesis")
print("=" * 60)
result = generate_response(user, model_rejected, shap, rag_lookup=mock_rag)
print(result["result_th"])

print("\n" + "=" * 60)
print("APPROVED CASE — LLM synthesis")
print("=" * 60)
result2 = generate_response(user, model_approved, shap, rag_lookup=mock_rag)
print(result2["result_th"])
