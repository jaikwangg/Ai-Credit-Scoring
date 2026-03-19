"""
Test [IsSup] groundedness check for planning.
Usage:
    uv run python scripts/test_issup_plan.py
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from llama_index.core.settings import Settings
from config.settings import settings as cfg
from src.query_engine import _build_llm
from src.planner.planning import NO_ANSWER_SENTINEL, generate_response
import time

print(f"Provider : {'Gemini' if cfg.USE_GEMINI else 'Ollama'}", flush=True)
Settings.llm = _build_llm()
print("LLM ready.\n", flush=True)

_KB = {
    "เอกสารที่ต้องใช้สมัครสินเชื่อบ้านมีอะไรบ้าง": ("ใช้บัตรประชาชน ทะเบียนบ้าน และเอกสารแสดงรายได้ตามประเภทอาชีพ", "โฮมโลนฟอร์ยู | สินเชื่อบ้าน | CIMB TH", "policy_requirement", 0.91),
    "ต้องมีคุณสมบัติอย่างไรถึงจะกู้บ้านได้": ("ต้องมีสัญชาติไทยและมีรายได้สม่ำเสมอตามเกณฑ์", "โฮมโลนฟอร์ยู | สินเชื่อบ้าน | CIMB TH", "policy_requirement", 0.89),
    "ผ่อนไม่ไหวต้องทำอย่างไร ปรับโครงสร้างหนี้": ("สามารถยื่นคำขอปรับโครงสร้างหนี้และขยายงวดผ่อนภายใต้เงื่อนไขสถาบันการเงิน", "ใบคำขอปรับปรุงโครงสร้างหนี้", "hardship_support", 0.87),
    "ขอขยายระยะเวลาผ่อนได้ไหม": ("สามารถขอขยายระยะเวลาผ่อนได้", "ใบคำขอปรับปรุงโครงสร้างหนี้", "hardship_support", 0.86),
    "มีมาตรการช่วยเหลือลูกหนี้อะไรบ้าง": ("มีมาตรการช่วยเหลือลูกหนี้เป็นระยะ", "มาตรการช่วยเหลือลูกหนี้", "hardship_support", 0.83),
    "อัตราดอกเบี้ยสินเชื่อบ้านเท่าไหร่": ("อัตราดอกเบี้ยขึ้นกับแผนสินเชื่อ", "อัตราดอกเบี้ยสินเชื่อบ้าน", "interest_structure", 0.88),
    "เครดิตบูโรสำคัญอย่างไร": (NO_ANSWER_SENTINEL, "", "", 0.0),
}

def mock_rag(q):
    item = _KB.get(q)
    if not item:
        return {"answer": NO_ANSWER_SENTINEL, "sources": []}
    ans, title, cat, score = item
    if ans == NO_ANSWER_SENTINEL:
        return {"answer": ans, "sources": []}
    return {"answer": ans, "sources": [{"title": title, "category": cat, "score": score}]}

user = {
    "Salary": 35000, "Occupation": "Salaried_Employee", "Marriage_Status": "Single",
    "credit_score": 520, "credit_grade": "DD", "outstanding": 280000,
    "overdue": 45, "Coapplicant": False, "loan_amount": 2000000, "loan_term": 20,
}
model_out = {"prediction": 0, "probabilities": {"0": 0.72, "1": 0.28}}
shap = {"base_value": 0.5, "values": {
    "credit_score": -0.12, "credit_grade": -0.18,
    "outstanding": -0.08, "overdue": -0.10,
    "Salary": -0.05, "loan_amount": -0.03,
}}

print("=" * 60, flush=True)
print("REJECTED — use_issup=False (baseline)", flush=True)
print("=" * 60, flush=True)
t0 = time.time()
r_base = generate_response(user, model_out, shap, rag_lookup=mock_rag, use_issup=False)
print(f"time: {time.time()-t0:.1f}s", flush=True)
print(r_base["result_th"], flush=True)

print("\n" + "=" * 60, flush=True)
print("REJECTED — use_issup=True (with [IsSup] check)", flush=True)
print("=" * 60, flush=True)
t0 = time.time()
r_issup = generate_response(user, model_out, shap, rag_lookup=mock_rag, use_issup=True)
elapsed = time.time() - t0
print(f"time         : {elapsed:.1f}s", flush=True)
print(f"issup_score  : {r_issup['issup_score']}/5", flush=True)
print(f"issup_passed : {r_issup['issup_passed']}", flush=True)
print(f"mode         : {r_issup['mode']}", flush=True)
print(flush=True)
print(r_issup["result_th"], flush=True)
