import json
import re
from llama_index.core import StorageContext, load_index_from_storage
from llama_index.core.settings import Settings
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.response_synthesizers import ResponseSynthesizer
from llama_index.llms.ollama import Ollama

from .settings import INDEX_DIR, OLLAMA_BASE_URL, LLM_MODEL, EMBED_MODEL
from .schema import AssistantResponse

def extract_json(text: str) -> dict:
    """
    LLM ชอบแถมข้อความก่อน/หลัง JSON — จับเฉพาะก้อน JSON ที่ใหญ่สุด
    """
    match = re.search(r"\{.*\}", text, flags=re.S)
    if not match:
        raise ValueError("No JSON object found in model output.")
    return json.loads(match.group(0))

def get_engine(top_k: int = 8):
    # LLM
    Settings.llm = Ollama(
        model=LLM_MODEL,
        base_url=OLLAMA_BASE_URL,
        request_timeout=120
    )

    # โหลด index
    storage_context = StorageContext.from_defaults(persist_dir=INDEX_DIR)
    index = load_index_from_storage(storage_context)

    # Retriever (+ metadata filters ได้ตอนใช้ Qdrant/PGVector จะเด่นกว่า FAISS)
    retriever = VectorIndexRetriever(
        index=index,
        similarity_top_k=top_k
    )

    # Synthesizer (แบบ custom prompt)
    synthesizer = ResponseSynthesizer.from_args(
        response_mode="compact"  # บีบ context + ลดเยิ่นเย้อ
    )

    return RetrieverQueryEngine(
        retriever=retriever,
        response_synthesizer=synthesizer
    )

PROMPT_TEMPLATE = """You are a Credit Underwriting Assistant.
You MUST answer ONLY in valid JSON matching this schema:
{
  "summary": string,
  "decision": "approve"|"decline"|"need_more_info"|"review",
  "reasons": [{"type":"rule"|"model"|"policy","text":string,"evidence":[{"doc_title":string,"version":string|null,"section":string|null,"page":number|null}]}],
  "missing_info": [string],
  "next_actions": [string],
  "customer_message_draft": string|null,
  "risk_note": string|null
}

Rules:
- Do NOT invent policy thresholds. If not found, say need_more_info or review and explain what is missing.
- Reasons must be consistent with provided decision_json.
- Evidence must cite retrieved documents when referencing policies or rules. If no evidence, leave evidence=[] and avoid quoting numbers.

decision_json:
{decision_json}

User question:
{question}
"""

def explain_case(question: str, decision_json: dict):
    engine = get_engine(top_k=10)

    prompt = PROMPT_TEMPLATE.format(
        decision_json=json.dumps(decision_json, ensure_ascii=False),
        question=question
    )

    # Query (RAG)
    raw = engine.query(prompt)
    text = str(raw)

    # Parse + Validate
    data = extract_json(text)
    validated = AssistantResponse.model_validate(data)
    return validated

if __name__ == "__main__":
    # ตัวอย่าง decision_json (ของจริงให้เรียกจาก Decision Service)
    decision_json = {
        "decision": {"final": "review", "confidence": "medium"},
        "model": {"approval_prob": 0.62, "model_decision": "review",
                  "top_shap": [{"feature":"overdue","impact":-0.31,"direction":"negative"}]},
        "rules": {"hard_fail": False, "checks": []}
    }

    resp = explain_case(
        question="สรุปเหตุผลและ next steps สำหรับเคสนี้",
        decision_json=decision_json
    )
    print(resp.model_dump_json(indent=2, ensure_ascii=False))
