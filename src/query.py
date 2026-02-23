import json
import re
import socket

import chromadb
import httpx
import requests
from llama_index.core import StorageContext, VectorStoreIndex, load_index_from_storage
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.response_synthesizers import get_response_synthesizer
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.settings import Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.vector_stores.chroma import ChromaVectorStore

try:
    from .schema import AssistantResponse
    from .settings import (
        CHROMA_COLLECTION,
        CHROMA_PERSIST_DIR,
        EMBED_MODEL,
        INDEX_DIR,
        OLLAMA_BASE_URL,
        OLLAMA_MODEL,
        VECTOR_STORE_TYPE,
    )
except ImportError:  # pragma: no cover - script execution fallback
    from src.schema import AssistantResponse
    from src.settings import (
        CHROMA_COLLECTION,
        CHROMA_PERSIST_DIR,
        EMBED_MODEL,
        INDEX_DIR,
        OLLAMA_BASE_URL,
        OLLAMA_MODEL,
        VECTOR_STORE_TYPE,
    )


def _friendly_ollama_error(exc: Exception) -> RuntimeError:
    """Convert low-level errors into actionable Ollama guidance."""
    if isinstance(
        exc,
        (socket.timeout, TimeoutError, requests.exceptions.Timeout, httpx.TimeoutException),
    ):
        return RuntimeError(
            f"Ollama request timed out. Check server responsiveness at {OLLAMA_BASE_URL} and try again."
        )

    if isinstance(
        exc,
        (ConnectionRefusedError, requests.exceptions.ConnectionError, httpx.ConnectError),
    ):
        return RuntimeError(
            f"Cannot connect to Ollama at {OLLAMA_BASE_URL}. Ensure Ollama Desktop is running and reachable."
        )

    if isinstance(exc, httpx.HTTPStatusError):
        status_code = exc.response.status_code if exc.response is not None else None
        if status_code == 404:
            return RuntimeError(
                f"Ollama model '{OLLAMA_MODEL}' was not found (HTTP 404). Pull the model in Ollama Desktop or update OLLAMA_MODEL."
            )
        return RuntimeError(
            f"Ollama request failed with HTTP {status_code}. Verify OLLAMA_BASE_URL={OLLAMA_BASE_URL} and OLLAMA_MODEL={OLLAMA_MODEL}."
        )

    if isinstance(exc, requests.exceptions.HTTPError):
        status_code = exc.response.status_code if exc.response is not None else None
        if status_code == 404:
            return RuntimeError(
                f"Ollama model '{OLLAMA_MODEL}' was not found (HTTP 404). Pull the model in Ollama Desktop or update OLLAMA_MODEL."
            )
        return RuntimeError(
            f"Ollama request failed with HTTP {status_code}. Verify OLLAMA_BASE_URL={OLLAMA_BASE_URL} and OLLAMA_MODEL={OLLAMA_MODEL}."
        )

    return RuntimeError(
        "Ollama query failed unexpectedly. Verify OLLAMA_BASE_URL and OLLAMA_MODEL settings."
    )


def extract_json(text: str) -> dict:
    """Extract the first JSON object from model output."""
    match = re.search(r"\{.*\}", text, flags=re.S)
    if not match:
        raise ValueError("No JSON object found in model output.")
    return json.loads(match.group(0))


def _load_index():
    # BGE-M3 embeddings (must match index build) for query encoding
    Settings.embed_model = HuggingFaceEmbedding(
        model_name=EMBED_MODEL,
        embed_batch_size=32,
    )
    if VECTOR_STORE_TYPE == "chroma":
        try:
            chroma_client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
            chroma_collection = chroma_client.get_collection(CHROMA_COLLECTION)
            vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
            return VectorStoreIndex.from_vector_store(vector_store)
        except Exception as exc:
            raise RuntimeError(
                f"Unable to load ChromaDB collection '{CHROMA_COLLECTION}' from "
                f"'{CHROMA_PERSIST_DIR}'. Build the index first with "
                "VECTOR_STORE_TYPE=chroma."
            ) from exc

    storage_context = StorageContext.from_defaults(persist_dir=INDEX_DIR)
    return load_index_from_storage(storage_context)


def get_engine(top_k: int = 8):
    try:
        Settings.llm = Ollama(
            model=OLLAMA_MODEL,
            base_url=OLLAMA_BASE_URL,
            request_timeout=120,
        )
    except Exception as exc:
        raise _friendly_ollama_error(exc) from None

    index = _load_index()

    retriever = VectorIndexRetriever(index=index, similarity_top_k=top_k)

    synthesizer = get_response_synthesizer(
        llm=Settings.llm,
        response_mode="compact",
    )

    return RetrieverQueryEngine(
        retriever=retriever,
        response_synthesizer=synthesizer,
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
        question=question,
    )

    try:
        raw = engine.query(prompt)
    except Exception as exc:
        raise _friendly_ollama_error(exc) from None

    text = str(raw)
    data = extract_json(text)
    validated = AssistantResponse.model_validate(data)
    return validated


if __name__ == "__main__":
    decision_json = {
        "decision": {"final": "review", "confidence": "medium"},
        "model": {
            "approval_prob": 0.62,
            "model_decision": "review",
            "top_shap": [
                {"feature": "overdue", "impact": -0.31, "direction": "negative"}
            ],
        },
        "rules": {"hard_fail": False, "checks": []},
    }

    resp = explain_case(
        question="Summarize rationale and next steps for this case.",
        decision_json=decision_json,
    )
    print(resp.model_dump_json(indent=2, ensure_ascii=False))
