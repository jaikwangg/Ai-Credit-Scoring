"""Bridge between QueryEngineManager and the planner's rag_lookup interface."""

from __future__ import annotations

import logging
import threading
from typing import Any, Callable, Dict, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Interface adapter
# ---------------------------------------------------------------------------

def extract_rag_sources(plan_result: Dict[str, Any]) -> list:
    """Extract RAG evidence sources from a planner result dict."""
    sources = []
    for action in plan_result.get("plan", {}).get("actions", []) if "plan" in plan_result else []:
        sources.extend(action.get("evidence", []) or [])
    return sources


def make_rag_lookup(
    query_fn: Callable[[str], Dict[str, Any]],
    use_cache: bool = True,
) -> Callable[[str], dict]:
    """
    Wrap QueryEngineManager.query() into the planner's rag_lookup signature.

    planner expects: rag_lookup(query: str) -> {"answer": str, "sources": list}
    query_fn returns: {"answer": str, "sources": [...], "router_label": str, ...}

    Results are cached in the global QueryCache (TTL 1h, LRU 256 entries) when
    use_cache=True (default).  Empty / failed answers are never cached.
    """
    from src.rag.cache import get_cache
    cache = get_cache() if use_cache else None

    def rag_lookup(query: str) -> dict:
        if cache is not None:
            cached = cache.get(query)
            if cached is not None:
                logger.debug("RAG cache hit for query %r", query[:60])
                return cached

        try:
            result = query_fn(query)
            data = {
                "answer": result.get("answer", ""),
                "sources": result.get("sources", []),
            }
        except Exception as exc:
            logger.warning("RAG lookup failed for query %r: %s", query[:60], exc)
            return {"answer": "", "sources": []}

        # Only cache results that have BOTH an answer AND sources. Caching an
        # answer with empty sources poisons the cache: the planner would then
        # receive sourceless responses on every subsequent hit, even after the
        # underlying RAG index improves. Require sources so evidence is always
        # attachable to action items.
        if cache is not None and data.get("answer") and data.get("sources"):
            cache.set(query, data)

        return data

    return rag_lookup


# ---------------------------------------------------------------------------
# Input adapters
# ---------------------------------------------------------------------------

def build_user_input(payload: Any, merged_features: Dict[str, Any]) -> dict:
    """
    Map ScoringRequest + merged DB features to the planner's user_input format.

    Planner DRIVER_QUERY_MAP keys: overdue, outstanding, loan_amount, loan_term,
    Interest_rate, Occupation, Salary, credit_score, credit_grade, Coapplicant
    """
    financials = payload.financials
    loan = payload.loan_details
    demo = payload.demographics

    return {
        "Salary": float(financials.monthly_income),
        "Occupation": demo.employment_status,
        "Marriage_Status": demo.marital_status,
        "loan_amount": float(loan.loan_amount),
        "loan_term": float(loan.loan_term_months) / 12,  # convert to years
        "outstanding": float(financials.existing_debt),
        "overdue": float(merged_features.get("overdue_amount", 0.0)),
        "credit_score": float(merged_features.get("credit_bureau_score", 0.0)),
        "credit_grade": merged_features.get("credit_grade", ""),
        "Coapplicant": merged_features.get("has_coapplicant", False),
    }


def build_shap_json(shap_values: Dict[str, float], base_value: float = 0.5) -> dict:
    """
    Wrap a flat shap_values dict into the format expected by normalize_shap().

    normalize_shap() expects: {"base_value": float, "values": {"feature": shap, ...}}
    """
    return {
        "base_value": base_value,
        "values": {str(k): float(v) for k, v in (shap_values or {}).items()},
    }


# ---------------------------------------------------------------------------
# Lazy singleton for QueryEngineManager
# ---------------------------------------------------------------------------

_manager_lock = threading.Lock()
_manager: Optional[Any] = None
_UNAVAILABLE = object()  # sentinel: marks a failed init so we don't retry every request


def get_rag_manager() -> Optional[Any]:
    """
    Return a cached QueryEngineManager instance, loading it on first call.
    Returns None if the index is unavailable (service degrades gracefully).
    """
    global _manager
    if _manager is not None:
        return None if _manager is _UNAVAILABLE else _manager

    with _manager_lock:
        if _manager is not None:
            return None if _manager is _UNAVAILABLE else _manager
        try:
            import chromadb
            from llama_index.core import VectorStoreIndex
            from llama_index.core.settings import Settings
            from llama_index.embeddings.huggingface import HuggingFaceEmbedding
            from llama_index.vector_stores.chroma import ChromaVectorStore

            from config.settings import settings as cfg
            from src.query_engine import QueryEngineManager

            Settings.embed_model = HuggingFaceEmbedding(
                model_name=cfg.EMBEDDING_MODEL,
                embed_batch_size=32,
            )
            client = chromadb.PersistentClient(path=cfg.CHROMA_PERSIST_DIR)
            collection = client.get_collection(cfg.CHROMA_COLLECTION)
            vector_store = ChromaVectorStore(chroma_collection=collection)
            index = VectorStoreIndex.from_vector_store(vector_store)
            _manager = QueryEngineManager(index)
            logger.info("RAG manager initialised (collection=%s)", cfg.CHROMA_COLLECTION)
        except Exception as exc:
            logger.warning("RAG manager unavailable — planner will run without RAG: %s", exc)
            _manager = _UNAVAILABLE

    return None if _manager is _UNAVAILABLE else _manager
