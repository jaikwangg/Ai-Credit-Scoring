"""
RAG Evaluation — offline quality measurement.

Run:
    pytest tests/test_rag_eval.py -v -s

Thresholds (adjust as your corpus improves):
    ROUTE_ACCURACY_MIN    0.75  — router labels correct 75%+ of the time
    KEYWORD_HIT_MIN       0.60  — answer contains expected keywords 60%+ of the time
    NO_ANSWER_ACCURACY_MIN 0.80 — out-of-domain questions correctly rejected 80%+

To add new test cases, append a JSONL line to data/eval/test_cases.jsonl:
    {"question": "...", "expected_route": "interest_structure",
     "expected_keywords": ["%", "ดอกเบี้ย"], "should_answer": true}
"""

from __future__ import annotations

from pathlib import Path

import pytest

EVAL_CASES_PATH = Path("data/eval/test_cases.jsonl")

ROUTE_ACCURACY_MIN = 0.75
KEYWORD_HIT_MIN = 0.60
NO_ANSWER_ACCURACY_MIN = 0.80
# If more than this fraction of cases time out, skip the metric assertion
TIMEOUT_SKIP_THRESHOLD = 0.50


def _load_manager():
    import chromadb
    from llama_index.core.settings import Settings
    from llama_index.embeddings.huggingface import HuggingFaceEmbedding
    from llama_index.vector_stores.chroma import ChromaVectorStore
    from llama_index.core import VectorStoreIndex
    from llama_index.llms.ollama import Ollama

    try:
        from config.settings import settings as cfg
        from src.query_engine import QueryEngineManager

        Settings.embed_model = HuggingFaceEmbedding(
            model_name=cfg.EMBEDDING_MODEL,
            embed_batch_size=32,
        )
        chroma_client = chromadb.PersistentClient(path=cfg.CHROMA_PERSIST_DIR)
        chroma_collection = chroma_client.get_collection(cfg.CHROMA_COLLECTION)
        vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
        index = VectorStoreIndex.from_vector_store(vector_store)
        manager = QueryEngineManager(index)
        # Use a longer timeout for eval so individual slow queries don't crash the run
        manager.llm = Ollama(
            model=cfg.OLLAMA_MODEL,
            base_url=cfg.OLLAMA_BASE_URL,
            temperature=0.1,
            request_timeout=300.0,
        )
        return manager
    except Exception as exc:
        pytest.skip(f"Could not load index: {exc}")


@pytest.mark.skipif(
    not EVAL_CASES_PATH.exists(),
    reason=f"Test cases not found at {EVAL_CASES_PATH}",
)
def test_rag_route_accuracy():
    """Router must label at least ROUTE_ACCURACY_MIN of questions correctly."""
    from src.rag.eval import load_test_cases, run_eval, compute_metrics, print_report

    manager = _load_manager()
    cases = load_test_cases(EVAL_CASES_PATH)
    assert cases, "No eval cases loaded"

    results = run_eval(cases, manager.query)
    metrics = compute_metrics(results)
    print_report(metrics, results)

    timed_out = metrics.get("timed_out", 0)
    if timed_out / metrics["total"] >= TIMEOUT_SKIP_THRESHOLD:
        pytest.skip(f"{timed_out}/{metrics['total']} cases timed out — increase Ollama request_timeout or reduce test set")

    assert metrics["route_accuracy"] >= ROUTE_ACCURACY_MIN, (
        f"Route accuracy {metrics['route_accuracy']:.1%} < {ROUTE_ACCURACY_MIN:.1%}\n"
        "Check router.py ROUTE_KEYWORDS for missing Thai terms."
    )


@pytest.mark.skipif(
    not EVAL_CASES_PATH.exists(),
    reason=f"Test cases not found at {EVAL_CASES_PATH}",
)
def test_rag_keyword_hit_rate():
    """Answers must contain expected keywords at least KEYWORD_HIT_MIN of the time."""
    from src.rag.eval import load_test_cases, run_eval, compute_metrics

    manager = _load_manager()
    cases = [c for c in load_test_cases(EVAL_CASES_PATH) if c.should_answer]
    assert cases

    results = run_eval(cases, manager.query)
    metrics = compute_metrics(results)

    timed_out = metrics.get("timed_out", 0)
    if timed_out / metrics["total"] >= TIMEOUT_SKIP_THRESHOLD:
        pytest.skip(f"{timed_out}/{metrics['total']} cases timed out — increase Ollama request_timeout or reduce test set")

    assert metrics["keyword_hit_rate"] >= KEYWORD_HIT_MIN, (
        f"Keyword hit rate {metrics['keyword_hit_rate']:.1%} < {KEYWORD_HIT_MIN:.1%}\n"
        "Check validator.py ROUTE_MUST_HAVE or document coverage."
    )


@pytest.mark.skipif(
    not EVAL_CASES_PATH.exists(),
    reason=f"Test cases not found at {EVAL_CASES_PATH}",
)
def test_rag_no_answer_rejection():
    """Out-of-domain questions must be rejected at least NO_ANSWER_ACCURACY_MIN of the time."""
    from src.rag.eval import load_test_cases, run_eval, compute_metrics

    manager = _load_manager()
    cases = [c for c in load_test_cases(EVAL_CASES_PATH) if not c.should_answer]
    if not cases:
        pytest.skip("No out-of-domain cases in test set")

    results = run_eval(cases, manager.query)
    metrics = compute_metrics(results)

    timed_out = metrics.get("timed_out", 0)
    if timed_out / metrics["total"] >= TIMEOUT_SKIP_THRESHOLD:
        pytest.skip(f"{timed_out}/{metrics['total']} cases timed out — increase Ollama request_timeout or reduce test set")

    no_ans_acc = metrics.get("no_answer_accuracy")
    assert no_ans_acc is not None and no_ans_acc >= NO_ANSWER_ACCURACY_MIN, (
        f"No-answer accuracy {no_ans_acc:.1%} < {NO_ANSWER_ACCURACY_MIN:.1%}\n"
        "Check validator.py GLOBAL_BLOCKLIST or HOME_DOMAIN_KEYWORDS."
    )
