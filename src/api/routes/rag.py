import asyncio
import logging
import time
from functools import partial

from fastapi import APIRouter, HTTPException

from src.api.schemas.payload import (
    RAGQueryRequest, RAGQueryResponse, RAGSource, SelfRAGTraceSchema,
    SimplePlanRequest, ExternalPlanResponse,
    SimulationRequest, SimulationResponse, ScenarioResult,
    BatchItem, BatchPlanRequest, BatchPlanResponse, BatchItemResult, BatchSummary,
)
from src.planner.rag_bridge import extract_rag_sources, get_rag_manager, make_rag_lookup
from src.planner.planning import generate_response
from src.planner.scoring import compute_plan_inputs

router = APIRouter()
logger = logging.getLogger(__name__)


@router.post("/rag/query", response_model=RAGQueryResponse)
async def rag_query(payload: RAGQueryRequest):
    """
    Query the RAG system directly.

    Retrieves relevant documents from ChromaDB and synthesizes a Thai-language answer.
    Useful for testing RAG quality or building a standalone Q&A interface.
    Results are cached (TTL 1h, LRU 256) — call DELETE /rag/cache to invalidate.
    """
    from src.rag.cache import get_cache

    manager = get_rag_manager()
    if manager is None:
        raise HTTPException(
            status_code=503,
            detail="RAG index unavailable. Run: uv run python -m src.ingest",
        )

    cache = get_cache()
    cached = cache.get(payload.question, top_k=payload.top_k)
    if cached is not None:
        logger.debug("RAG cache hit: %r", payload.question[:60])
        result = cached
    else:
        try:
            result = manager.query(
                question=payload.question,
                similarity_top_k=payload.top_k,
                include_sources=True,
            )
        except Exception as exc:
            logger.error("RAG query failed: %s", exc)
            raise HTTPException(status_code=500, detail="RAG query failed.")
        if result.get("answer"):
            cache.set(payload.question, result, top_k=payload.top_k)

    sources = []
    for src in result.get("sources", []):
        meta = src.get("metadata", {})
        sources.append(RAGSource(
            title=meta.get("title", "Unknown"),
            category=meta.get("category", "Uncategorized"),
            institution=meta.get("institution"),
            score=src.get("score"),
        ))

    return RAGQueryResponse(
        question=result["question"],
        answer=result["answer"],
        router_label=result.get("router_label", "general_info"),
        sources=sources,
        retrieved_count=result.get("retrieved_node_count", 0),
        validated_count=result.get("validated_node_count", 0),
    )


@router.post("/rag/query/self", response_model=RAGQueryResponse)
async def rag_query_self(payload: RAGQueryRequest):
    """
    Self-RAG query with three reflection steps:
      [Retrieve] → retrieval → [IsRel] filtering → synthesis → [IsSup] check → optional retry.

    Returns the same shape as /rag/query plus a 'self_rag_trace' field with
    per-step diagnostics (reflection scores, node counts, retry flag).
    """
    from src.rag.self_rag import SelfRAGOrchestrator

    manager = get_rag_manager()
    if manager is None:
        raise HTTPException(
            status_code=503,
            detail="RAG index unavailable. Run: uv run python -m src.ingest",
        )

    try:
        orch = SelfRAGOrchestrator(manager)
        result = orch.query(
            question=payload.question,
            similarity_top_k=payload.top_k,
            include_sources=True,
        )
    except Exception as exc:
        logger.error("Self-RAG query failed: %s", exc)
        raise HTTPException(status_code=500, detail="Self-RAG query failed.")

    sources = []
    for src in result.get("sources", []):
        meta = src.get("metadata") or {}
        if isinstance(meta, dict):
            sources.append(RAGSource(
                title=meta.get("title", "Unknown"),
                category=meta.get("category", "Uncategorized"),
                institution=meta.get("institution"),
                score=src.get("score"),
            ))

    trace_data = result.get("self_rag_trace")
    trace = SelfRAGTraceSchema(**trace_data) if trace_data else None

    return RAGQueryResponse(
        question=result["question"],
        answer=result["answer"],
        router_label=result.get("router_label", "general_info"),
        sources=sources,
        retrieved_count=result.get("retrieved_node_count", 0),
        validated_count=result.get("validated_node_count", 0),
        self_rag_trace=trace,
    )


# ---------------------------------------------------------------------------
# Cache management
# ---------------------------------------------------------------------------

@router.get("/rag/cache/stats")
async def rag_cache_stats():
    """
    Return current RAG query cache statistics.

    - **size**: number of entries currently in cache
    - **hit_rate**: cache hits / (hits + misses) since startup
    - **expired_entries**: entries still in cache but past their TTL
    - **evictions**: entries dropped due to max_size limit
    """
    from src.rag.cache import get_cache
    return get_cache().stats()


@router.delete("/rag/cache")
async def rag_cache_clear():
    """
    Clear all entries from the RAG query cache.

    Useful after re-ingesting documents so stale answers are not served.
    Returns the number of entries that were cleared.
    """
    from src.rag.cache import get_cache
    cleared = get_cache().clear()
    logger.info("RAG cache cleared: %d entries removed", cleared)
    return {"cleared": cleared, "message": f"Cache cleared — {cleared} entries removed."}


@router.post("/plan/simple", response_model=ExternalPlanResponse)
async def plan_simple(payload: SimplePlanRequest, use_issup: bool = False):
    """
    Simple planning endpoint — accepts flat user features.

    Computes risk score and SHAP values internally (no external ML model needed),
    then returns a Thai-language improvement plan or approval checklist.

    Optional query param `use_issup=true` enables a Self-RAG [IsSup] groundedness
    check on the LLM-synthesized plan. If the plan scores < 2/5, it falls back to
    the rule-based renderer (always grounded). Adds ~3-5s latency.
    """
    logger.info("Simple plan request: %s (use_issup=%s)", payload.request_id, use_issup)

    try:
        user_input, shap_json, risk_prob = compute_plan_inputs(payload.features)
    except Exception as exc:
        logger.error("Feature computation failed: %s", exc)
        raise HTTPException(status_code=422, detail=f"Feature computation failed: {exc}")

    approved = risk_prob < 0.50
    model_output = {
        "prediction": 1 if approved else 0,
        "probabilities": {"1": round(1.0 - risk_prob, 4), "0": round(risk_prob, 4)},
    }

    try:
        manager = get_rag_manager()
        rag_lookup = make_rag_lookup(manager.query) if manager else None
        plan_result = generate_response(
            user_input, model_output, shap_json,
            rag_lookup=rag_lookup,
            use_issup=use_issup,
        )
    except Exception as exc:
        logger.error("Planner failed: %s", exc)
        raise HTTPException(status_code=500, detail="Planning failed.")

    rag_sources = extract_rag_sources(plan_result)

    decision = plan_result.get("decision", {})
    return ExternalPlanResponse(
        request_id=payload.request_id,
        mode=plan_result.get("mode", ""),
        approved=decision.get("approved", approved),
        p_approve=decision.get("p_approve", round(1.0 - risk_prob, 4)),
        p_reject=decision.get("p_reject", round(risk_prob, 4)),
        result_th=plan_result.get("result_th", ""),
        rag_sources=rag_sources,
        issup_score=plan_result.get("issup_score"),
        issup_passed=plan_result.get("issup_passed"),
    )


# ---------------------------------------------------------------------------
# What-If / Counterfactual Simulation helpers
# ---------------------------------------------------------------------------

_NUMERIC_FEATURES = {
    "Salary", "credit_score", "outstanding", "overdue",
    "loan_amount", "loan_term", "Interest_rate",
}
_VALID_GRADES = {"AA", "BB", "CC", "DD", "EE", "FF"}


def _apply_what_if(features, what_if: dict):
    """Return a new UserInputFeatures with what_if changes applied."""
    from src.api.schemas.payload import UserInputFeatures

    data = features.model_dump()

    for field_name, change in what_if.items():
        if field_name not in data:
            raise ValueError(f"Unknown feature: {field_name!r}. Valid fields: {list(data.keys())}")

        change_dict = change if isinstance(change, dict) else change.model_dump(exclude_none=True)
        current = data[field_name]

        if "value" in change_dict and change_dict["value"] is not None:
            new_val = change_dict["value"]
            if field_name == "credit_grade" and str(new_val).upper() not in _VALID_GRADES:
                raise ValueError(f"credit_grade must be one of {_VALID_GRADES}, got {new_val!r}")
            data[field_name] = new_val

        elif "delta" in change_dict and change_dict["delta"] is not None:
            if field_name not in _NUMERIC_FEATURES:
                raise ValueError(f"'delta' is only valid for numeric features, got {field_name!r}")
            data[field_name] = max(0.0, (current or 0.0) + change_dict["delta"])

        elif "delta_pct" in change_dict and change_dict["delta_pct"] is not None:
            if field_name not in _NUMERIC_FEATURES:
                raise ValueError(f"'delta_pct' is only valid for numeric features, got {field_name!r}")
            data[field_name] = max(0.0, (current or 0.0) * (1.0 + change_dict["delta_pct"] / 100.0))

        else:
            raise ValueError(
                f"WhatIfChange for {field_name!r} must specify one of: value, delta, delta_pct"
            )

    return UserInputFeatures(**data)


def _build_verdict(baseline_approved: bool, sim_approved: bool, delta: float) -> str:
    if not baseline_approved and sim_approved:
        return f"การปรับปรุงนี้เปลี่ยนผลเป็น 'อนุมัติ' — P(อนุมัติ) เพิ่มขึ้น {delta:+.1%}"
    if baseline_approved and not sim_approved:
        return f"การเปลี่ยนแปลงนี้ทำให้ผลกลายเป็น 'ปฏิเสธ' — P(อนุมัติ) ลดลง {delta:+.1%}"
    if baseline_approved and sim_approved:
        return f"โปรไฟล์เดิมได้รับการอนุมัติอยู่แล้ว การปรับปรุงเพิ่ม P(อนุมัติ) อีก {delta:+.1%}"
    if delta > 0.15:
        return f"ผลดีอย่างมีนัยสำคัญ — P(อนุมัติ) เพิ่มขึ้น {delta:+.1%} แต่ยังไม่ถึงเกณฑ์อนุมัติ"
    if delta > 0.05:
        return f"ผลดีในระดับปานกลาง — P(อนุมัติ) เพิ่มขึ้น {delta:+.1%}"
    if delta < -0.05:
        return f"การเปลี่ยนแปลงนี้ส่งผลเชิงลบ — P(อนุมัติ) ลดลง {abs(delta):.1%}"
    return f"ผลกระทบเล็กน้อย — P(อนุมัติ) เปลี่ยนแปลง {delta:+.1%}"


@router.post("/plan/simulate", response_model=SimulationResponse)
async def plan_simulate(payload: SimulationRequest):
    """
    What-If / Counterfactual Simulation.

    Computes a baseline score from the provided features, applies the requested
    feature changes (what_if), then returns a side-by-side comparison of:
      - approval probability (before / after)
      - SHAP values (before / after / diff)
      - a Thai-language verdict on the impact

    Supported change types per feature:
      - **value**: set to exact value  (all features)
      - **delta**: add/subtract amount (numeric features only)
      - **delta_pct**: percentage change (numeric features only)

    Example what_if:
    ```json
    {
      "outstanding": {"delta": -200000},
      "Salary":      {"delta_pct": 15},
      "credit_grade": {"value": "BB"}
    }
    ```
    """
    # ── Baseline ──────────────────────────────────────────────────────────────
    try:
        base_user_input, base_shap_json, base_risk_prob = compute_plan_inputs(payload.features)
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Baseline computation failed: {exc}")

    base_approved = base_risk_prob < 0.50
    base_p_approve = round(1.0 - base_risk_prob, 4)
    base_p_reject = round(base_risk_prob, 4)

    # ── Apply what-if ────────────────────────────────────────────────────────
    try:
        sim_features = _apply_what_if(payload.features, payload.what_if)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))

    # ── Simulated ─────────────────────────────────────────────────────────────
    try:
        sim_user_input, sim_shap_json, sim_risk_prob = compute_plan_inputs(sim_features)
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Simulation computation failed: {exc}")

    sim_approved = sim_risk_prob < 0.50
    sim_p_approve = round(1.0 - sim_risk_prob, 4)
    sim_p_reject = round(sim_risk_prob, 4)

    # ── Diff ─────────────────────────────────────────────────────────────────
    base_shap = base_shap_json.get("values", {})
    sim_shap = sim_shap_json.get("values", {})
    all_features = set(base_shap) | set(sim_shap)
    shap_diff = {
        f: round(sim_shap.get(f, 0.0) - base_shap.get(f, 0.0), 4)
        for f in all_features
    }
    delta_p = round(sim_p_approve - base_p_approve, 4)
    changed_features = list(payload.what_if.keys())

    return SimulationResponse(
        request_id=payload.request_id,
        baseline=ScenarioResult(
            approved=base_approved,
            p_approve=base_p_approve,
            p_reject=base_p_reject,
            shap_values=base_shap,
            features=base_user_input,
        ),
        simulated=ScenarioResult(
            approved=sim_approved,
            p_approve=sim_p_approve,
            p_reject=sim_p_reject,
            shap_values=sim_shap,
            features=sim_user_input,
        ),
        delta_p_approve=delta_p,
        shap_diff=shap_diff,
        changed_features=changed_features,
        verdict=_build_verdict(base_approved, sim_approved, delta_p),
    )


# ---------------------------------------------------------------------------
# Batch Evaluation
# ---------------------------------------------------------------------------

@router.post("/plan/batch", response_model=BatchPlanResponse)
async def plan_batch(payload: BatchPlanRequest):
    """
    Batch Evaluation — score multiple applicants in one request.

    Processes each item independently.  If a single item fails (bad features, etc.)
    its result carries an `error` field and the rest of the batch continues.

    - **include_plan=false** (default): score-only mode — fast, no LLM calls.
    - **include_plan=true**: generates a full Thai-language improvement plan per
      item via LLM + RAG (same as `/plan/simple`).  Expect ~3-10 s per item
      depending on the LLM backend.

    Max batch size: 200 items.
    """
    t0 = time.monotonic()

    manager = get_rag_manager() if payload.include_plan else None
    rag_lookup = make_rag_lookup(manager.query) if manager else None

    loop = asyncio.get_event_loop()

    def _process_item(item: BatchItem) -> BatchItemResult:
        """Run one item synchronously — called in a thread pool to avoid blocking the event loop."""
        try:
            user_input, shap_json, risk_prob = compute_plan_inputs(item.features)
        except Exception as exc:
            return BatchItemResult(
                request_id=item.request_id,
                approved=False,
                p_approve=0.0,
                p_reject=1.0,
                error=str(exc),
            )

        approved = risk_prob < 0.50
        p_approve = round(1.0 - risk_prob, 4)
        p_reject = round(risk_prob, 4)

        mode: str | None = None
        result_th: str | None = None

        if payload.include_plan:
            model_output = {
                "prediction": 1 if approved else 0,
                "probabilities": {"1": p_approve, "0": p_reject},
            }
            try:
                plan_result = generate_response(
                    user_input, model_output, shap_json, rag_lookup=rag_lookup
                )
                mode = plan_result.get("mode")
                result_th = plan_result.get("result_th")
            except Exception as exc:
                logger.warning("Plan generation failed for %s: %s", item.request_id, exc)

        return BatchItemResult(
            request_id=item.request_id,
            approved=approved,
            p_approve=p_approve,
            p_reject=p_reject,
            mode=mode,
            result_th=result_th,
        )

    # Run all items concurrently in the default thread pool so the event loop stays free.
    results: list[BatchItemResult] = list(
        await asyncio.gather(
            *[loop.run_in_executor(None, _process_item, item) for item in payload.items]
        )
    )

    # ── Aggregate stats ───────────────────────────────────────────────────────
    ok = [r for r in results if r.error is None]
    approved_count = sum(1 for r in ok if r.approved)
    error_count = sum(1 for r in results if r.error is not None)
    avg_p_approve = round(sum(r.p_approve for r in ok) / len(ok), 4) if ok else 0.0
    approval_rate = round(approved_count / len(ok), 4) if ok else 0.0

    return BatchPlanResponse(
        batch_id=payload.batch_id,
        summary=BatchSummary(
            total=len(results),
            approved_count=approved_count,
            rejected_count=len(ok) - approved_count,
            error_count=error_count,
            approval_rate=approval_rate,
            avg_p_approve=avg_p_approve,
            elapsed_s=round(time.monotonic() - t0, 2),
        ),
        results=results,
    )
