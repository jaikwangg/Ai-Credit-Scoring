from fastapi import APIRouter, HTTPException
import logging
import math

from src.api.schemas.payload import (
    RAGQueryRequest, RAGQueryResponse, RAGSource,
    SimplePlanRequest, ExternalPlanResponse,
)
from src.planner.rag_bridge import extract_rag_sources, get_rag_manager, make_rag_lookup
from src.planner.planning import generate_response

router = APIRouter()
logger = logging.getLogger(__name__)


@router.post("/rag/query", response_model=RAGQueryResponse)
async def rag_query(payload: RAGQueryRequest):
    """
    Query the RAG system directly.

    Retrieves relevant documents from ChromaDB and synthesizes a Thai-language answer.
    Useful for testing RAG quality or building a standalone Q&A interface.
    """
    manager = get_rag_manager()
    if manager is None:
        raise HTTPException(
            status_code=503,
            detail="RAG index unavailable. Run: uv run python -m src.ingest",
        )

    try:
        result = manager.query(
            question=payload.question,
            similarity_top_k=payload.top_k,
            include_sources=True,
        )
    except Exception as exc:
        logger.error("RAG query failed: %s", exc)
        raise HTTPException(status_code=500, detail="RAG query failed.")

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


# ---------------------------------------------------------------------------
# Scoring helpers (mirrors ModelRunnerService logic for flat UserInputFeatures)
# ---------------------------------------------------------------------------

_GRADE_RISK = {"AA": 0.05, "BB": 0.20, "CC": 0.40, "DD": 0.60, "EE": 0.75, "FF": 0.90}
_OCC_ADJ = {
    "Salaried_Employee": -0.03, "Employed": -0.03,
    "Freelancer": 0.00, "Self_Employed": 0.02, "Self-Employed": 0.02, "Unemployed": 0.15,
}
_W = {"credit_score": 0.25, "credit_grade": 0.38, "outstanding": 0.14,
      "overdue": 0.10, "lti": 0.08, "salary_level": 0.03}
_NEUTRAL = {"credit_score": 0.30, "credit_grade": 0.40, "outstanding": 0.10,
            "overdue": 0.05, "lti": 0.33, "salary_level": 0.40}


def _compute_plan_inputs(f) -> tuple[dict, dict, float]:
    """Return (user_input_dict, shap_json_dict, risk_prob) from UserInputFeatures."""
    salary = float(f.Salary)
    credit_score = float(f.credit_score)
    grade = str(f.credit_grade).upper()
    outstanding = float(f.outstanding)
    overdue = float(f.overdue)
    loan_amount = float(f.loan_amount)
    loan_term_years = max(float(f.loan_term), 0.1)
    occupation = str(f.Occupation or "")
    coapplicant = bool(f.Coapplicant)

    cs_norm = max(0.0, min(1.0, (750.0 - credit_score) / 350.0))
    grade_norm = _GRADE_RISK.get(grade, 0.40)
    annual_income = salary * 12.0
    lti = loan_amount / max(salary * loan_term_years, 1.0)
    out_norm = min(1.0, outstanding / max(annual_income * 2.0, 1.0))
    ov_norm = min(1.0, overdue / 90.0)
    lti_norm = min(1.0, lti / 3.0)
    salary_norm = max(0.0, min(1.0, 1.0 - salary / 150_000.0))
    occ_adj = _OCC_ADJ.get(occupation, 0.0)
    coop_adj = -0.05 if coapplicant else 0.0

    base_risk = max(0.05, min(0.95,
        _W["credit_score"] * cs_norm + _W["credit_grade"] * grade_norm
        + _W["outstanding"] * out_norm + _W["overdue"] * ov_norm
        + _W["lti"] * lti_norm + _W["salary_level"] * salary_norm
        + occ_adj + coop_adj
    ))
    risk_prob = 1.0 / (1.0 + math.exp(-4.5 * (base_risk - 0.35)))

    def _shap(w, actual, neutral): return round(w * (actual - neutral), 4)
    lti_shap = _shap(_W["lti"], lti_norm, _NEUTRAL["lti"])

    shap_json = {
        "base_value": 0.5,
        "values": {
            "credit_score": -_shap(_W["credit_score"], cs_norm, _NEUTRAL["credit_score"]),
            "credit_grade": -_shap(_W["credit_grade"], grade_norm, _NEUTRAL["credit_grade"]),
            "outstanding":  -_shap(_W["outstanding"], out_norm, _NEUTRAL["outstanding"]),
            "overdue":      -_shap(_W["overdue"], ov_norm, _NEUTRAL["overdue"]),
            "loan_amount":  round(-lti_shap * 0.5, 4),
            "loan_term":    round(-lti_shap * 0.5, 4),
            "Salary":       -_shap(_W["salary_level"], salary_norm, _NEUTRAL["salary_level"]),
        },
    }

    user_input = {
        "Salary": salary, "Occupation": occupation,
        "Marriage_Status": str(f.Marriage_Status or "Unknown"),
        "credit_score": credit_score, "credit_grade": grade,
        "outstanding": outstanding, "overdue": overdue,
        "Coapplicant": coapplicant,
        "loan_amount": loan_amount, "loan_term": loan_term_years,
        "Interest_rate": float(f.Interest_rate) if f.Interest_rate is not None else None,
    }

    return user_input, shap_json, risk_prob


@router.post("/plan/simple", response_model=ExternalPlanResponse)
async def plan_simple(payload: SimplePlanRequest):
    """
    Simple planning endpoint — accepts flat user features.

    Computes risk score and SHAP values internally (no external ML model needed),
    then returns a Thai-language improvement plan or approval checklist.
    """
    logger.info("Simple plan request: %s", payload.request_id)

    try:
        user_input, shap_json, risk_prob = _compute_plan_inputs(payload.features)
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
        plan_result = generate_response(user_input, model_output, shap_json, rag_lookup=rag_lookup)
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
    )
