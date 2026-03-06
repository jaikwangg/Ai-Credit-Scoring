from fastapi import APIRouter, HTTPException, BackgroundTasks, Depends
from sqlalchemy.orm import Session
from typing import Optional
import logging
from datetime import datetime

from src.api.schemas.payload import ScoringRequest, ScoringResponse, ModelExplanations, PlannerAdvice
from src.db.database import get_db
from src.db import models
from src.services.feature_merger import FeatureMergerService
from src.services.model_runner import ModelRunnerService
from src.planner.planning import generate_response
from src.planner.rag_bridge import build_shap_json, build_user_input, get_rag_manager, make_rag_lookup

router = APIRouter()
logger = logging.getLogger(__name__)

async def _audit_log_async(payload: dict):
    # Simulates pushing payload to a Kafka Topic or Elasticsearch
    logger.info(f"[AUDIT LOG] Logged payload for request {payload.get('request_id')} at {datetime.utcnow().isoformat()}")

@router.post("/score/request", response_model=ScoringResponse)
async def request_credit_score(
    payload: ScoringRequest,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db)
):
    """
    1. Receives strict JSON payload.
    2. Validates & Normalizes (handled via Pydantic).
    3. Merges with historical DB data.
    4. Calls Model Inference.
    5. Dispatches async Audit Logs.
    6. Returns Classification Result.
    """
    logger.info(f"Incoming scoring request: {payload.request_id} for customer {payload.customer_id}")
    
    try:
        # Step 1: Feature Merger (Querying DB and Feature Store)
        merged_features = FeatureMergerService.merge_features(payload.customer_id, db)
        
        # Step 2: Model Inference
        model_result = ModelRunnerService.run_inference(merged_features, payload)
        
        # Step 3: Planner + RAG advice
        advice: Optional[PlannerAdvice] = None
        try:
            manager = get_rag_manager()
            rag_lookup = make_rag_lookup(manager.query) if manager else None
            user_input = build_user_input(payload, merged_features)
            # probability_score is risk/default probability (higher = riskier).
            # Planner expects probabilities["1"] = P(approved), so we invert.
            risk_prob = model_result["probability_score"]
            model_output = {
                "prediction": 1 if model_result["approved"] else 0,
                "probabilities": {
                    "1": round(1.0 - risk_prob, 4),  # P(approved)
                    "0": round(risk_prob, 4),          # P(default)
                },
            }
            shap_json = build_shap_json(model_result["shap_values"])
            plan_result = generate_response(user_input, model_output, shap_json, rag_lookup=rag_lookup)

            rag_sources = []
            for action in plan_result.get("plan", {}).get("actions", []) if "plan" in plan_result else []:
                rag_sources.extend(action.get("evidence", []) or [])

            advice = PlannerAdvice(
                mode=plan_result.get("mode", ""),
                result_th=plan_result.get("result_th", ""),
                rag_sources=rag_sources,
            )
        except Exception as exc:
            logger.warning("Planner advice generation failed (non-fatal): %s", exc)

        # Step 4: Build response
        response = ScoringResponse(
            request_id=payload.request_id,
            approved=model_result["approved"],
            probability_score=model_result["probability_score"],
            explanations=ModelExplanations(
                is_thin_file=merged_features["is_thin_file"],
                tree_shap_values=model_result["shap_values"]
            ),
            advice=advice,
        )
        
        # Step 5: Persist to Operational DB
        # Idempotent write by request_id so repeated test runs don't fail with UNIQUE errors.
        db_result = (
            db.query(models.CreditScoreResult)
            .filter(models.CreditScoreResult.request_id == response.request_id)
            .first()
        )
        if db_result is None:
            db_result = models.CreditScoreResult(
                request_id=response.request_id,
                customer_id=payload.customer_id,
                approved=response.approved,
                probability_score=response.probability_score,
                is_thin_file=merged_features["is_thin_file"]
            )
            db.add(db_result)
        else:
            db_result.customer_id = payload.customer_id
            db_result.approved = response.approved
            db_result.probability_score = response.probability_score
            db_result.is_thin_file = merged_features["is_thin_file"]

        db.commit()
        
        # Step 6: Dispatch Async tasks (Audit, saving to DB)
        background_tasks.add_task(
            _audit_log_async,
            {
                "request_id": response.request_id,
                "customer_id": payload.customer_id,
                "decision": response.approved,
                "score": response.probability_score
            }
        )
        
        return response
    
    except Exception as e:
        logger.error(f"Error processing scoring request {payload.request_id}: {str(e)}")
        # DLQ logic would go here
        raise HTTPException(status_code=500, detail="Internal Server Error during scoring.")
