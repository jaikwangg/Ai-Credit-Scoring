from fastapi import APIRouter, HTTPException, BackgroundTasks, Depends
from sqlalchemy.orm import Session
import logging
import uuid
from datetime import datetime

from src.api.schemas.payload import ScoringRequest, ScoringResponse, ModelExplanations
from src.db.database import get_db
from src.db import models
from src.services.feature_merger import FeatureMergerService
from src.services.model_runner import ModelRunnerService

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
        
        # Step 3: Build response
        response = ScoringResponse(
            request_id=payload.request_id,
            approved=model_result["approved"],
            probability_score=model_result["probability_score"],
            explanations=ModelExplanations(
                is_thin_file=merged_features["is_thin_file"],
                tree_shap_values=model_result["shap_values"]
            )
        )
        
        # Step 4: Persist to Operational DB
        db_result = models.CreditScoreResult(
            request_id=response.request_id,
            customer_id=payload.customer_id,
            approved=response.approved,
            probability_score=response.probability_score,
            is_thin_file=merged_features["is_thin_file"]
        )
        db.add(db_result)
        db.commit()
        
        # Step 5: Dispatch Async tasks (Audit, saving to DB)
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
