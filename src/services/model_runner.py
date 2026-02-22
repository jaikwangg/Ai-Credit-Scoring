import logging
from typing import Dict, Any
from src.api.schemas.payload import ScoringRequest

logger = logging.getLogger(__name__)

class ModelRunnerService:
    @staticmethod
    def run_inference(merged_features: Dict[str, Any], payload: ScoringRequest) -> Dict[str, Any]:
        """
        Simulates sending the final feature vector to a BentoML/Triton inference server.
        """
        logger.info(f"Running model inference for request: {payload.request_id}")
        
        raw_income = float(payload.financials.monthly_income)
        raw_loan = float(payload.loan_details.loan_amount)
        
        # Extract features
        bureau_score = merged_features.get("credit_bureau_score", 600)
        is_thin_file = merged_features.get("is_thin_file", True)
        
        base_prob = 0.5
        
        # Income to loan ratio heuristics (mocking what a model would learn)
        if raw_income > (raw_loan / 24):
            base_prob += 0.2
            
        if bureau_score > 700:
            base_prob += 0.2
        elif bureau_score < 650:
            base_prob -= 0.2
            
        if is_thin_file:
            base_prob -= 0.15 # Penalty for thin file
            
        prob = max(0.01, min(0.99, base_prob))
        
        return {
            "approved": prob > 0.6,
            "probability_score": prob,
            "shap_values": {
                "income_ratio": 0.15 * (1 if prob > 0.5 else -1),
                "bureau_score_impact": 0.1 * (1 if bureau_score > 650 else -1),
                "thin_file_penalty": -0.15 if is_thin_file else 0.0
            }
        }
