import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)

class FeatureMergerService:
    @staticmethod
    def merge_features(customer_id: str, db_session: Any) -> Dict[str, Any]:
        """
        Simulates querying:
        1. Operational DB (history of customer)
        2. Credit Bureau DB
        3. Feature Store

        If the customer hasn't been seen before, we flag them as `is_thin_file` 
        and provide baseline/imputed features.
        """
        logger.info(f"Merging features for customer_id: {customer_id}")
        
        # Real-world: query db_session for history
        is_known = "existing" in customer_id.lower()
        
        if is_known:
            return {
                "historical_defaults": 0,
                "credit_bureau_score": 750,
                "is_thin_file": False,
                "months_since_last_delinquency": 36
            }
        else:
            # Impute for thin file
            return {
                "historical_defaults": -1, # Unknown
                "credit_bureau_score": 600, # Median default
                "is_thin_file": True,
                "months_since_last_delinquency": -1
            }
