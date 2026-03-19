import logging
from typing import Dict, Any

from src.db.models import CreditScoreResult

logger = logging.getLogger(__name__)

class FeatureMergerService:
    @staticmethod
    def merge_features(customer_id: str, db_session: Any) -> Dict[str, Any]:
        """
        Queries the operational DB for customer history, then returns merged
        features for the scoring model.

        If the customer has no prior records, they are flagged as `is_thin_file`
        and conservative baseline features are imputed.
        """
        logger.info(f"Merging features for customer_id: {customer_id}")

        prior_record = (
            db_session.query(CreditScoreResult)
            .filter(CreditScoreResult.customer_id == customer_id)
            .order_by(CreditScoreResult.id.desc())
            .first()
        )
        is_known = prior_record is not None

        if is_known:
            # Populate from the most recent DB record.
            # TODO: extend with credit bureau and feature store queries.
            approved_history = prior_record.approved
            return {
                "historical_defaults": 0 if approved_history else 1,
                "credit_bureau_score": 700,  # placeholder until bureau API is integrated
                "credit_grade": "BB",        # placeholder
                "outstanding": 0.0,
                "overdue_amount": 0.0,
                "has_coapplicant": False,
                "is_thin_file": False,
                "months_since_last_delinquency": 36,
            }
        else:
            # Thin-file: no history found — impute conservative defaults
            return {
                "historical_defaults": -1,   # Unknown
                "credit_bureau_score": 600,  # Median default
                "credit_grade": "CC",        # Median grade imputed
                "outstanding": 0.0,
                "overdue_amount": 0.0,
                "has_coapplicant": False,
                "is_thin_file": True,
                "months_since_last_delinquency": -1,
            }
