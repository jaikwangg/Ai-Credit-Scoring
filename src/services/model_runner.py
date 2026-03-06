import math
import logging
from typing import Dict, Any
from src.api.schemas.payload import ScoringRequest

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Feature mappings
# ---------------------------------------------------------------------------

# Credit grade ordinal risk (AA = safest, FF = riskiest)
_GRADE_RISK: Dict[str, float] = {
    "AA": 0.05,
    "BB": 0.20,
    "CC": 0.40,
    "DD": 0.60,
    "EE": 0.75,
    "FF": 0.90,
}
_GRADE_NEUTRAL = 0.40  # CC grade = neutral

# Occupation-level risk adjustment
_OCCUPATION_ADJ: Dict[str, float] = {
    "Salaried_Employee": -0.03,
    "Employed":          -0.03,
    "Freelancer":         0.00,
    "Self_Employed":      0.02,
    "Self-Employed":      0.02,
    "Unemployed":         0.15,
}

# Feature weights
_W = {
    "credit_score": 0.25,
    "credit_grade": 0.38,
    "outstanding":  0.14,
    "overdue":      0.10,
    "lti":          0.08,
    "salary_level": 0.03,
}

# Neutral component values for SHAP baseline (medium-risk applicant)
_NEUTRAL = {
    "credit_score": 0.30,   # ≈ credit_score 645
    "credit_grade": _GRADE_NEUTRAL,
    "outstanding":  0.10,
    "overdue":      0.05,
    "lti":          0.33,
    "salary_level": 0.40,
}

# Reference monthly income (THB) for salary-level normalization
_SALARY_REF = 150_000.0


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


class ModelRunnerService:
    @staticmethod
    def run_inference(
        merged_features: Dict[str, Any],
        payload: ScoringRequest,
    ) -> Dict[str, Any]:
        """
        Calibrated credit-risk scoring model.

        Returns probability of DEFAULT (risk): higher value = riskier applicant.
        Threshold for approval: risk_prob < 0.50.

        SHAP convention: positive = feature increases risk (bad for applicant),
                         negative = feature decreases risk (good for applicant).

        Feature names match planner's DRIVER_QUERY_MAP and FEATURE_LABELS_TH so
        that RAG evidence is fetched for the correct risk drivers.

        Approximate calibration targets:
          Low risk  (AA, score 700+, clean)  → prob < 0.30
          High risk (FF, score <650, overdue) → prob ≈ 0.68
          Medium risk (CC, score 700, some debt) → prob ≈ 0.50
        """
        logger.info("Running model inference for request: %s", payload.request_id)

        # ---------------------------------------------------------------
        # Feature extraction
        # ---------------------------------------------------------------
        credit_score    = float(merged_features.get("credit_bureau_score", 600))
        credit_grade    = str(merged_features.get("credit_grade", "CC")).upper()
        outstanding     = float(merged_features.get("outstanding", 0.0))
        overdue         = float(merged_features.get("overdue_amount", 0.0))
        has_coapplicant = bool(merged_features.get("has_coapplicant", False))
        is_thin_file    = bool(merged_features.get("is_thin_file", True))

        salary          = float(payload.financials.monthly_income)
        loan_amount     = float(payload.loan_details.loan_amount)
        # API stores term in months; convert to years for LTI
        loan_term_years = max(float(payload.loan_details.loan_term_months) / 12.0, 0.1)
        occupation      = payload.demographics.employment_status

        # ---------------------------------------------------------------
        # Derived features
        # ---------------------------------------------------------------
        annual_income = salary * 12.0
        # Loan-to-income ratio: loan vs total income over term
        lti = loan_amount / max(salary * loan_term_years, 1.0)

        # ---------------------------------------------------------------
        # Normalized component scores [0, 1]  (0 = safe, 1 = risky)
        # ---------------------------------------------------------------
        # credit_score: 750+ → 0.0 (safe), 400 → 1.0 (risky)
        cs_norm    = max(0.0, min(1.0, (750.0 - credit_score) / 350.0))
        grade_norm = _GRADE_RISK.get(credit_grade, _GRADE_NEUTRAL)
        # outstanding as fraction of 2× annual income
        out_norm   = min(1.0, outstanding / max(annual_income * 2.0, 1.0))
        # overdue days, normalized to 90-day window
        ov_norm    = min(1.0, overdue / 90.0)
        # LTI > 3 = very risky territory
        lti_norm   = min(1.0, lti / 3.0)
        # Low salary relative to reference = higher risk
        salary_norm = max(0.0, min(1.0, 1.0 - salary / _SALARY_REF))

        # ---------------------------------------------------------------
        # Occupation & file adjustments (direct additions to base_risk)
        # ---------------------------------------------------------------
        occ_adj         = _OCCUPATION_ADJ.get(occupation, 0.0)
        thin_file_adj   = 0.08 if is_thin_file else 0.0
        coapplicant_adj = -0.05 if has_coapplicant else 0.0

        # ---------------------------------------------------------------
        # Weighted risk score
        # ---------------------------------------------------------------
        base_risk = (
            _W["credit_score"] * cs_norm
            + _W["credit_grade"] * grade_norm
            + _W["outstanding"]  * out_norm
            + _W["overdue"]      * ov_norm
            + _W["lti"]          * lti_norm
            + _W["salary_level"] * salary_norm
            + occ_adj
            + thin_file_adj
            + coapplicant_adj
        )
        base_risk = max(0.05, min(0.95, base_risk))

        # Sigmoid calibration centred at 0.35
        logit     = 4.5 * (base_risk - 0.35)
        risk_prob = _sigmoid(logit)

        # ---------------------------------------------------------------
        # Pseudo-SHAP values (deviation from neutral baseline)
        # Positive = increases risk (bad), Negative = decreases risk (good)
        # Feature names match planner's DRIVER_QUERY_MAP & FEATURE_LABELS_TH
        # ---------------------------------------------------------------
        def _shap(weight: float, actual: float, neutral: float) -> float:
            return round(weight * (actual - neutral), 4)

        lti_shap = _shap(_W["lti"], lti_norm, _NEUTRAL["lti"])

        shap_values: Dict[str, float] = {
            "credit_score": _shap(_W["credit_score"], cs_norm,     _NEUTRAL["credit_score"]),
            "credit_grade": _shap(_W["credit_grade"], grade_norm,  _NEUTRAL["credit_grade"]),
            "outstanding":  _shap(_W["outstanding"],  out_norm,    _NEUTRAL["outstanding"]),
            "overdue":      _shap(_W["overdue"],       ov_norm,     _NEUTRAL["overdue"]),
            # Split LTI contribution across loan_amount and loan_term
            "loan_amount":  round(lti_shap * 0.5, 4),
            "loan_term":    round(lti_shap * 0.5, 4),
            "Salary":       _shap(_W["salary_level"], salary_norm, _NEUTRAL["salary_level"]),
        }

        approved = risk_prob < 0.50

        return {
            "approved": approved,
            "probability_score": round(risk_prob, 4),
            "shap_values": shap_values,
        }
