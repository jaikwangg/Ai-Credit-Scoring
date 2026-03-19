"""Rule-based credit risk scoring for the planning endpoints.

Computes risk_prob and SHAP-like values from flat UserInputFeatures
without requiring an external ML model.
"""
from __future__ import annotations

import math
from typing import Any

_GRADE_RISK = {"AA": 0.05, "BB": 0.20, "CC": 0.40, "DD": 0.60, "EE": 0.75, "FF": 0.90}
_OCC_ADJ = {
    "Salaried_Employee": -0.03, "Employed": -0.03,
    "Freelancer": 0.00, "Self_Employed": 0.02, "Self-Employed": 0.02, "Unemployed": 0.15,
}
_W = {"credit_score": 0.25, "credit_grade": 0.38, "outstanding": 0.14,
      "overdue": 0.10, "lti": 0.08, "salary_level": 0.03}
_NEUTRAL = {"credit_score": 0.30, "credit_grade": 0.40, "outstanding": 0.10,
            "overdue": 0.05, "lti": 0.33, "salary_level": 0.40}


def compute_plan_inputs(f: Any) -> tuple[dict, dict, float]:
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
