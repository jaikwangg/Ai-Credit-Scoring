"""
Demo: Run all 3 test cases through the planner + mock RAG.

Test Case 1 : Low Risk   → approved_guidance mode
Test Case 2 : High Risk  → improvement_plan mode  (credit_score main driver)
Test Case 3 : Medium Risk → improvement_plan mode (loan_term / credit_grade drivers)

Run from project root:
    python examples/planner/test_cases_demo.py
"""
from __future__ import annotations

import sys
from pathlib import Path

project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

from src.planner.planning import NO_ANSWER_SENTINEL, generate_response, render_plan_th

# ---------------------------------------------------------------------------
# Mock RAG (same KB as mock_demo.py)
# ---------------------------------------------------------------------------

_KB = {
    "เอกสารที่ต้องใช้สมัครสินเชื่อบ้านมีอะไรบ้าง": (
        "ใช้บัตรประชาชน ทะเบียนบ้าน และเอกสารแสดงรายได้ตามประเภทอาชีพ",
        "โฮมโลนฟอร์ยู | สินเชื่อบ้าน | CIMB TH", "policy_requirement", 0.91,
    ),
    "ต้องมีคุณสมบัติอย่างไรถึงจะกู้บ้านได้": (
        "ต้องมีสัญชาติไทยและมีรายได้สม่ำเสมอตามเกณฑ์ธนาคาร",
        "โฮมโลนฟอร์ยู | สินเชื่อบ้าน | CIMB TH", "policy_requirement", 0.89,
    ),
    "รายได้ขั้นต่ำเท่าไหร่ถึงจะกู้บ้านได้": (
        "รายได้ขั้นต่ำเป็นไปตามเงื่อนไขผลิตภัณฑ์และประเภทผู้กู้",
        "โฮมโลนฟอร์ยู | สินเชื่อบ้าน | CIMB TH", "policy_requirement", 0.84,
    ),
    "ผ่อนไม่ไหวต้องทำอย่างไร ปรับโครงสร้างหนี้": (
        "สามารถยื่นคำขอปรับโครงสร้างหนี้และขยายงวดผ่อนภายใต้เงื่อนไขธนาคาร",
        "ใบคำขอปรับปรุงโครงสร้างหนี้ (ขยายระยะเวลาผ่อน)", "hardship_support", 0.87,
    ),
    "ขอขยายระยะเวลาผ่อนได้ไหม": (
        "สามารถขอขยายระยะเวลาผ่อนได้ โดยธนาคารจะพิจารณาตามความสามารถชำระ",
        "ใบคำขอปรับปรุงโครงสร้างหนี้ (ขยายระยะเวลาผ่อน)", "hardship_support", 0.86,
    ),
    "มีมาตรการช่วยเหลือลูกหนี้อะไรบ้าง": (
        "มีมาตรการช่วยเหลือลูกหนี้เป็นระยะตามประกาศของธนาคาร",
        "มาตรการช่วยเหลือลูกหนี้ระยะที่ 2", "hardship_support", 0.83,
    ),
    "อัตราดอกเบี้ยสินเชื่อบ้านเท่าไหร่": (
        "อัตราดอกเบี้ยขึ้นกับแผนสินเชื่อและช่วงเวลาโปรโมชัน",
        "อัตราดอกเบี้ยสินเชื่อบ้านใหม่ (Generic) ปี 2568/2569", "interest_structure", 0.88,
    ),
    "มี fixed rate หรือ floating rate บ้าง": (
        "มีทั้ง fixed rate และ floating rate ตามแผนสินเชื่อ",
        "loan-interest-rates-th.txt", "interest_structure", 0.81,
    ),
    "เครดิตบูโรสำคัญอย่างไร": (NO_ANSWER_SENTINEL, "", "", 0.0),
}


def mock_rag_lookup(query: str) -> dict:
    item = _KB.get(query)
    if not item:
        return {"answer": NO_ANSWER_SENTINEL, "sources": []}
    answer, title, category, score = item
    if answer == NO_ANSWER_SENTINEL:
        return {"answer": answer, "sources": []}
    return {"answer": answer, "sources": [{"title": title, "category": category, "score": score}]}


# ---------------------------------------------------------------------------
# Test Cases
# user_input  : flat dict fed into planner (matches DRIVER_QUERY_MAP keys)
# model_output: {"prediction": 0|1, "probabilities": {"0": p_risk, "1": p_approve}}
# shap_json   : {"base_value": 0.5, "values": {feature: shap_contribution}}
#
# SHAP sign convention (approval probability):
#   negative = feature HURTS approval  → planner builds improvement actions
#   positive = feature HELPS approval  → planner leaves as strength
#
# The user's expected SHAP values are in risk-probability convention,
# so we negate them here to match the planner's approval-probability convention.
# ---------------------------------------------------------------------------

TEST_CASES = [
    # ------------------------------------------------------------------
    # Test Case 1 : Low Risk  (probability_risk < 0.30 → approved)
    # ------------------------------------------------------------------
    {
        "label": "Test Case 1 — Low Risk (Approved)",
        "user_input": {
            "Sex": "Male",
            "Occupation": "Freelancer",
            "Salary": 500_000.0,
            "Marriage_Status": "Single",
            "credit_score": 700.0,
            "credit_grade": "AA",
            "outstanding": 0.0,
            "overdue": 0.0,
            "Coapplicant": 0,
            "loan_amount": 1_000_000.0,
            "loan_term": 10.0,
            "Interest_rate": 5.0,
        },
        # risk_prob ≈ 0.21 → P(approved) ≈ 0.79
        "model_output": {
            "prediction": 1,
            "probabilities": {"1": 0.79, "0": 0.21},
        },
        # SHAP in approval-probability convention (negated from risk SHAP):
        #   credit_score=+0.15, loan_term=+0.12, Salary=+0.04, credit_grade=+0.04
        "shap_json": {
            "base_value": 0.5,
            "values": {
                "credit_score":  0.15,
                "loan_term":     0.12,
                "Salary":        0.04,
                "credit_grade":  0.04,
                "Interest_rate": 0.02,
                "outstanding":   0.0,
                "overdue":       0.0,
                "loan_amount":  -0.01,
            },
        },
    },

    # ------------------------------------------------------------------
    # Test Case 2 : High Risk  (probability_risk ≈ 0.68 → rejected)
    # ------------------------------------------------------------------
    {
        "label": "Test Case 2 — High Risk (Rejected)",
        "user_input": {
            "Sex": "Male",
            "Occupation": "Salaried_Employee",
            "Salary": 55_000.0,
            "Marriage_Status": "Single",
            "credit_score": 652.0,
            "credit_grade": "FF",
            "outstanding": 601_387.0,
            "overdue": 60.0,
            "Coapplicant": 0,
            "loan_amount": 800_000.0,
            "loan_term": 26.0,
            "Interest_rate": 5.83,
        },
        # risk_prob ≈ 0.71 → P(approved) ≈ 0.29
        "model_output": {
            "prediction": 0,
            "probabilities": {"1": 0.32, "0": 0.68},
        },
        # SHAP (approval convention): credit_score is the main negative driver
        "shap_json": {
            "base_value": 0.5,
            "values": {
                "credit_score":  -0.75,
                "credit_grade":   0.13,
                "Interest_rate": -0.01,
                "loan_amount":   -0.01,
                "loan_term":     -0.01,
                "outstanding":   -0.08,
                "overdue":       -0.05,
                "Salary":        -0.03,
            },
        },
    },

    # ------------------------------------------------------------------
    # Test Case 3 : Medium Risk  (probability_risk ≈ 0.52 → borderline rejected)
    # ------------------------------------------------------------------
    {
        "label": "Test Case 3 — Medium Risk (Borderline)",
        "user_input": {
            "Sex": "Male",
            "Occupation": "Salaried_Employee",
            "Salary": 55_000.0,
            "Marriage_Status": "Single",
            "credit_score": 700.0,
            "credit_grade": "CC",
            "outstanding": 70_000.0,
            "overdue": 15.0,
            "Coapplicant": 0,
            "loan_amount": 1_100_000.0,
            "loan_term": 27.0,   # assumed 27 years (2727 in original looks like typo)
            "Interest_rate": 5.82,
        },
        # risk_prob ≈ 0.52 → P(approved) ≈ 0.48
        "model_output": {
            "prediction": 0,
            "probabilities": {"1": 0.48, "0": 0.52},
        },
        # SHAP (approval convention): negated from risk SHAP
        "shap_json": {
            "base_value": 0.5,
            "values": {
                "loan_term":     -0.05,
                "credit_grade":   0.04,
                "outstanding":    0.02,
                "credit_score":   0.02,
                "loan_amount":    0.01,
                "overdue":       -0.02,
                "Salary":        -0.01,
                "Interest_rate": -0.01,
            },
        },
    },
]


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    sep = "=" * 72

    for tc in TEST_CASES:
        print(f"\n{sep}")
        print(f"  {tc['label']}")
        print(sep)

        result = generate_response(
            user_input=tc["user_input"],
            model_output=tc["model_output"],
            shap_json=tc["shap_json"],
            rag_lookup=mock_rag_lookup,
        )

        print(f"Mode   : {result['mode']}")
        print(f"Decision: {'อนุมัติ' if result['decision']['approved'] else 'ไม่อนุมัติ'}"
              f"  (P_approve={result['decision']['p_approve']:.3f},"
              f" P_reject={result['decision']['p_reject']:.3f})")
        print()
        print(result["result_th"])


if __name__ == "__main__":
    main()
