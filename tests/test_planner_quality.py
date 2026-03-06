"""
Plan quality validation for the 3 canonical test cases.

Checks four correctness layers per test case:
  1. Mode      — approved_guidance vs improvement_plan matches decision
  2. Driver    — top SHAP driver triggers the right action group
  3. Evidence  — RAG sources are cited when mock returns an answer
  4. Safety    — no forbidden tokens (guarantee / fraud language)

Run:
    pytest tests/test_planner_quality.py -v
"""
from __future__ import annotations

import re
import sys
import unittest
from pathlib import Path
from typing import Optional

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.planner.planning import (
    GENERAL_ONLY_NOTE,
    NO_ANSWER_SENTINEL,
    FORBIDDEN_FRAUD_TOKENS,
    FORBIDDEN_PROMISE_TOKENS,
    generate_response,
)

# ---------------------------------------------------------------------------
# Shared mock RAG (same KB as test_cases_demo.py)
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


def _mock_rag(query: str) -> dict:
    item = _KB.get(query)
    if not item:
        return {"answer": NO_ANSWER_SENTINEL, "sources": []}
    answer, title, category, score = item
    if answer == NO_ANSWER_SENTINEL:
        return {"answer": answer, "sources": []}
    return {"answer": answer, "sources": [{"title": title, "category": category, "score": score}]}


def _no_rag(_: str) -> dict:
    return {"answer": NO_ANSWER_SENTINEL, "sources": []}


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _all_action_text(result: dict) -> str:
    """Concatenate all action fields for safety scanning."""
    parts = [result.get("result_th", "")]
    for action in result.get("plan", {}).get("actions", []):
        parts += [
            str(action.get("title_th", "")),
            str(action.get("why_th", "")),
            str(action.get("how_th", "")),
        ]
    return " ".join(parts).lower()


def _has_rag_evidence(result: dict) -> bool:
    """True if at least one action cites a documented RAG source."""
    for action in result.get("plan", {}).get("actions", []):
        if action.get("evidence_confidence") == "documented" and action.get("evidence"):
            return True
    return False


def _top_driver_feature(result: dict) -> Optional[str]:
    """Return the feature name of the most-negative SHAP driver."""
    negatives = result.get("plan", {}).get("risk_drivers", {}).get("top_negative", [])
    return negatives[0]["feature"] if negatives else None


# ---------------------------------------------------------------------------
# Test Case 1 — Low Risk (Approved)
# ---------------------------------------------------------------------------

class TestLowRiskApproved(unittest.TestCase):
    """TC1: Freelancer, salary=500k, credit_score=700, grade=AA, no debt."""

    def setUp(self):
        self.user_input = {
            "Occupation": "Freelancer", "Salary": 500_000.0,
            "Marriage_Status": "Single", "credit_score": 700.0,
            "credit_grade": "AA", "outstanding": 0.0, "overdue": 0.0,
            "Coapplicant": 0, "loan_amount": 1_000_000.0, "loan_term": 10.0,
            "Interest_rate": 5.0,
        }
        self.model_output = {"prediction": 1, "probabilities": {"1": 0.79, "0": 0.21}}
        self.shap_json = {
            "base_value": 0.5,
            "values": {
                "credit_score": 0.15, "loan_term": 0.12, "Salary": 0.04,
                "credit_grade": 0.04, "Interest_rate": 0.02,
                "outstanding": 0.0, "overdue": 0.0, "loan_amount": -0.01,
            },
        }
        self.result = generate_response(
            self.user_input, self.model_output, self.shap_json, rag_lookup=_mock_rag
        )

    # --- Layer 1: Mode ---
    def test_mode_is_approved_guidance(self):
        self.assertEqual(self.result["mode"], "approved_guidance")

    # --- Layer 2: Driver (approved → no negative drivers, checklist instead) ---
    def test_approved_result_has_checklist(self):
        self.assertIn("เช็กลิสต์", self.result["result_th"])

    def test_no_improvement_plan_in_approved(self):
        self.assertNotIn("plan", self.result)

    # --- Layer 3: Evidence — checklist items should cite RAG sources ---
    def test_approved_checklist_cites_sources(self):
        self.assertIn("แหล่งข้อมูล", self.result["result_th"])
        self.assertIn("CIMB TH", self.result["result_th"])

    # --- Layer 4: Safety ---
    def test_no_forbidden_tokens(self):
        text = _all_action_text(self.result)
        for token in FORBIDDEN_PROMISE_TOKENS:
            self.assertNotIn(token.lower(), text, msg=f"Found forbidden promise token: {token}")
        for token in FORBIDDEN_FRAUD_TOKENS:
            self.assertNotIn(token.lower(), text, msg=f"Found forbidden fraud token: {token}")


# ---------------------------------------------------------------------------
# Test Case 2 — High Risk (Rejected, credit_score main driver)
# ---------------------------------------------------------------------------

class TestHighRiskRejected(unittest.TestCase):
    """TC2: Salaried, salary=55k, credit_score=652, grade=FF, 601k outstanding."""

    def setUp(self):
        self.user_input = {
            "Occupation": "Salaried_Employee", "Salary": 55_000.0,
            "Marriage_Status": "Single", "credit_score": 652.0,
            "credit_grade": "FF", "outstanding": 601_387.0, "overdue": 60.0,
            "Coapplicant": 0, "loan_amount": 800_000.0, "loan_term": 26.0,
            "Interest_rate": 5.83,
        }
        self.model_output = {"prediction": 0, "probabilities": {"1": 0.32, "0": 0.68}}
        # SHAP approval convention: credit_score=-0.75 is the dominant negative driver
        self.shap_json = {
            "base_value": 0.5,
            "values": {
                "credit_score": -0.75, "credit_grade": 0.13,
                "Interest_rate": -0.01, "loan_amount": -0.01,
                "loan_term": -0.01, "outstanding": -0.08,
                "overdue": -0.05, "Salary": -0.03,
            },
        }
        self.result = generate_response(
            self.user_input, self.model_output, self.shap_json, rag_lookup=_mock_rag
        )

    # --- Layer 1: Mode ---
    def test_mode_is_improvement_plan(self):
        self.assertEqual(self.result["mode"], "improvement_plan")

    def test_decision_is_rejected(self):
        self.assertFalse(self.result["decision"]["approved"])
        self.assertGreater(self.result["decision"]["p_reject"], 0.60)

    # --- Layer 2: Driver — credit_score should be the #1 negative driver ---
    def test_top_driver_is_credit_score(self):
        top = _top_driver_feature(self.result)
        self.assertEqual(top, "credit_score",
                         msg=f"Expected credit_score as top driver, got: {top}")

    def test_plan_contains_credit_action(self):
        """Plan must mention credit rehabilitation for the credit_score driver."""
        text = self.result["result_th"].lower()
        self.assertTrue(
            any(kw in text for kw in ["เครดิต", "credit", "ฟื้นฟูวินัย"]),
            msg="Plan should contain credit-related action for credit_score driver",
        )

    def test_plan_contains_debt_action(self):
        """outstanding + overdue are also negative → debt restructuring action expected."""
        text = self.result["result_th"].lower()
        self.assertTrue(
            any(kw in text for kw in ["หนี้", "ปรับโครงสร้าง", "ค้างชำระ"]),
            msg="Plan should mention debt management actions",
        )

    # --- Layer 3: Evidence ---
    def test_plan_has_rag_evidence(self):
        self.assertTrue(
            _has_rag_evidence(self.result),
            msg="At least one action should cite a RAG source (documented evidence)",
        )

    def test_rag_source_is_cited_in_text(self):
        self.assertIn("อ้างอิง", self.result["result_th"])

    # --- Layer 4: Safety ---
    def test_no_forbidden_tokens(self):
        text = _all_action_text(self.result)
        for token in FORBIDDEN_PROMISE_TOKENS:
            self.assertNotIn(token.lower(), text)
        for token in FORBIDDEN_FRAUD_TOKENS:
            self.assertNotIn(token.lower(), text)

    def test_disclaimer_present(self):
        self.assertIn("ไม่สามารถรับประกันผลอนุมัติ", self.result["result_th"])

    # --- Structural completeness ---
    def test_plan_has_three_sections(self):
        text = self.result["result_th"]
        self.assertIn("1) ทำทันที", text)
        self.assertIn("2) ภายใน 1-3 เดือน", text)
        self.assertIn("3) ภายใน 3-6 เดือน", text)


# ---------------------------------------------------------------------------
# Test Case 3 — Medium Risk (Borderline, loan_term main driver)
# ---------------------------------------------------------------------------

class TestMediumRiskBorderline(unittest.TestCase):
    """TC3: Salaried, salary=55k, credit_score=700, grade=CC, 70k outstanding."""

    def setUp(self):
        self.user_input = {
            "Occupation": "Salaried_Employee", "Salary": 55_000.0,
            "Marriage_Status": "Single", "credit_score": 700.0,
            "credit_grade": "CC", "outstanding": 70_000.0, "overdue": 15.0,
            "Coapplicant": 0, "loan_amount": 1_100_000.0, "loan_term": 27.0,
            "Interest_rate": 5.82,
        }
        self.model_output = {"prediction": 0, "probabilities": {"1": 0.48, "0": 0.52}}
        # SHAP approval convention: loan_term=-0.05 is top negative driver
        self.shap_json = {
            "base_value": 0.5,
            "values": {
                "loan_term": -0.05, "credit_grade": 0.04, "outstanding": 0.02,
                "credit_score": 0.02, "loan_amount": 0.01,
                "overdue": -0.02, "Salary": -0.01, "Interest_rate": -0.01,
            },
        }
        self.result = generate_response(
            self.user_input, self.model_output, self.shap_json, rag_lookup=_mock_rag
        )

    # --- Layer 1: Mode ---
    def test_mode_is_improvement_plan(self):
        self.assertEqual(self.result["mode"], "improvement_plan")

    def test_decision_is_borderline_rejected(self):
        self.assertFalse(self.result["decision"]["approved"])
        p_reject = self.result["decision"]["p_reject"]
        self.assertGreater(p_reject, 0.50)
        self.assertLess(p_reject, 0.70, msg="Borderline: p_reject should not be too high")

    # --- Layer 2: Driver — loan_term is main driver ---
    def test_top_driver_is_loan_term(self):
        top = _top_driver_feature(self.result)
        self.assertEqual(top, "loan_term",
                         msg=f"Expected loan_term as top driver, got: {top}")

    def test_plan_mentions_loan_structure(self):
        """loan_term driver → plan should suggest adjusting loan structure."""
        text = self.result["result_th"].lower()
        self.assertTrue(
            any(kw in text for kw in ["ระยะเวลากู้", "วงเงิน", "ค่างวด", "ปรับโครงสร้าง"]),
            msg="Plan should contain loan structure advice for loan_term driver",
        )

    # --- Layer 3: Evidence ---
    def test_plan_has_rag_evidence(self):
        self.assertTrue(
            _has_rag_evidence(self.result),
            msg="At least one action should cite a RAG source",
        )

    # --- Layer 4: Safety ---
    def test_no_forbidden_tokens(self):
        text = _all_action_text(self.result)
        for token in FORBIDDEN_PROMISE_TOKENS:
            self.assertNotIn(token.lower(), text)
        for token in FORBIDDEN_FRAUD_TOKENS:
            self.assertNotIn(token.lower(), text)

    def test_disclaimer_present(self):
        self.assertIn("ไม่สามารถรับประกันผลอนุมัติ", self.result["result_th"])


# ---------------------------------------------------------------------------
# Cross-cutting: degraded mode (no RAG)
# ---------------------------------------------------------------------------

class TestPlannerDegradedMode(unittest.TestCase):
    """Planner must still produce a valid plan when RAG is unavailable."""

    _rejected_output = {"prediction": 0, "probabilities": {"1": 0.32, "0": 0.68}}
    _shap_json = {
        "base_value": 0.5,
        "values": {"credit_score": -0.75, "outstanding": -0.08, "overdue": -0.05},
    }
    _user_input = {
        "Occupation": "Salaried_Employee", "Salary": 55_000.0,
        "credit_score": 652.0, "credit_grade": "FF",
        "outstanding": 601_387.0, "overdue": 60.0,
        "loan_amount": 800_000.0, "loan_term": 26.0,
    }

    def setUp(self):
        self.result = generate_response(
            self._user_input, self._rejected_output, self._shap_json, rag_lookup=None
        )

    def test_plan_generated_without_rag(self):
        self.assertEqual(self.result["mode"], "improvement_plan")
        self.assertIn("plan", self.result)

    def test_all_actions_marked_general_only(self):
        for action in self.result["plan"]["actions"]:
            self.assertEqual(
                action["evidence_confidence"], "general_only",
                msg="Without RAG, all actions should be general_only",
            )
            self.assertIn(GENERAL_ONLY_NOTE, action["how_th"],
                          msg="general_only actions must include the GENERAL_ONLY_NOTE disclaimer")

    def test_no_fake_rag_sources(self):
        for action in self.result["plan"]["actions"]:
            self.assertEqual(
                action.get("evidence", []), [],
                msg="No RAG sources should appear when RAG is unavailable",
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
