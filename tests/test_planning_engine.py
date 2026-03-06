"""Unit tests for planner/planning.py."""

import sys
import unittest
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.planner.planning import (  # noqa: E402
    GENERAL_ONLY_NOTE,
    NO_ANSWER_SENTINEL,
    build_actions,
    build_clarifying_questions,
    generate_plan,
    generate_response,
    normalize_shap,
    parse_model_output,
    plan_to_thai_text,
    render_plan_th,
    summarize_shap,
)


class TestPlanningEngine(unittest.TestCase):
    def setUp(self):
        self.user_input = {
            "Sex": "M",
            "Occupation": "Employee",
            "Salary": 45000.0,
            "Marriage_Status": "Single",
            "credit_score": 640.0,
            "credit_grade": "B",
            "outstanding": 300000.0,
            "overdue": 15000.0,
            "Coapplicant": False,
            "loan_amount": 2500000.0,
            "loan_term": 25.0,
            "Interest_rate": 5.9,
        }
        self.approved_model_output = {
            "prediction": 1,
            "probabilities": {"0": 0.12, "1": 0.88},
        }
        self.rejected_model_output = {
            "prediction": 0,
            "probabilities": {"0": 0.72, "1": 0.28},
        }
        self.shap_json = {
            "base_value": 0.5,
            "values": {
                "Salary": 0.18,
                "outstanding": -0.35,
                "overdue": -0.22,
                "loan_amount": -0.15,
                "loan_term": -0.05,
                "Interest_rate": -0.03,
                "credit_score": -0.10,
                "credit_grade": -0.02,
                "Occupation": 0.01,
                "Coapplicant": 0.01,
                "Marriage_Status": 0.0,
                "Sex": -0.01,
            },
        }

    def test_parse_model_output_robust_keys(self):
        parsed = parse_model_output(self.approved_model_output)
        self.assertTrue(parsed["approved"])
        self.assertAlmostEqual(parsed["p_approve"], 0.88, places=6)
        self.assertAlmostEqual(parsed["p_reject"], 0.12, places=6)

        parsed_int_key = parse_model_output({"prediction": "0", "probabilities": {0: 0.67, 1: 0.33}})
        self.assertFalse(parsed_int_key["approved"])
        self.assertAlmostEqual(parsed_int_key["p_approve"], 0.33, places=6)
        self.assertAlmostEqual(parsed_int_key["p_reject"], 0.67, places=6)

    def test_normalize_shap_requires_values(self):
        with self.assertRaises(ValueError):
            normalize_shap({"base_value": 0.1})

        normalized = normalize_shap(self.shap_json)
        self.assertIn("outstanding", normalized)
        self.assertEqual(normalized["outstanding"], -0.35)

    def test_summarize_shap_non_actionable(self):
        summary = summarize_shap(normalize_shap(self.shap_json), top_k=4)
        self.assertEqual(summary["non_actionable"], ["Sex"])
        self.assertIn("Sex", summary["labels_th"])
        self.assertEqual(summary["top_negative"][0]["feature"], "outstanding")

    def test_build_actions_documented_and_general_only(self):
        summary = summarize_shap(normalize_shap(self.shap_json), top_k=6)

        def partial_rag(query: str):
            if "ขยายระยะเวลา" in query:
                return {
                    "answer": "สามารถขอขยายระยะเวลาผ่อนได้ตามเงื่อนไข",
                    "sources": [{"title": "Relief Doc", "category": "hardship_support", "score": 0.91}],
                }
            return {"answer": NO_ANSWER_SENTINEL, "sources": []}

        actions = build_actions(self.user_input, summary, rag_lookup=partial_rag)
        self.assertTrue(actions)
        self.assertTrue(any(a["evidence_confidence"] == "documented" for a in actions))
        self.assertTrue(any(a["evidence_confidence"] == "general_only" for a in actions))

        for action in actions:
            blob = f"{action['title_th']} {action['why_th']} {action['how_th']}".lower()
            self.assertNotIn("เปลี่ยนเพศ", blob)
            self.assertNotIn("change sex", blob)
            self.assertNotIn("รับประกันอนุมัติ", blob)
            if action["evidence_confidence"] == "general_only":
                self.assertIn(GENERAL_ONLY_NOTE, action["how_th"])

    def test_build_clarifying_questions_max_three(self):
        questions = build_clarifying_questions(self.user_input)
        self.assertGreaterEqual(len(questions), 1)
        self.assertLessEqual(len(questions), 3)
        self.assertIn("สินเชื่อประเภทใด", questions[0])

    def test_generate_response_approved_mode(self):
        rag_calls = []

        def rag_lookup(query: str):
            rag_calls.append(query)
            return {
                "answer": "เอกสารหลักคือบัตรประชาชนและเอกสารรายได้",
                "sources": [{"title": "Policy Doc", "category": "policy_requirement", "score": 0.9}],
            }

        response = generate_response(
            user_input=self.user_input,
            model_output=self.approved_model_output,
            shap_json=self.shap_json,
            rag_lookup=rag_lookup,
        )
        self.assertEqual(response["mode"], "approved_guidance")
        self.assertNotIn("plan", response)
        self.assertIn("เช็กลิสต์", response["result_th"])
        self.assertEqual(len(rag_calls), 3)

    def test_generate_response_rejected_mode(self):
        def rag_lookup(_: str):
            return {"answer": NO_ANSWER_SENTINEL, "sources": []}

        response = generate_response(
            user_input=self.user_input,
            model_output=self.rejected_model_output,
            shap_json=self.shap_json,
            rag_lookup=rag_lookup,
        )
        self.assertEqual(response["mode"], "improvement_plan")
        self.assertIn("plan", response)
        self.assertIn("actions", response["plan"])
        self.assertGreaterEqual(len(response["plan"]["clarifying_questions"]), 1)

    def test_generate_plan_contract_and_text(self):
        plan = generate_plan(
            user_input=self.user_input,
            model_output=self.rejected_model_output,
            shap_json=self.shap_json,
            rag_lookup=None,
        )
        self.assertIn("decision", plan)
        self.assertIn("risk_drivers", plan)
        self.assertIn("actions", plan)
        self.assertIn("clarifying_questions", plan)
        self.assertIn("ไม่สามารถรับประกันผลอนุมัติ", plan["disclaimer_th"])

        text = plan_to_thai_text(plan)
        self.assertIn("สรุปสั้น", text)
        self.assertIn("1) ทำทันที", text)
        self.assertIn("ข้อมูลที่ควรยืนยันเพิ่มเติม", text)
        self.assertIn("ไม่สามารถรับประกันผลอนุมัติ", text)

    def test_render_plan_th_dedup_and_interest_evidence_alignment(self):
        plan = {
            "decision": {"approved": False, "p_approve": 0.2, "p_reject": 0.8},
            "risk_drivers": {
                "top_negative": [
                    {"feature": "outstanding", "shap": -0.35, "label_th": "ยอดหนี้คงค้าง"},
                    {"feature": "overdue", "shap": -0.22, "label_th": "ยอดค้างชำระ"},
                    {"feature": "Interest_rate", "shap": -0.04, "label_th": "อัตราดอกเบี้ย"},
                ]
            },
            "actions": [
                {
                    "title_th": "จัดการหนี้ค้างและวางแผนปรับโครงสร้างภาระหนี้",
                    "why_th": "ปัจจัย 'ยอดหนี้คงค้าง' กดผลประเมิน (SHAP -0.35)",
                    "how_th": "ลดภาระหนี้",
                    "evidence": [{"source_title": "Relief A", "category": "hardship_support", "score": 0.9}],
                    "evidence_confidence": "documented",
                },
                {
                    "title_th": "จัดการหนี้ค้างและวางแผนปรับโครงสร้างภาระหนี้",
                    "why_th": "ปัจจัย 'ยอดค้างชำระ' กดผลประเมิน (SHAP -0.22)",
                    "how_th": "เคลียร์ค้างชำระ",
                    "evidence": [{"source_title": "Relief B", "category": "hardship_support", "score": 0.8}],
                    "evidence_confidence": "documented",
                },
                {
                    "title_th": "เปรียบเทียบทางเลือกดอกเบี้ยอย่างโปร่งใส",
                    "why_th": "ปัจจัย 'อัตราดอกเบี้ย' กดผลประเมิน (SHAP -0.04)",
                    "how_th": "เทียบทางเลือกดอกเบี้ย",
                    "evidence": [{"source_title": "เอกสารรายได้ทั่วไป", "category": "policy_requirement", "query": "รายได้ขั้นต่ำ", "score": 0.95}],
                    "evidence_confidence": "documented",
                },
            ],
            "clarifying_questions": ["ต้องการยื่นสินเชื่อประเภทใด?"],
            "disclaimer_th": "ไม่สามารถรับประกันผลอนุมัติ",
        }

        text_123 = render_plan_th(plan, style="123")
        text_para = render_plan_th(plan, style="paragraph")
        text_abc = render_plan_th(plan, style="ABC")

        self.assertIn("สรุปสั้น", text_123)
        self.assertIn("1) ทำทันที", text_123)
        self.assertEqual(text_123.count("จัดการหนี้ค้างและวางแผนปรับโครงสร้างภาระหนี้"), 1)
        self.assertNotIn("เอกสารรายได้ทั่วไป", text_123)

        self.assertIn("สรุปสั้น", text_para)
        self.assertIn("ข้อมูลที่ควรยืนยันเพิ่มเติม", text_para)

        self.assertIn("แผน A", text_abc)
        self.assertIn("แผน B", text_abc)
        self.assertIn("แผน C", text_abc)


if __name__ == "__main__":
    unittest.main(verbosity=2)
