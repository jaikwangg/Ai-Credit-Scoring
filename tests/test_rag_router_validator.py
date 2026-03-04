"""Unit tests for RAG router and relevance validator."""

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.rag.router import metadata_matches_route, route_query
from src.rag.validator import needs_close_account_clarification, validate_nodes


def _node(text: str, metadata: dict, score: float = 0.5):
    inner = SimpleNamespace(text=text, metadata=metadata)
    return SimpleNamespace(node=inner, score=score)


class TestRagRouterAndValidator(unittest.TestCase):
    def test_route_query_keywords(self):
        self.assertEqual(route_query("ค่าจดจำนองเท่าไหร่"), "fee_structure")
        self.assertEqual(route_query("fixed rate กับ floating rate ต่างกันยังไง"), "interest_structure")
        self.assertEqual(route_query("ผ่อนไม่ไหวต้องทำอย่างไร"), "hardship_support")
        self.assertEqual(route_query("รีไฟแนนซ์คืออะไร"), "refinance")
        self.assertEqual(route_query("ต้องมีคุณสมบัติอะไรบ้าง"), "policy_requirement")

    def test_metadata_matches_route_fee(self):
        metadata = {
            "title": "ค่าธรรมเนียมบริการสินเชื่อบ้าน",
            "category": "fee_structure",
            "doc_kind": "policy",
            "topic_tags": ["fee"],
        }
        self.assertTrue(metadata_matches_route(metadata, "fee_structure"))

    def test_validate_nodes_drops_unrelated_domain(self):
        good = _node(
            "สินเชื่อบ้าน อัตราดอกเบี้ยคงที่",
            {"institution": "CIMB Thai", "doc_kind": "rate_sheet", "domain": "loan"},
            0.8,
        )
        bad = _node(
            "บัญชีเงินฝากออมทรัพย์และบัตรเครดิต",
            {"institution": "CIMB Thai", "doc_kind": "policy", "domain": "deposit"},
            0.9,
        )
        kept = validate_nodes("ดอกเบี้ยสินเชื่อบ้านเท่าไหร่", [bad, good])
        self.assertEqual(len(kept), 1)
        self.assertIn("สินเชื่อบ้าน", kept[0].node.text)

    def test_close_account_clarification_trigger(self):
        evidence = _node(
            "ค่าปรับกรณีปิดสินเชื่อก่อนกำหนดก่อน 5 ปี คิด 1% ของวงเงินกู้",
            {"institution": "CIMB Thai", "doc_kind": "policy", "domain": "loan"},
            0.7,
        )
        self.assertTrue(
            needs_close_account_clarification("ค่าปิดบัญชีเท่าไหร่", [evidence])
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
