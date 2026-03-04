"""Tests for parser quarantine and directory-level gating behavior."""

import sys
import tempfile
import unittest
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.document_parser import (
    StructuredDocumentParser,
    UNRELATED_SUMMARY_MESSAGE,
    WEB_CHROME_QUARANTINE_SUMMARY,
)


def _write_structured_doc(path: Path, *, title: str, url: str, category: str, body: str) -> None:
    payload = f"""TITLE: {title}
SOURCE URL: {url}
INSTITUTION: CIMB Thai
PUBLICATION DATE: 2026-01-01
CATEGORY: {category}
---
SUMMARY (3-5 sentences relevance)
Legacy summary that should be ignored.
---
FULL CLEANED TEXT CONTENT
{body}
"""
    path.write_text(payload, encoding="utf-8")


class TestDocumentParserQuarantine(unittest.TestCase):
    def test_unrelated_customer_profiling_is_quarantined(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            file_path = Path(tmp_dir) / "customer-profiling.txt"
            _write_structured_doc(
                file_path,
                title="Customer Profiling ฟีเจอร์ใหม่เพิ่มความปลอดภัยให้ธุรกรรมออนไลน์",
                url="https://www.cimbthai.com/th/personal/important-notices/2025/customer-profiling.html",
                category="bank_policy",
                body="\n".join(
                    [
                        "Search",
                        "Quicklinks",
                        "Debit card",
                        "เงินฝาก",
                        "FX",
                        "All rights reserved",
                    ]
                ),
            )

            doc = StructuredDocumentParser.parse_file(file_path)
            self.assertIsNotNone(doc)
            self.assertTrue(doc.metadata.get("quarantined"))
            self.assertIn(doc.metadata.get("topic"), {"unrelated", "unrelated_web_chrome"})
            self.assertEqual(doc.metadata.get("category"), "unrelated")
            self.assertTrue(
                (UNRELATED_SUMMARY_MESSAGE in doc.text)
                or (WEB_CHROME_QUARANTINE_SUMMARY in doc.text)
            )

    def test_parse_directory_skips_quarantined_documents(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp = Path(tmp_dir)

            _write_structured_doc(
                tmp / "unrelated.txt",
                title="Customer Profiling Notice",
                url="https://www.cimbthai.com/th/personal/important-notices/2025/customer-profiling.html",
                category="bank_policy",
                body="Search\nQuicklinks\nDebit card\nFX",
            )

            _write_structured_doc(
                tmp / "home-loan.txt",
                title="อัตราดอกเบี้ยสินเชื่อบ้าน 2569",
                url="https://www.cimbthai.com/th/personal/products/loans/home-loan/home-loan-4u.html",
                category="interest_structure",
                body="\n".join(
                    [
                        "สินเชื่อบ้านสำหรับลูกค้ารายย่อยที่มีรายได้ประจำ",
                        "อัตราดอกเบี้ยคงที่ปีแรก 5.50% และปีถัดไปอ้างอิง MRR",
                        "เงื่อนไขการสมัครและเอกสารประกอบเป็นไปตามประกาศธนาคาร",
                    ]
                ),
            )

            docs = StructuredDocumentParser.parse_directory(tmp)
            report = StructuredDocumentParser.get_last_parse_report()

            self.assertEqual(len(docs), 1)
            self.assertEqual(report["total_docs"], 2)
            self.assertEqual(report["indexed_docs"], 1)
            self.assertEqual(report["quarantined_docs"], 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
