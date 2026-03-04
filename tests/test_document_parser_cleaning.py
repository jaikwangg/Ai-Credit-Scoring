"""Unit tests for document cleaning and metadata enrichment helpers."""

import sys
import unittest
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.document_parser import clean_scraped_text, extract_effective_year


class TestDocumentParserCleaning(unittest.TestCase):
    def test_clean_scraped_text_removes_known_noise_lines(self):
        raw = """
ค้นหา
ติดต่อเรา
สมัคร
โปรโมชั่น
สินเชื่อบ้านรีไฟแนนซ์ อัตราดอกเบี้ยพิเศษ
ค่าธรรมเนียม เริ่มต้น 1%
Cookie
Share
""".strip()

        cleaned = clean_scraped_text(raw)

        removed_terms = ["ค้นหา", "ติดต่อเรา", "สมัคร", "โปรโมชั่น", "cookie", "share"]
        removed_count = sum(1 for term in removed_terms if term.lower() not in cleaned.lower())

        self.assertGreaterEqual(removed_count, 4)
        self.assertIn("สินเชื่อบ้านรีไฟแนนซ์", cleaned)

    def test_extract_effective_year_from_title_range(self):
        title = "อัตราดอกเบี้ยสินเชื่อบ้านใหม่ ปี 2568/2569"
        year = extract_effective_year(title)
        self.assertEqual(year, "2568/2569")


if __name__ == "__main__":
    unittest.main(verbosity=2)
