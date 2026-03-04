#!/usr/bin/env python3
"""Quality-aware test runner for CIMB Thai home-loan RAG."""

from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from config.settings import settings
from src.document_parser import StructuredDocumentParser
from src.indexer import IndexManager
from src.query_engine import QueryEngineManager
from src.rag.validator import NO_ANSWER_MESSAGE

POISON_TOKENS = ("ndid", "กรมสรรพากร", "พร้อมเพย์")
RELIEF_CATEGORIES = {"hardship_support", "consumer_guideline", "relief"}
INTEREST_HEADER_HINTS = ("ประเภทหลักประกัน", "กลุ่มลูกค้า", "ระยะเวลา", "อายุสัญญา", "เฉลี่ย 3 ปี")


def configure_console_utf8() -> None:
    for stream_name in ("stdout", "stderr"):
        stream = getattr(sys, stream_name, None)
        if hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(encoding="utf-8", errors="replace")
            except Exception:
                pass


def _contains_any(text: str, terms: Tuple[str, ...]) -> bool:
    lower = (text or "").lower()
    return any(term in lower for term in terms)


def _sentence_count(text: str) -> int:
    normalized = (text or "").replace("\r", "\n")
    chunks = re.split(r"\n+|(?<!\d)[.!?]+(?!\d)", normalized)
    parts = [p.strip(" -*\t") for p in chunks if p.strip(" -*\t")]
    return len(parts)


def _is_no_answer(answer: str) -> bool:
    return (answer or "").strip().startswith(NO_ANSWER_MESSAGE)


def _extract_source_categories(result: Dict[str, Any]) -> List[str]:
    categories: List[str] = []
    for source in result.get("sources", []):
        metadata = source.get("metadata", {})
        category = str(metadata.get("category", "")).strip()
        if category:
            categories.append(category)
    return categories


def evaluate_quality(question_type: str, query: str, result: Dict[str, Any]) -> List[str]:
    answer = str(result.get("answer", ""))
    issues: List[str] = []
    no_answer = _is_no_answer(answer)
    lower_answer = answer.lower()

    if question_type == "refinance" and _contains_any(lower_answer, POISON_TOKENS):
        issues.append("poisoned_refinance_answer")

    if question_type == "hardship_support" and not no_answer:
        source_categories = {cat.lower() for cat in _extract_source_categories(result)}
        if not source_categories.intersection(RELIEF_CATEGORIES):
            issues.append("hardship_without_relief_source")

    if question_type == "interest_structure" and not no_answer:
        if not re.search(r"\d+(?:\.\d+)?\s*%|mrr", answer, flags=re.IGNORECASE):
            issues.append("interest_missing_numeric_or_mrr")
        if _sentence_count(answer) < 2:
            issues.append("interest_answer_too_short")
        if _contains_any(answer, INTEREST_HEADER_HINTS) and _sentence_count(answer) < 3:
            issues.append("interest_looks_like_raw_table_fragment")

    if no_answer:
        has_context = bool(result.get("sources"))
        blocked_count = int(result.get("domain_drift_blocked_count", 0))
        if has_context and blocked_count <= 0:
            issues.append("no_answer_without_context_or_block_reason")

    return issues


def _collect_quarantine_metrics() -> Dict[str, Any]:
    StructuredDocumentParser.parse_directory(Path(settings.DOCUMENTS_DIR))
    report = StructuredDocumentParser.get_last_parse_report()
    return {
        "total_docs": int(report.get("total_docs", 0)),
        "indexed_docs": int(report.get("indexed_docs", 0)),
        "quarantined_docs_count": int(report.get("quarantined_docs", 0)),
    }


def main() -> None:
    configure_console_utf8()

    print("Testing CIMB Thai Loan Documents RAG (quality mode)")
    print("=" * 72)

    quarantine_metrics = _collect_quarantine_metrics()
    print(
        "Document gating: "
        f"total={quarantine_metrics['total_docs']} "
        f"indexed={quarantine_metrics['indexed_docs']} "
        f"quarantined={quarantine_metrics['quarantined_docs_count']}"
    )

    index_manager = IndexManager()
    index = index_manager.load_index()
    if index is None:
        print("No index found. Creating new index...")
        index = index_manager.create_index()

    print("Index ready.")
    print()

    query_manager = QueryEngineManager(index)

    test_queries: List[Dict[str, str]] = [
        {"type": "policy_requirement", "query": "ต้องมีคุณสมบัติอย่างไรถึงจะกู้บ้านได้"},
        {"type": "policy_requirement", "query": "เอกสารที่ต้องใช้สมัครสินเชื่อบ้านมีอะไรบ้าง"},
        {"type": "interest_structure", "query": "มี fixed rate หรือ floating rate บ้าง"},
        {"type": "interest_structure", "query": "อัตราดอกเบี้ยสินเชื่อบ้านเท่าไหร่"},
        {"type": "fee_structure", "query": "ค่าจดจำนองเท่าไหร่"},
        {"type": "fee_structure", "query": "ค่าปิดบัญชีเท่าไหร่"},
        {"type": "refinance", "query": "รีไฟแนนซ์คืออะไร"},
        {"type": "refinance", "query": "เงื่อนไขการรีไฟแนนซ์เป็นอย่างไร"},
        {"type": "hardship_support", "query": "ผ่อนไม่ไหวต้องทำอย่างไร"},
        {"type": "hardship_support", "query": "มีมาตรการช่วยเหลือโควิดมีอะไรบ้าง"},
        {"type": "hardship_support", "query": "มีมาตรการช่วยเหลือผู้ประสบภัยน้ำท่วมไหม"},
    ]

    total = 0
    passed = 0
    failed = 0
    poisoned_answer_count = 0
    domain_drift_blocked_count = 0

    print("Running tests:")
    print("-" * 72)

    for item in test_queries:
        total += 1
        question_type = item["type"]
        query = item["query"]

        result = query_manager.query(query, similarity_top_k=8)
        answer = str(result.get("answer", "")).strip()
        issues = evaluate_quality(question_type, query, result)
        blocked_count = int(result.get("domain_drift_blocked_count", 0))
        domain_drift_blocked_count += blocked_count

        if "poisoned_refinance_answer" in issues:
            poisoned_answer_count += 1

        status = "PASS" if not issues else "FAIL"
        if not issues:
            passed += 1
        else:
            failed += 1

        print(f"[{status}] ({question_type}) Q: {query}")
        print(f"       A: {answer[:240]}{'...' if len(answer) > 240 else ''}")
        if result.get("sources"):
            first = result["sources"][0]
            meta = first.get("metadata", {})
            print(
                "       Source: "
                f"{meta.get('title', 'N/A')[:80]} | "
                f"category={meta.get('category', 'N/A')} | "
                f"score={first.get('score', 0)}"
            )
        print(
            "       Stats: "
            f"retrieved={result.get('retrieved_node_count', 0)} "
            f"validated={result.get('validated_node_count', 0)} "
            f"blocked={blocked_count}"
        )
        if issues:
            print(f"       Issues: {', '.join(issues)}")
        print()

    print("=" * 72)
    print("Quality Summary")
    print("=" * 72)
    print(f"Total queries: {total}")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    print(f"Quality pass rate: {(passed / total * 100.0):.1f}%")
    print(f"poisoned_answer_count: {poisoned_answer_count}")
    print(f"quarantined_docs_count: {quarantine_metrics['quarantined_docs_count']}")
    print(f"domain_drift_blocked_count: {domain_drift_blocked_count}")


if __name__ == "__main__":
    main()
