"""Audit structured scraped documents for cleaning quality."""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path
from typing import Dict, List

from src.document_parser import (
    StructuredDocumentParser,
    analyze_scraped_text,
    clean_scraped_text,
    extract_effective_year,
)

NOISE_REVIEW_THRESHOLD = 0.20
TABLE_LIKENESS_THRESHOLD = 0.30
ROW_CONVERSION_THRESHOLD = 0.45


def _extract_header_field(text: str, label: str) -> str:
    match = re.search(rf"^{re.escape(label)}:\s*(.+)$", text, flags=re.MULTILINE)
    if not match:
        return ""
    return match.group(1).strip()


def _extract_main_content(text: str) -> str:
    match = re.search(r"FULL CLEANED TEXT CONTENT\n(.*)", text, flags=re.DOTALL)
    if match:
        return match.group(1).strip()
    return text


def _audit_document(file_path: Path) -> Dict[str, object]:
    raw = file_path.read_text(encoding="utf-8")

    title = _extract_header_field(raw, "TITLE")
    category = _extract_header_field(raw, "CATEGORY") or "unknown"
    main_content = _extract_main_content(raw)

    quality = analyze_scraped_text(main_content)
    cleaned = clean_scraped_text(main_content)
    inferred_doc_kind = StructuredDocumentParser._infer_doc_kind(title, file_path.name)

    effective_year = extract_effective_year(f"{title}\n{cleaned}")
    has_effective_year = bool(effective_year)

    high_table_low_conversion = (
        quality["table_likeness_score"] > TABLE_LIKENESS_THRESHOLD
        and quality["row_conversion_score"] < ROW_CONVERSION_THRESHOLD
    )

    missing_year_for_rate_sheet = inferred_doc_kind == "rate_sheet" and not has_effective_year

    needs_review = (
        quality["noise_line_ratio"] > NOISE_REVIEW_THRESHOLD
        or missing_year_for_rate_sheet
        or high_table_low_conversion
    )

    return {
        "file_name": file_path.name,
        "char_count": int(quality["char_count"]),
        "line_count": int(quality["line_count"]),
        "noise_line_ratio": round(float(quality["noise_line_ratio"]), 4),
        "table_likeness_score": round(float(quality["table_likeness_score"]), 4),
        "row_conversion_score": round(float(quality["row_conversion_score"]), 4),
        "duplicate_boilerplate_score": round(float(quality["duplicate_boilerplate_score"]), 4),
        "has_effective_year": has_effective_year,
        "effective_year": effective_year,
        "inferred_doc_kind": inferred_doc_kind,
        "category": category,
        "needs_review": needs_review,
    }


def _severity_score(record: Dict[str, object]) -> float:
    noise = float(record["noise_line_ratio"])
    duplicate = float(record["duplicate_boilerplate_score"])
    table = float(record["table_likeness_score"])
    row_conversion = float(record["row_conversion_score"])

    missing_year_penalty = 0.35 if (record["inferred_doc_kind"] == "rate_sheet" and not record["has_effective_year"]) else 0.0
    table_penalty = table * (1.0 - row_conversion)

    return (0.45 * noise) + (0.25 * duplicate) + (0.30 * table_penalty) + missing_year_penalty


def _write_csv(records: List[Dict[str, object]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fields = [
        "file_name",
        "char_count",
        "line_count",
        "noise_line_ratio",
        "table_likeness_score",
        "row_conversion_score",
        "duplicate_boilerplate_score",
        "has_effective_year",
        "effective_year",
        "inferred_doc_kind",
        "category",
        "needs_review",
    ]

    with output_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fields)
        writer.writeheader()
        for record in records:
            writer.writerow(record)


def _write_summary(records: List[Dict[str, object]], summary_path: Path) -> None:
    ranked = sorted(records, key=_severity_score, reverse=True)
    top = ranked[:10]

    total = len(records)
    flagged = sum(1 for record in records if record["needs_review"])

    lines = [
        "Document Audit Summary",
        "======================",
        f"Total docs: {total}",
        f"Needs review: {flagged}",
        "",
        "Top 10 docs to review:",
    ]

    for idx, record in enumerate(top, start=1):
        lines.append(
            (
                f"{idx:02d}. {record['file_name']} | severity={_severity_score(record):.3f} | "
                f"noise={record['noise_line_ratio']} | table={record['table_likeness_score']} | "
                f"row_conv={record['row_conversion_score']} | dup={record['duplicate_boilerplate_score']} | "
                f"year={record['has_effective_year']} | kind={record['inferred_doc_kind']} | "
                f"needs_review={record['needs_review']}"
            )
        )

    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_audit(input_dir: Path, output_csv: Path) -> None:
    files = sorted(input_dir.glob("*.txt"))
    records = [_audit_document(path) for path in files]

    _write_csv(records, output_csv)

    summary_path = output_csv.with_name(f"{output_csv.stem}_summary.txt")
    _write_summary(records, summary_path)

    print(f"Audited {len(records)} documents from {input_dir}")
    print(f"CSV report: {output_csv}")
    print(f"Summary report: {summary_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit structured banking documents for RAG readiness.")
    parser.add_argument("--input", required=True, help="Input directory containing structured .txt docs")
    parser.add_argument("--out", required=True, help="Output CSV report path")
    args = parser.parse_args()

    run_audit(Path(args.input), Path(args.out))


if __name__ == "__main__":
    main()
