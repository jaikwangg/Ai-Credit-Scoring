"""Similarity distribution reporting from retrieval logs."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List

from src.rag.validator import NO_ANSWER_MESSAGE


def _as_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def load_retrieval_logs(log_path: Path) -> List[Dict[str, Any]]:
    if not log_path.exists():
        return []

    rows: List[Dict[str, Any]] = []
    with log_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def build_similarity_stats(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    top1_scores: List[float] = []
    topk_means: List[float] = []
    top12_gaps: List[float] = []

    fallback_len = len(NO_ANSWER_MESSAGE)
    no_answer_count = 0

    for row in rows:
        retrieved = row.get("retrieved", []) or []
        scores = [_as_float(item.get("score")) for item in retrieved]
        scores = [s for s in scores if s is not None]

        if scores:
            top1_scores.append(scores[0])
            topk_means.append(mean(scores))
        if len(scores) >= 2:
            top12_gaps.append(scores[0] - scores[1])

        is_no_answer = row.get("is_no_answer")
        if is_no_answer is None:
            is_no_answer = row.get("final_answer_length") == fallback_len
        if bool(is_no_answer):
            no_answer_count += 1

    return {
        "total_queries": len(rows),
        "mean_top1_score": mean(top1_scores) if top1_scores else 0.0,
        "mean_topk_score": mean(topk_means) if topk_means else 0.0,
        "mean_top1_top2_gap": mean(top12_gaps) if top12_gaps else 0.0,
        "count_no_answer": no_answer_count,
    }


def write_report_txt(stats: Dict[str, Any], output_path: Path) -> None:
    lines = [
        "Similarity Distribution Report",
        "==============================",
        f"Total queries: {stats['total_queries']}",
        f"Mean top1 score: {stats['mean_top1_score']:.4f}",
        f"Mean topK score: {stats['mean_topk_score']:.4f}",
        f"Mean top1-top2 gap: {stats['mean_top1_top2_gap']:.4f}",
        f"No-answer count: {stats['count_no_answer']}",
    ]
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_report_csv(stats: Dict[str, Any], output_path: Path) -> None:
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "value"])
        writer.writerow(["total_queries", stats["total_queries"]])
        writer.writerow(["mean_top1_score", f"{stats['mean_top1_score']:.6f}"])
        writer.writerow(["mean_topk_score", f"{stats['mean_topk_score']:.6f}"])
        writer.writerow(["mean_top1_top2_gap", f"{stats['mean_top1_top2_gap']:.6f}"])
        writer.writerow(["count_no_answer", stats["count_no_answer"]])


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate similarity distribution report.")
    parser.add_argument(
        "--log-path",
        default="logs/retrieval_logs.jsonl",
        help="Path to retrieval logs JSONL file",
    )
    parser.add_argument("--report-txt", default="report.txt", help="Output TXT report path")
    parser.add_argument("--report-csv", default="report.csv", help="Output CSV report path")
    args = parser.parse_args()

    logs = load_retrieval_logs(Path(args.log_path))
    stats = build_similarity_stats(logs)

    txt_path = Path(args.report_txt)
    csv_path = Path(args.report_csv)
    write_report_txt(stats, txt_path)
    write_report_csv(stats, csv_path)

    print(f"Processed {len(logs)} log rows from {args.log_path}")
    print(f"Wrote report: {txt_path}")
    print(f"Wrote report: {csv_path}")


if __name__ == "__main__":
    main()
