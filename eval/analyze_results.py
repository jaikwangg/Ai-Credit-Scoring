"""Analyze advisor evaluation results and emit thesis-ready tables.

Reads ``data/eval/advisor_results.jsonl`` and computes:

  Per-approach metrics
  --------------------
  - n_total            number of test cases evaluated
  - latency_mean / median / p95
  - avg_n_checks       avg requirement checks identified
  - avg_n_pass / fail / unknown
  - avg_n_actions      avg recommended actions
  - avg_n_sources      avg sources cited
  - keyword_recall     fraction of expected_keywords mentioned in output
  - verdict_agreement  fraction matching expected_verdict (only labelled cases)
  - issup_score_mean   (A3 only) avg Self-RAG groundedness 1-5
  - sub_q_mean         (A2/A3) avg sub-questions per query

  Per-question-type breakdown
  ---------------------------
  Same metrics but bucketed by factual / single_eligibility / multi_eligibility / advice

  Outputs
  -------
  - Pretty console table
  - data/eval/summary.json
  - data/eval/summary.md  (Markdown table for the thesis)

Usage::

    python eval/analyze_results.py
"""
from __future__ import annotations

import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence
import sys

sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).parent.parent
RESULTS_PATH = ROOT / "data" / "eval" / "advisor_results.jsonl"
SUMMARY_JSON = ROOT / "data" / "eval" / "summary.json"
SUMMARY_MD = ROOT / "data" / "eval" / "summary.md"


def load_results() -> List[Dict]:
    if not RESULTS_PATH.exists():
        sys.exit(f"Results not found: {RESULTS_PATH}")
    out: List[Dict] = []
    with RESULTS_PATH.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def safe_mean(xs: Sequence[float]) -> float:
    return round(statistics.fmean(xs), 3) if xs else 0.0


def safe_median(xs: Sequence[float]) -> float:
    return round(statistics.median(xs), 3) if xs else 0.0


def safe_p95(xs: Sequence[float]) -> float:
    if not xs:
        return 0.0
    s = sorted(xs)
    idx = max(0, min(len(s) - 1, int(round(0.95 * (len(s) - 1)))))
    return round(s[idx], 3)


def keyword_recall_one(record: Dict) -> Optional[float]:
    """Fraction of expected keywords that appear in the verdict_summary +
    requirement_checks + recommended_actions text. None if no keywords expected."""
    expected = record.get("expected_keywords") or []
    if not expected:
        return None
    haystack_parts = [
        record.get("verdict_summary") or "",
    ]
    for c in record.get("requirement_checks") or []:
        haystack_parts.append(str(c.get("requirement", "")))
        haystack_parts.append(str(c.get("user_value", "")))
        haystack_parts.append(str(c.get("explanation", "")))
    for a in record.get("recommended_actions") or []:
        haystack_parts.append(str(a))
    haystack = "\n".join(haystack_parts).lower()
    hits = sum(1 for k in expected if str(k).lower() in haystack)
    return hits / len(expected)


def verdict_agreement_one(record: Dict) -> Optional[bool]:
    expected = record.get("expected_verdict")
    if not expected:
        return None
    actual = record.get("verdict")
    return actual == expected


def aggregate(records: List[Dict]) -> Dict[str, Any]:
    if not records:
        return {"n": 0}

    latencies = [r.get("elapsed_s") or 0.0 for r in records]
    n_checks = [r.get("n_checks") or 0 for r in records]
    n_pass = [r.get("n_pass") or 0 for r in records]
    n_fail = [r.get("n_fail") or 0 for r in records]
    n_unknown = [r.get("n_unknown") or 0 for r in records]
    n_actions = [r.get("n_actions") or 0 for r in records]
    n_sources = [r.get("n_sources") or 0 for r in records]

    keyword_recalls = [
        v for v in (keyword_recall_one(r) for r in records) if v is not None
    ]
    agreements = [
        v for v in (verdict_agreement_one(r) for r in records) if v is not None
    ]

    issup_scores: List[float] = []
    sub_q_counts: List[int] = []
    for r in records:
        trace = r.get("reasoning_trace") or {}
        score = trace.get("issup_score")
        if isinstance(score, (int, float)):
            issup_scores.append(float(score))
        subs = trace.get("sub_questions")
        if isinstance(subs, list) and subs:
            sub_q_counts.append(len(subs))

    return {
        "n": len(records),
        "latency_mean_s": safe_mean(latencies),
        "latency_median_s": safe_median(latencies),
        "latency_p95_s": safe_p95(latencies),
        "avg_n_checks": safe_mean(n_checks),
        "avg_n_pass": safe_mean(n_pass),
        "avg_n_fail": safe_mean(n_fail),
        "avg_n_unknown": safe_mean(n_unknown),
        "avg_n_actions": safe_mean(n_actions),
        "avg_n_sources": safe_mean(n_sources),
        "keyword_recall": safe_mean(keyword_recalls) if keyword_recalls else None,
        "keyword_recall_n": len(keyword_recalls),
        "verdict_agreement": safe_mean([1.0 if a else 0.0 for a in agreements]) if agreements else None,
        "verdict_agreement_n": len(agreements),
        "issup_score_mean": safe_mean(issup_scores) if issup_scores else None,
        "issup_score_n": len(issup_scores),
        "sub_q_mean": safe_mean(sub_q_counts) if sub_q_counts else None,
        "sub_q_n": len(sub_q_counts),
    }


def main() -> None:
    records = load_results()
    print(f"Loaded {len(records)} result records from {RESULTS_PATH.name}\n")

    by_approach: Dict[str, List[Dict]] = defaultdict(list)
    for r in records:
        by_approach[r.get("approach", "?")].append(r)

    by_approach_type: Dict[tuple, List[Dict]] = defaultdict(list)
    for r in records:
        by_approach_type[(r.get("approach", "?"), r.get("question_type", "?"))].append(r)

    summary = {
        "overall": {ap: aggregate(rs) for ap, rs in by_approach.items()},
        "by_question_type": {
            f"{ap}::{qt}": aggregate(rs) for (ap, qt), rs in by_approach_type.items()
        },
    }

    SUMMARY_JSON.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY_JSON.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {SUMMARY_JSON}")

    # ── Console table: per-approach overall ──
    print("\n" + "=" * 100)
    print("OVERALL — per approach")
    print("=" * 100)
    cols = [
        ("n", "n", 5),
        ("latency_mean_s", "lat_avg", 8),
        ("latency_p95_s", "lat_p95", 8),
        ("avg_n_checks", "checks", 7),
        ("avg_n_pass", "pass", 6),
        ("avg_n_fail", "fail", 6),
        ("avg_n_unknown", "unkn", 6),
        ("avg_n_actions", "act", 6),
        ("avg_n_sources", "src", 6),
        ("keyword_recall", "kw_rec", 8),
        ("verdict_agreement", "verd_ok", 9),
        ("issup_score_mean", "issup", 7),
        ("sub_q_mean", "subq", 6),
    ]
    header = f"{'approach':18}" + "".join(f"{label:>{w}}" for _, label, w in cols)
    print(header)
    print("-" * len(header))
    for ap in sorted(summary["overall"]):
        row = summary["overall"][ap]
        line = f"{ap:18}"
        for key, _, w in cols:
            v = row.get(key)
            if v is None:
                line += f"{'-':>{w}}"
            elif isinstance(v, float):
                line += f"{v:>{w}.2f}"
            else:
                line += f"{v:>{w}}"
        print(line)

    # ── Console table: per-question-type for each approach ──
    print("\n" + "=" * 100)
    print("BY QUESTION TYPE")
    print("=" * 100)
    qtypes = sorted({k.split("::")[1] for k in summary["by_question_type"]})
    approaches = sorted({k.split("::")[0] for k in summary["by_question_type"]})
    for qt in qtypes:
        print(f"\n[{qt}]")
        sub_header = f"{'approach':18}" + "".join(f"{label:>{w}}" for _, label, w in cols)
        print(sub_header)
        print("-" * len(sub_header))
        for ap in approaches:
            key = f"{ap}::{qt}"
            row = summary["by_question_type"].get(key)
            if not row:
                continue
            line = f"{ap:18}"
            for k, _, w in cols:
                v = row.get(k)
                if v is None:
                    line += f"{'-':>{w}}"
                elif isinstance(v, float):
                    line += f"{v:>{w}.2f}"
                else:
                    line += f"{v:>{w}}"
            print(line)

    # ── Markdown summary for thesis ──
    md_lines: List[str] = []
    md_lines.append("# Advisor Evaluation Summary")
    md_lines.append("")
    md_lines.append(f"Total records: **{len(records)}**")
    md_lines.append("")
    md_lines.append("## Overall metrics by approach")
    md_lines.append("")
    md_lines.append(
        "| Approach | N | Latency (s) | Checks | Pass | Fail | Unkn | Actions | Sources | Keyword Recall | Verdict Agreement | IsSup | Sub-Q |"
    )
    md_lines.append(
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|"
    )
    for ap in sorted(summary["overall"]):
        row = summary["overall"][ap]

        def fmt(v: Any) -> str:
            if v is None:
                return "—"
            if isinstance(v, float):
                return f"{v:.2f}"
            return str(v)

        md_lines.append(
            "| {ap} | {n} | {lat} | {ch} | {p} | {f} | {u} | {a} | {s} | {kr} | {va} | {is_} | {sq} |".format(
                ap=ap,
                n=row["n"],
                lat=f"{fmt(row['latency_mean_s'])} (p95 {fmt(row['latency_p95_s'])})",
                ch=fmt(row["avg_n_checks"]),
                p=fmt(row["avg_n_pass"]),
                f=fmt(row["avg_n_fail"]),
                u=fmt(row["avg_n_unknown"]),
                a=fmt(row["avg_n_actions"]),
                s=fmt(row["avg_n_sources"]),
                kr=fmt(row["keyword_recall"]),
                va=fmt(row["verdict_agreement"]),
                is_=fmt(row["issup_score_mean"]),
                sq=fmt(row["sub_q_mean"]),
            )
        )
    md_lines.append("")

    md_lines.append("## Per-question-type metrics")
    md_lines.append("")
    for qt in qtypes:
        md_lines.append(f"### {qt}")
        md_lines.append("")
        md_lines.append("| Approach | N | Latency | Checks | Keyword Recall | Verdict Agreement |")
        md_lines.append("|---|---|---|---|---|---|")
        for ap in approaches:
            key = f"{ap}::{qt}"
            row = summary["by_question_type"].get(key)
            if not row:
                continue

            def f2(v):
                return "—" if v is None else (f"{v:.2f}" if isinstance(v, float) else str(v))

            md_lines.append(
                f"| {ap} | {row['n']} | {f2(row['latency_mean_s'])}s | {f2(row['avg_n_checks'])} | {f2(row['keyword_recall'])} | {f2(row['verdict_agreement'])} |"
            )
        md_lines.append("")

    SUMMARY_MD.write_text("\n".join(md_lines), encoding="utf-8")
    print(f"\nWrote {SUMMARY_MD}")


if __name__ == "__main__":
    main()
