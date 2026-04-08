"""Generate the Chapter 4 ablation table from advisor evaluation results.

Reads ``data/eval/advisor_results.jsonl`` and produces:

  1. **Console table** — quick scan
  2. **data/eval/ablation_table.md** — Markdown for thesis Chapter 4
  3. **data/eval/ablation_table.tex** — LaTeX longtable for direct paste
  4. **data/eval/ablation_table.csv** — CSV for plotting in Excel/Pandas

Metrics in the main table:
  - n            number of test cases evaluated
  - latency      mean ± std (s)
  - checks       avg requirement checks identified
  - pass / fail  avg pass / fail counts
  - actions      avg recommended actions
  - sources      avg sources cited
  - kw_recall    keyword coverage (factual questions)
  - verdict_acc  verdict-agreement on labelled cases
  - issup        Self-RAG IsSup score (A3 only)

Run::

    python eval/generate_ablation.py

Optional flags::

    --by-type           emit one table per question type
    --min-records 30    require at least N records per approach before reporting
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).parent.parent
RESULTS_PATH = ROOT / "data" / "eval" / "advisor_results.jsonl"
OUT_MD = ROOT / "data" / "eval" / "ablation_table.md"
OUT_TEX = ROOT / "data" / "eval" / "ablation_table.tex"
OUT_CSV = ROOT / "data" / "eval" / "ablation_table.csv"

APPROACH_LABELS = {
    "a1": "A1 — Profile-conditioned (single-hop)",
    "a2": "A2 — A1 + Multi-hop decomposition",
    "a3": "A3 — A2 + Self-RAG reflection",
}


def load_results() -> List[Dict]:
    if not RESULTS_PATH.exists():
        sys.exit(f"Results file missing: {RESULTS_PATH}")
    out: List[Dict] = []
    with RESULTS_PATH.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return out


def stat_pair(xs: Sequence[float]) -> str:
    if not xs:
        return "—"
    if len(xs) == 1:
        return f"{xs[0]:.1f}"
    mean = statistics.fmean(xs)
    sd = statistics.pstdev(xs) if len(xs) > 1 else 0.0
    return f"{mean:.1f} ± {sd:.1f}"


def _safe_mean(xs: Sequence[float]) -> Optional[float]:
    return statistics.fmean(xs) if xs else None


def _fmt(v: Optional[float], decimals: int = 2) -> str:
    if v is None:
        return "—"
    return f"{v:.{decimals}f}"


def keyword_recall_one(record: Dict) -> Optional[float]:
    expected = record.get("expected_keywords") or []
    if not expected:
        return None
    parts = [record.get("verdict_summary") or ""]
    for c in record.get("requirement_checks") or []:
        parts.append(str(c.get("requirement", "")))
        parts.append(str(c.get("user_value", "")))
        parts.append(str(c.get("explanation", "")))
    for a in record.get("recommended_actions") or []:
        parts.append(str(a))
    haystack = "\n".join(parts).lower()
    return sum(1 for k in expected if str(k).lower() in haystack) / len(expected)


def compute_row(records: List[Dict]) -> Dict[str, Any]:
    """Compute one row of the ablation table from a list of result records."""
    if not records:
        return {"n": 0}

    latencies = [r.get("elapsed_s") or 0.0 for r in records]
    n_checks = [r.get("n_checks") or 0 for r in records]
    n_pass = [r.get("n_pass") or 0 for r in records]
    n_fail = [r.get("n_fail") or 0 for r in records]
    n_unknown = [r.get("n_unknown") or 0 for r in records]
    n_actions = [r.get("n_actions") or 0 for r in records]
    n_sources = [r.get("n_sources") or 0 for r in records]

    # keyword recall — only on records with expected_keywords
    kw_vals = [v for v in (keyword_recall_one(r) for r in records) if v is not None]

    # verdict agreement — only on records with expected_verdict
    agreements: List[int] = []
    for r in records:
        ev = r.get("expected_verdict")
        if ev:
            agreements.append(1 if r.get("verdict") == ev else 0)

    # IsSup — only when reasoning_trace contains a score
    issups: List[float] = []
    sub_q_counts: List[int] = []
    for r in records:
        trace = r.get("reasoning_trace") or {}
        s = trace.get("issup_score")
        if isinstance(s, (int, float)):
            issups.append(float(s))
        subs = trace.get("sub_questions") or []
        if isinstance(subs, list) and subs:
            sub_q_counts.append(len(subs))

    return {
        "n": len(records),
        "latency_str": stat_pair(latencies),
        "latency_mean": _safe_mean(latencies),
        "checks_str": stat_pair(n_checks),
        "checks_mean": _safe_mean(n_checks),
        "pass_mean": _safe_mean(n_pass),
        "fail_mean": _safe_mean(n_fail),
        "unknown_mean": _safe_mean(n_unknown),
        "actions_mean": _safe_mean(n_actions),
        "sources_mean": _safe_mean(n_sources),
        "kw_recall": _safe_mean(kw_vals),
        "kw_recall_n": len(kw_vals),
        "verdict_acc": _safe_mean(agreements),
        "verdict_acc_n": len(agreements),
        "issup_mean": _safe_mean(issups),
        "issup_n": len(issups),
        "sub_q_mean": _safe_mean(sub_q_counts),
    }


def render_md_table(rows_by_approach: Dict[str, Dict[str, Any]], title: str) -> str:
    lines: List[str] = [f"### {title}", ""]
    lines.append(
        "| Approach | N | Latency (s) | Checks | Pass | Fail | Unkn | Actions | Sources | Keyword Recall | Verdict Acc. | IsSup |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for code in ("a1", "a2", "a3"):
        row = rows_by_approach.get(code)
        if not row or row.get("n", 0) == 0:
            continue
        label = APPROACH_LABELS.get(code, code)
        lines.append(
            f"| {label} | {row['n']} | {row['latency_str']} | {row['checks_str']} | "
            f"{_fmt(row['pass_mean'])} | {_fmt(row['fail_mean'])} | {_fmt(row['unknown_mean'])} | "
            f"{_fmt(row['actions_mean'])} | {_fmt(row['sources_mean'])} | "
            f"{_fmt(row['kw_recall'])} (n={row['kw_recall_n']}) | "
            f"{_fmt(row['verdict_acc'])} (n={row['verdict_acc_n']}) | "
            f"{_fmt(row['issup_mean'])} |"
        )
    lines.append("")
    return "\n".join(lines)


def render_console_table(rows: Dict[str, Dict[str, Any]], title: str) -> None:
    print(f"\n{title}")
    print("=" * 110)
    cols = [
        ("approach", 35),
        ("n", 5),
        ("latency", 14),
        ("checks", 14),
        ("pass", 7),
        ("fail", 7),
        ("act", 7),
        ("src", 7),
        ("kw_rec", 9),
        ("verd_acc", 10),
    ]
    print("".join(f"{c[0]:<{c[1]}}" for c in cols))
    print("-" * 110)
    for code in ("a1", "a2", "a3"):
        row = rows.get(code)
        if not row or row.get("n", 0) == 0:
            continue
        label = APPROACH_LABELS.get(code, code)
        print(
            f"{label[:34]:<35}"
            f"{row['n']:<5}"
            f"{row['latency_str']:<14}"
            f"{row['checks_str']:<14}"
            f"{_fmt(row['pass_mean']):<7}"
            f"{_fmt(row['fail_mean']):<7}"
            f"{_fmt(row['actions_mean']):<7}"
            f"{_fmt(row['sources_mean']):<7}"
            f"{_fmt(row['kw_recall']):<9}"
            f"{_fmt(row['verdict_acc']):<10}"
        )


def render_latex(rows: Dict[str, Dict[str, Any]]) -> str:
    """Emit a thesis-ready longtable. Caption + label included."""
    head = r"""\begin{table}[h]
\centering
\caption{Ablation study of advisor reasoning approaches on 100 test cases.}
\label{tab:advisor-ablation}
\small
\begin{tabular}{lrrrrrrrrr}
\toprule
Approach & N & Latency (s) & Checks & Pass & Fail & Actions & Sources & Keyword & Verdict \\
         &   &             &        &      &      &         &         & Recall  & Acc.    \\
\midrule
"""
    body_lines: List[str] = []
    for code in ("a1", "a2", "a3"):
        row = rows.get(code)
        if not row or row.get("n", 0) == 0:
            continue
        label = APPROACH_LABELS.get(code, code).replace("&", r"\&")
        body_lines.append(
            f"{label} & {row['n']} & {row['latency_str']} & {row['checks_str']} & "
            f"{_fmt(row['pass_mean'])} & {_fmt(row['fail_mean'])} & "
            f"{_fmt(row['actions_mean'])} & {_fmt(row['sources_mean'])} & "
            f"{_fmt(row['kw_recall'])} & {_fmt(row['verdict_acc'])} \\\\"
        )
    tail = r"""\bottomrule
\end{tabular}
\end{table}
"""
    return head + "\n".join(body_lines) + "\n" + tail


def render_csv(rows: Dict[str, Dict[str, Any]]) -> str:
    header = (
        "approach,n,latency_mean,checks_mean,pass,fail,unknown,actions,sources,"
        "kw_recall,kw_recall_n,verdict_acc,verdict_acc_n,issup_mean,issup_n,sub_q_mean\n"
    )
    out = [header]
    for code in ("a1", "a2", "a3"):
        row = rows.get(code)
        if not row or row.get("n", 0) == 0:
            continue
        label = APPROACH_LABELS.get(code, code)

        def f(v):
            return "" if v is None else f"{v:.4f}"

        out.append(
            ",".join(
                [
                    f'"{label}"',
                    str(row["n"]),
                    f(row["latency_mean"]),
                    f(row["checks_mean"]),
                    f(row["pass_mean"]),
                    f(row["fail_mean"]),
                    f(row["unknown_mean"]),
                    f(row["actions_mean"]),
                    f(row["sources_mean"]),
                    f(row["kw_recall"]),
                    str(row["kw_recall_n"]),
                    f(row["verdict_acc"]),
                    str(row["verdict_acc_n"]),
                    f(row["issup_mean"]),
                    str(row["issup_n"]),
                    f(row["sub_q_mean"]),
                ]
            )
            + "\n"
        )
    return "".join(out)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--by-type", action="store_true", help="Emit per-question-type tables")
    parser.add_argument("--min-records", type=int, default=1)
    args = parser.parse_args()

    records = load_results()
    print(f"Loaded {len(records)} records")

    by_approach: Dict[str, List[Dict]] = defaultdict(list)
    for r in records:
        by_approach[r.get("approach", "?")].append(r)

    overall_rows = {ap: compute_row(rs) for ap, rs in by_approach.items()}

    # Drop approaches with too few records
    overall_rows = {
        ap: row for ap, row in overall_rows.items() if row.get("n", 0) >= args.min_records
    }

    render_console_table(overall_rows, "OVERALL ABLATION (100 questions)")

    md_chunks = ["# Advisor Ablation Study\n", f"Total records analysed: **{len(records)}**\n"]
    md_chunks.append(render_md_table(overall_rows, "Overall"))

    if args.by_type:
        by_at: Dict[tuple, List[Dict]] = defaultdict(list)
        for r in records:
            by_at[(r.get("approach"), r.get("question_type"))].append(r)
        qtypes = sorted({k[1] for k in by_at})
        for qt in qtypes:
            rows = {
                ap: compute_row(by_at.get((ap, qt), []))
                for ap in ("a1", "a2", "a3")
            }
            rows = {ap: row for ap, row in rows.items() if row.get("n", 0) >= 1}
            if not rows:
                continue
            render_console_table(rows, f"BY TYPE: {qt}")
            md_chunks.append(render_md_table(rows, f"By type: {qt}"))

    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUT_MD.write_text("\n".join(md_chunks), encoding="utf-8")
    OUT_TEX.write_text(render_latex(overall_rows), encoding="utf-8")
    OUT_CSV.write_text(render_csv(overall_rows), encoding="utf-8")

    print(f"\nWrote:")
    print(f"  {OUT_MD}")
    print(f"  {OUT_TEX}")
    print(f"  {OUT_CSV}")


if __name__ == "__main__":
    main()
