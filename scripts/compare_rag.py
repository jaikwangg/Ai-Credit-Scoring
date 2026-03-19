"""
Baseline vs Self-RAG Side-by-Side Comparison
=============================================
รัน test suite เดียวกันทั้งสอง mode แล้วแสดงตาราง diff

Usage:
    uv run python scripts/compare_rag.py 2>$null
    uv run python scripts/compare_rag.py --judge 2>$null   # + LLM-as-judge
    uv run python scripts/compare_rag.py --verbose 2>$null
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import List, Optional

try:
    sys.stdout.reconfigure(encoding="utf-8")
except AttributeError:
    pass
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.evaluate_rag import (
    TEST_CASES,
    RAGCaseReport,
    evaluate_all,
    init_rag_manager,
    init_query_fn,
    NO_ANSWER,
)


def _pct(val: Optional[float]) -> str:
    return f"{val:.0%}" if val is not None else "N/A"

def _score(val: Optional[float]) -> str:
    return f"{val:.2f}/5" if val is not None else "N/A"

def _delta(a: Optional[float], b: Optional[float], fmt=".2f") -> str:
    if a is None or b is None:
        return ""
    d = b - a
    sign = "+" if d >= 0 else ""
    return f"({sign}{d:{fmt}})"


def print_comparison(base: List[RAGCaseReport], self_rag: List[RAGCaseReport]) -> None:
    # ── aggregate metrics ──────────────────────────────────────────────────────
    def agg(reports: List[RAGCaseReport]):
        total_checks = sum(r.total for r in reports)
        total_passed = sum(r.passed for r in reports)
        answered   = sum(1 for r in reports if r.case.expect_answer
                         and r.answer.strip() and r.answer.strip() != NO_ANSWER)
        exp_ans    = sum(1 for r in reports if r.case.expect_answer)
        router_ok  = sum(1 for r in reports if r.router_label == r.case.expected_route)
        latencies  = [r.elapsed_s for r in reports]
        prec_vals  = [r.precision_at_k for r in reports if r.precision_at_k is not None]
        g_vals     = [r.groundedness_score for r in reports if r.groundedness_score is not None]
        r_vals     = [r.relevance_score    for r in reports if r.relevance_score    is not None]
        return {
            "overall":     total_passed / total_checks if total_checks else 0,
            "answer_rate": answered / exp_ans if exp_ans else 0,
            "router_acc":  router_ok / len(reports) if reports else 0,
            "mean_latency":sum(latencies) / len(latencies) if latencies else 0,
            "mean_prec":   sum(prec_vals) / len(prec_vals) if prec_vals else None,
            "mean_g":      sum(g_vals) / len(g_vals) if g_vals else None,
            "mean_r":      sum(r_vals) / len(r_vals) if r_vals else None,
        }

    b = agg(base)
    s = agg(self_rag)

    W = 65
    print(f"\n{'='*W}")
    print(f"  COMPARISON: Baseline RAG  vs  Self-RAG")
    print(f"{'='*W}")
    print(f"  {'Metric':<32} {'Baseline':>10}  {'Self-RAG':>10}  {'Delta':>10}")
    print(f"  {'-'*60}")

    rows = [
        ("Router Accuracy",      _pct(b["router_acc"]),   _pct(s["router_acc"]),   _delta(b["router_acc"],   s["router_acc"],   ".0%")),
        ("Answer Rate",          _pct(b["answer_rate"]),  _pct(s["answer_rate"]),  _delta(b["answer_rate"],  s["answer_rate"],  ".0%")),
        ("Mean Precision@K",     _score(b["mean_prec"]).replace("/5","") if b["mean_prec"] else "N/A",
                                 _score(s["mean_prec"]).replace("/5","") if s["mean_prec"] else "N/A",
                                 _delta(b["mean_prec"],   s["mean_prec"])),
        ("Groundedness (LLM)",   _score(b["mean_g"]),     _score(s["mean_g"]),     _delta(b["mean_g"],   s["mean_g"])),
        ("Answer Relevance (LLM)",_score(b["mean_r"]),    _score(s["mean_r"]),     _delta(b["mean_r"],   s["mean_r"])),
        ("Overall Checks",       _pct(b["overall"]),      _pct(s["overall"]),      _delta(b["overall"],      s["overall"],      ".0%")),
        ("Mean Latency",         f"{b['mean_latency']:.1f}s", f"{s['mean_latency']:.1f}s",
                                 f"(+{s['mean_latency']-b['mean_latency']:.1f}s)"),
    ]
    for label, bv, sv, dv in rows:
        print(f"  {label:<32} {bv:>10}  {sv:>10}  {dv:>10}")

    # ── per-case diff ──────────────────────────────────────────────────────────
    print(f"\n{'='*W}")
    print(f"  PER-CASE  (Baseline → Self-RAG)")
    print(f"{'='*W}")
    print(f"  {'Description':<46} {'Base':>5}  {'SRAG':>5}  {'Δ':>4}")
    print(f"  {'-'*62}")

    for b_rep, s_rep in zip(base, self_rag):
        label = b_rep.case.description or b_rep.case.query[:40]
        b_ok = "PASS" if b_rep.score == 1.0 else ("WARN" if b_rep.score >= 0.7 else "FAIL")
        s_ok = "PASS" if s_rep.score == 1.0 else ("WARN" if s_rep.score >= 0.7 else "FAIL")
        delta = ""
        if b_rep.score != s_rep.score:
            d = s_rep.score - b_rep.score
            delta = f"{'+' if d>0 else ''}{d:+.0%}"
        print(f"  {label:<46} {b_ok:>5}  {s_ok:>5}  {delta:>4}")

    print()
    # ── Self-RAG trace summary ─────────────────────────────────────────────────
    traces = []
    for r in self_rag:
        t = getattr(r, "_raw_result", {})  # not stored; skip if unavailable
    print(f"  Self-RAG adds ~{s['mean_latency']-b['mean_latency']:.1f}s latency per query")
    print(f"  Answer rate change: {_pct(b['answer_rate'])} → {_pct(s['answer_rate'])}  {_delta(b['answer_rate'], s['answer_rate'], '.0%')}")
    print()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--judge",   action="store_true", help="LLM-as-judge scores")
    parser.add_argument("--verbose", action="store_true", help="Show per-query details")
    args = parser.parse_args()

    manager = init_rag_manager()

    if args.judge:
        from src.query_engine import _build_llm
        from llama_index.core.settings import Settings
        Settings.llm = _build_llm()
        print("LLM-as-judge enabled\n", flush=True)

    # ── Baseline ──
    print("=" * 50, flush=True)
    print("BASELINE RAG", flush=True)
    print("=" * 50, flush=True)
    base_fn = init_query_fn(manager, use_self_rag=False)
    base_reports = evaluate_all(base_fn, verbose=args.verbose, use_judge=args.judge)

    # ── Self-RAG ──
    print("\n" + "=" * 50, flush=True)
    print("SELF-RAG ([Retrieve] + [IsRel] + [IsSup] + [IsGen])", flush=True)
    print("=" * 50, flush=True)
    self_fn = init_query_fn(manager, use_self_rag=True)
    self_reports = evaluate_all(self_fn, verbose=args.verbose, use_judge=args.judge)

    # ── Comparison ──
    print_comparison(base_reports, self_reports)


if __name__ == "__main__":
    main()
