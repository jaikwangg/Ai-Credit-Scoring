"""
Compare Gemini vs Ollama across RAG + Planning evaluation.
==========================================================
Runs all test cases (RAG + Planning) for each provider and saves:
  results/eval_gemini.json  — full Gemini results
  results/eval_ollama.json  — full Ollama results
  results/compare_<timestamp>.json — side-by-side comparison table

Usage:
    uv run python scripts/compare_providers.py 2>/dev/null
    uv run python scripts/compare_providers.py --verbose 2>/dev/null
    uv run python scripts/compare_providers.py --skip-ollama 2>/dev/null  # Gemini only
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
except AttributeError:
    pass

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from llama_index.core.settings import Settings

from config.settings import settings as cfg
from scripts.evaluate_rag import (
    NO_ANSWER,
    PLAN_TEST_CASES,
    TEST_CASES,
    PlanCaseReport,
    RAGCaseReport,
    evaluate_all,
    evaluate_planning,
    init_rag_manager,
    init_query_fn,
    save_results_json,
)


# ── LLM factories ───────────────────────────────────────────────────────────────
def build_gemini_llm():
    from llama_index.llms.google_genai import GoogleGenAI
    return GoogleGenAI(
        model=cfg.GEMINI_MODEL,
        api_key=cfg.GEMINI_API_KEY,
        temperature=0.1,
    )


def build_ollama_llm():
    from llama_index.llms.ollama import Ollama
    return Ollama(
        model=cfg.OLLAMA_MODEL,
        base_url=cfg.OLLAMA_BASE_URL,
        temperature=0.1,
        request_timeout=120.0,
    )


# ── per-provider run ────────────────────────────────────────────────────────────
def run_provider(
    name: str,
    label: str,
    llm_factory,
    manager,
    verbose: bool = False,
) -> tuple[list[RAGCaseReport], list[PlanCaseReport], str]:
    print(f"\n{'='*65}")
    print(f"  Provider : {label}")
    print(f"{'='*65}\n")

    llm = llm_factory()
    manager.llm = llm
    Settings.llm = llm

    print(f"[RAG Tests — {len(TEST_CASES)} cases]")
    rag_reports = evaluate_all(
        manager.query, verbose=verbose, use_judge=False
    )

    print(f"\n[Planning Tests — {len(PLAN_TEST_CASES)} cases]")
    plan_reports = evaluate_planning(manager, verbose=verbose)

    out_path = f"results/eval_{name}.json"
    save_results_json(rag_reports, out_path, plan_reports=plan_reports)

    return rag_reports, plan_reports, out_path


# ── comparison builder ──────────────────────────────────────────────────────────
def _rag_summary(reports: list[RAGCaseReport]) -> dict:
    total = sum(r.total for r in reports)
    passed = sum(r.passed for r in reports)
    router_ok = sum(1 for r in reports if r.router_label == r.case.expected_route)
    answered = sum(
        1 for r in reports
        if r.case.expect_answer and r.answer.strip() and r.answer.strip() != NO_ANSWER
    )
    expected_ans = sum(1 for r in reports if r.case.expect_answer)
    prec_vals = [r.precision_at_k for r in reports if r.precision_at_k is not None]
    mean_lat = sum(r.elapsed_s for r in reports) / len(reports) if reports else 0.0
    return {
        "overall_pct":        round(passed / total, 4) if total else 0,
        "router_accuracy":    round(router_ok / len(reports), 4) if reports else 0,
        "answer_rate":        round(answered / expected_ans, 4) if expected_ans else 0,
        "mean_precision_at_k": round(sum(prec_vals) / len(prec_vals), 4) if prec_vals else None,
        "mean_latency_s":     round(mean_lat, 3),
        "passed": passed,
        "total": total,
    }


def _plan_summary(reports: list[PlanCaseReport]) -> dict:
    total = sum(r.total for r in reports)
    passed = sum(r.passed for r in reports)
    mode_ok = sum(1 for r in reports if r.mode == r.case.expected_mode)
    doc_cases = [r for r in reports if r.case.expect_documented_evidence]
    doc_pass = sum(1 for r in doc_cases if r.documented_count >= 1)
    issup_cases = [r for r in reports if r.case.use_issup]
    issup_scores = [r.issup_score for r in issup_cases if r.issup_score is not None]
    issup_pass = sum(1 for r in issup_cases if r.issup_passed)
    mean_lat = sum(r.elapsed_s for r in reports) / len(reports) if reports else 0.0
    return {
        "overall_pct":              round(passed / total, 4) if total else 0,
        "mode_accuracy":            round(mode_ok / len(reports), 4) if reports else 0,
        "documented_evidence_rate": round(doc_pass / len(doc_cases), 4) if doc_cases else None,
        "issup_pass_rate":          round(issup_pass / len(issup_cases), 4) if issup_cases else None,
        "mean_issup_score":         round(sum(issup_scores) / len(issup_scores), 2) if issup_scores else None,
        "mean_latency_s":           round(mean_lat, 3),
        "passed": passed,
        "total": total,
    }


def build_comparison(all_results: dict) -> dict:
    providers = list(all_results.keys())

    rag = {p: _rag_summary(all_results[p][0]) for p in providers}
    plan = {p: _plan_summary(all_results[p][1]) for p in providers}

    def _fmt(val, fmt=".0%"):
        if val is None:
            return "N/A"
        if fmt == ".0%":
            return f"{val:.0%}"
        if fmt == ".2f":
            return f"{val:.2f}"
        if fmt == ".1f":
            return f"{val:.1f}s"
        return str(val)

    metrics = [
        ("RAG Overall",           lambda p: _fmt(rag[p]["overall_pct"])),
        ("RAG Router Accuracy",   lambda p: _fmt(rag[p]["router_accuracy"])),
        ("RAG Answer Rate",       lambda p: _fmt(rag[p]["answer_rate"])),
        ("RAG Mean Precision@K",  lambda p: _fmt(rag[p]["mean_precision_at_k"], ".2f")),
        ("RAG Mean Latency",      lambda p: _fmt(rag[p]["mean_latency_s"], ".1f")),
        ("Plan Overall",          lambda p: _fmt(plan[p]["overall_pct"])),
        ("Plan Mode Accuracy",    lambda p: _fmt(plan[p]["mode_accuracy"])),
        ("Plan Documented Evid.", lambda p: _fmt(plan[p]["documented_evidence_rate"])),
        ("Plan IsSup Pass Rate",  lambda p: _fmt(plan[p]["issup_pass_rate"])),
        ("Plan Mean IsSup Score", lambda p: _fmt(plan[p]["mean_issup_score"], ".2f")),
        ("Plan Mean Latency",     lambda p: _fmt(plan[p]["mean_latency_s"], ".1f")),
    ]

    table = []
    for metric_name, fn in metrics:
        row = {"metric": metric_name}
        for p in providers:
            row[p] = fn(p)
        table.append(row)

    return {
        "meta": {
            "timestamp": datetime.now().isoformat(),
            "providers": providers,
            "rag_cases": len(TEST_CASES),
            "plan_cases": len(PLAN_TEST_CASES),
        },
        "rag": rag,
        "planning": plan,
        "summary_table": table,
    }


def print_comparison_table(comparison: dict) -> None:
    providers = comparison["meta"]["providers"]
    table = comparison["summary_table"]

    col_w = 28
    p_w = 22

    header = f"{'Metric':<{col_w}}" + "".join(f"{p:<{p_w}}" for p in providers)
    sep = "-" * (col_w + p_w * len(providers))

    print(f"\n{'='*65}")
    print("  Gemini vs Ollama — Side-by-Side Comparison")
    print(f"{'='*65}")
    print(header)
    print(sep)
    prev_section = ""
    for row in table:
        section = row["metric"].split(" ")[0]
        if section != prev_section and prev_section:
            print()
        prev_section = section
        line = f"{row['metric']:<{col_w}}" + "".join(f"{row.get(p, 'N/A'):<{p_w}}" for p in providers)
        print(line)
    print()


# ── main ────────────────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--skip-ollama", action="store_true",
                        help="Run Gemini only (skip Ollama)")
    args = parser.parse_args()

    print("Initializing RAG (embedding model + ChromaDB)...")
    try:
        manager = init_rag_manager()
    except Exception as exc:
        print(f"[ERROR] Cannot initialize RAG: {exc}")
        print("  Make sure ChromaDB is populated: uv run python -m src.ingest")
        sys.exit(1)

    providers_to_run = [
        ("gemini", f"Gemini  [{cfg.GEMINI_MODEL}]", build_gemini_llm),
    ]
    if not args.skip_ollama:
        providers_to_run.append(
            ("ollama", f"Ollama  [{cfg.OLLAMA_MODEL}]", build_ollama_llm)
        )

    all_results: dict = {}
    output_files: dict = {}

    for name, label, factory in providers_to_run:
        rag_reports, plan_reports, out_path = run_provider(
            name, label, factory, manager, verbose=args.verbose
        )
        all_results[name] = (rag_reports, plan_reports)
        output_files[name] = out_path

    if len(all_results) >= 2:
        comparison = build_comparison(all_results)
        print_comparison_table(comparison)

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        cmp_path = Path(f"results/compare_{ts}.json")
        cmp_path.parent.mkdir(parents=True, exist_ok=True)
        cmp_path.write_text(
            json.dumps(comparison, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"Comparison saved → {cmp_path.resolve()}")
    else:
        print(f"\nSingle-provider run complete. Results: {list(output_files.values())}")


if __name__ == "__main__":
    main()
