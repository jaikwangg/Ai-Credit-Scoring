"""RAG evaluation pipeline — offline quality measurement."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from statistics import mean
from typing import Any, Callable, Dict, List, Optional

from src.rag.validator import NO_ANSWER_MESSAGE


@dataclass
class EvalCase:
    question: str
    expected_route: str
    expected_keywords: List[str]  # at least one must appear in answer (case-insensitive)
    should_answer: bool = True  # False = NO_ANSWER_MESSAGE is the correct response
    expected_doc_hint: Optional[str] = None  # substring of expected source doc title

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "EvalCase":
        return cls(
            question=d["question"],
            expected_route=d["expected_route"],
            expected_keywords=d.get("expected_keywords", []),
            should_answer=d.get("should_answer", True),
            expected_doc_hint=d.get("expected_doc_hint"),
        )


@dataclass
class EvalResult:
    case: EvalCase
    answer: str
    router_label: str
    retrieved_count: int
    validated_count: int
    route_correct: bool
    has_answer: bool
    keyword_hit: bool
    keywords_found: List[str] = field(default_factory=list)
    source_titles: List[str] = field(default_factory=list)
    doc_hint_found: bool = False
    error: Optional[str] = None  # set when query raised an exception

    @property
    def timed_out(self) -> bool:
        return self.error is not None

    @property
    def correct(self) -> bool:
        if self.timed_out:
            return False
        if not self.case.should_answer:
            return not self.has_answer
        return self.route_correct and self.has_answer and self.keyword_hit


def load_test_cases(path: Path) -> List[EvalCase]:
    cases: List[EvalCase] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                cases.append(EvalCase.from_dict(json.loads(line)))
            except (json.JSONDecodeError, KeyError):
                continue
    return cases


def _keyword_check(answer: str, keywords: List[str]) -> List[str]:
    lower = answer.lower()
    return [kw for kw in keywords if kw.lower() in lower]


def evaluate_single(
    case: EvalCase,
    query_fn: Callable[[str], Dict[str, Any]],
) -> EvalResult:
    _failed = EvalResult(
        case=case,
        answer="",
        router_label="general_info",
        retrieved_count=0,
        validated_count=0,
        route_correct=False,
        has_answer=False,
        keyword_hit=False,
    )

    try:
        result = query_fn(case.question)
    except Exception as exc:
        err = f"{type(exc).__name__}: {exc}"
        print(f"  [ERROR] {err}")
        _failed.error = err
        return _failed

    answer = result.get("answer", "")
    router_label = result.get("router_label", "general_info")
    retrieved_count = result.get("retrieved_node_count", 0)
    validated_count = result.get("validated_node_count", 0)

    has_answer = answer.strip() != NO_ANSWER_MESSAGE and bool(answer.strip())
    found_keywords = _keyword_check(answer, case.expected_keywords) if case.expected_keywords else []
    keyword_hit = bool(found_keywords) if case.expected_keywords else has_answer

    source_titles = [
        s.get("metadata", {}).get("title", "")
        for s in result.get("sources", [])
    ]
    doc_hint_found = False
    if case.expected_doc_hint:
        doc_hint_found = any(
            case.expected_doc_hint.lower() in t.lower() for t in source_titles
        )

    return EvalResult(
        case=case,
        answer=answer,
        router_label=router_label,
        retrieved_count=retrieved_count,
        validated_count=validated_count,
        route_correct=router_label == case.expected_route,
        has_answer=has_answer,
        keyword_hit=keyword_hit,
        keywords_found=found_keywords,
        source_titles=source_titles,
        doc_hint_found=doc_hint_found,
    )


def run_eval(
    cases: List[EvalCase],
    query_fn: Callable[[str], Dict[str, Any]],
) -> List[EvalResult]:
    results = []
    for i, case in enumerate(cases, 1):
        print(f"[{i}/{len(cases)}] {case.question[:60]}...")
        results.append(evaluate_single(case, query_fn))
    return results


def compute_metrics(results: List[EvalResult]) -> Dict[str, Any]:
    if not results:
        return {}

    total = len(results)
    timed_out = sum(1 for r in results if r.timed_out)
    # Metrics are computed only on cases that did not error/timeout
    valid = [r for r in results if not r.timed_out]
    if not valid:
        return {"total": total, "timed_out": timed_out, "valid": 0}

    correct = sum(1 for r in valid if r.correct)
    route_correct = sum(1 for r in valid if r.route_correct)
    has_answer = sum(1 for r in valid if r.has_answer)
    keyword_hit = sum(1 for r in valid if r.keyword_hit)

    no_answer_cases = [r for r in valid if not r.case.should_answer]
    no_answer_correct = sum(1 for r in no_answer_cases if not r.has_answer)

    mean_retrieved = mean(r.retrieved_count for r in valid)
    mean_validated = mean(r.validated_count for r in valid)

    by_route: Dict[str, Dict[str, int]] = {}
    for r in valid:
        label = r.case.expected_route
        if label not in by_route:
            by_route[label] = {"total": 0, "correct": 0}
        by_route[label]["total"] += 1
        if r.correct:
            by_route[label]["correct"] += 1

    n = len(valid)
    return {
        "total": total,
        "timed_out": timed_out,
        "valid": n,
        "overall_accuracy": correct / n,
        "route_accuracy": route_correct / n,
        "answer_rate": has_answer / n,
        "keyword_hit_rate": keyword_hit / n,
        "no_answer_accuracy": (
            no_answer_correct / len(no_answer_cases) if no_answer_cases else None
        ),
        "mean_retrieved_count": mean_retrieved,
        "mean_validated_count": mean_validated,
        "by_route": {
            label: {"accuracy": v["correct"] / v["total"], "total": v["total"]}
            for label, v in by_route.items()
        },
    }


def print_report(metrics: Dict[str, Any], results: List[EvalResult]) -> None:
    print("\n" + "=" * 60)
    print("RAG Evaluation Report")
    print("=" * 60)
    print(f"Total cases       : {metrics['total']}")
    if metrics.get("timed_out", 0):
        print(f"Timed out         : {metrics['timed_out']}  ← increase Ollama request_timeout")
    print(f"Valid (evaluated) : {metrics.get('valid', metrics['total'])}")
    if metrics.get("valid", 1) == 0:
        print("No valid results to report.")
        print("=" * 60)
        return
    print(f"Overall accuracy  : {metrics['overall_accuracy']:.1%}")
    print(f"Route accuracy    : {metrics['route_accuracy']:.1%}")
    print(f"Answer rate       : {metrics['answer_rate']:.1%}")
    print(f"Keyword hit rate  : {metrics['keyword_hit_rate']:.1%}")
    if metrics.get("no_answer_accuracy") is not None:
        print(f"No-answer accuracy: {metrics['no_answer_accuracy']:.1%}")
    print(f"Mean retrieved    : {metrics['mean_retrieved_count']:.1f}")
    print(f"Mean validated    : {metrics['mean_validated_count']:.1f}")

    print("\nBy route:")
    for label, stats in metrics.get("by_route", {}).items():
        bar = "#" * int(stats["accuracy"] * 20)
        print(f"  {label:<22} {stats['accuracy']:.1%}  [{bar:<20}]  ({stats['total']} cases)")

    failed = [r for r in results if not r.correct]
    print(f"\nFailed cases ({len(failed)}):")
    if not failed:
        print("  (none)")
    for r in failed:
        parts = []
        if not r.route_correct:
            parts.append(f"route={r.router_label} (expected {r.case.expected_route})")
        if r.case.should_answer and not r.has_answer:
            parts.append("no_answer")
        if r.case.should_answer and not r.keyword_hit:
            parts.append(f"missing_keywords={r.case.expected_keywords}")
        if not r.case.should_answer and r.has_answer:
            parts.append("should_be_no_answer")
        print(f"  Q: {r.case.question[:58]}")
        print(f"     → {', '.join(parts)}")

    print("=" * 60)
