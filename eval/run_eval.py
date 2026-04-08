"""Run advisor approaches A1/A2/A3 against the test set and write JSONL results.

Usage::

    # Smoke test: run only first 10 entries through A1
    python eval/run_eval.py --limit 10 --approaches a1

    # Full run, all 100 entries through all 3 approaches
    python eval/run_eval.py --approaches a1,a2,a3

    # Resume from a previous interrupted run
    python eval/run_eval.py --resume

    # Filter by question type
    python eval/run_eval.py --type factual

Resumability: each (entry_id, approach) result is appended to the output
JSONL on the fly. The script skips any (entry_id, approach) pair that
already exists in the file when ``--resume`` is set.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set

import requests

sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).parent.parent
TEST_SET_PATH = ROOT / "data" / "eval" / "advisor_test_set.jsonl"
RESULTS_PATH = ROOT / "data" / "eval" / "advisor_results.jsonl"

PLANNER_URL = "http://localhost:8001/api/v1/rag/advisor"
CACHE_URL = "http://localhost:8001/api/v1/rag/cache"

APPROACHES = {
    "a1": ("Single-hop", {"use_multihop": False, "use_self_rag": False}),
    "a2": ("Multi-hop", {"use_multihop": True, "use_self_rag": False}),
    "a3": ("Multi+SelfRAG", {"use_multihop": True, "use_self_rag": True}),
}


def load_test_set() -> List[Dict]:
    if not TEST_SET_PATH.exists():
        sys.exit(f"Test set not found: {TEST_SET_PATH}\nRun: python eval/build_test_set.py")
    out: List[Dict] = []
    with TEST_SET_PATH.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def reorder_interleaved(entries: List[Dict]) -> List[Dict]:
    """Reorder entries so the runner cycles through profiles for each question.

    Default test set order is profile-major: P1Q1..P1Q10, P2Q1..P2Q10, ...
    Interleaved order is question-major: P1Q1, P2Q1, ..., P10Q1, P1Q2, P2Q2, ...

    Why: when a long run is interrupted at 17%, profile-major order means
    only the first 1-2 profiles got tested, biasing results. Question-major
    order ensures all 10 profiles are touched after the first ~10 records.
    """
    by_q: Dict[str, List[Dict]] = {}
    q_order: List[str] = []  # preserve original question order
    for e in entries:
        # use question text as key (stable across profiles)
        q = e.get("question", "")
        if q not in by_q:
            by_q[q] = []
            q_order.append(q)
        by_q[q].append(e)

    out: List[Dict] = []
    for q in q_order:
        out.extend(by_q[q])
    return out


def load_existing_keys() -> Set[str]:
    keys: Set[str] = set()
    if not RESULTS_PATH.exists():
        return keys
    with RESULTS_PATH.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                keys.add(f"{rec.get('id')}::{rec.get('approach')}")
            except json.JSONDecodeError:
                continue
    return keys


def call_advisor(question: str, profile: Dict, flags: Dict) -> Dict:
    body = {"question": question, "profile": profile, "top_k": 6, **flags}
    r = requests.post(PLANNER_URL, json=body, timeout=300)
    r.raise_for_status()
    return r.json()


def run(args: argparse.Namespace) -> None:
    entries = load_test_set()
    if args.type:
        entries = [e for e in entries if e.get("question_type") == args.type]
    if args.profile:
        entries = [e for e in entries if args.profile in e.get("profile_label", "")]
    if args.order == "interleave":
        entries = reorder_interleaved(entries)
    if args.limit:
        entries = entries[: args.limit]

    approach_codes = [a.strip().lower() for a in args.approaches.split(",") if a.strip()]
    for code in approach_codes:
        if code not in APPROACHES:
            sys.exit(f"Unknown approach: {code}. Valid: {list(APPROACHES)}")

    existing = load_existing_keys() if args.resume else set()
    if existing:
        print(f"Resuming — skipping {len(existing)} already-completed records")

    if args.clear_cache:
        try:
            requests.delete(CACHE_URL, timeout=10)
            print("RAG cache cleared")
        except Exception as exc:
            print(f"Cache clear failed (continuing): {exc}")

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)

    total = len(entries) * len(approach_codes)
    done = 0
    skipped = 0
    failed = 0
    started = time.monotonic()

    with RESULTS_PATH.open("a", encoding="utf-8") as out_f:
        for entry in entries:
            for code in approach_codes:
                key = f"{entry['id']}::{code}"
                done += 1
                if key in existing:
                    skipped += 1
                    continue
                approach_name, flags = APPROACHES[code]
                t0 = time.monotonic()
                try:
                    res = call_advisor(entry["question"], entry["profile"], flags)
                    elapsed = round(time.monotonic() - t0, 2)
                    record = {
                        "id": entry["id"],
                        "approach": code,
                        "approach_name": approach_name,
                        "question": entry["question"],
                        "question_type": entry["question_type"],
                        "profile_label": entry["profile_label"],
                        "expected_keywords": entry.get("expected_keywords", []),
                        "expected_verdict": entry.get("expected_verdict"),
                        "elapsed_s": elapsed,
                        "verdict": res.get("verdict"),
                        "verdict_summary": res.get("verdict_summary"),
                        "n_checks": len(res.get("requirement_checks") or []),
                        "n_pass": sum(
                            1 for c in (res.get("requirement_checks") or []) if c.get("status") == "pass"
                        ),
                        "n_fail": sum(
                            1 for c in (res.get("requirement_checks") or []) if c.get("status") == "fail"
                        ),
                        "n_unknown": sum(
                            1 for c in (res.get("requirement_checks") or []) if c.get("status") == "unknown"
                        ),
                        "n_actions": len(res.get("recommended_actions") or []),
                        "n_sources": len(res.get("sources") or []),
                        "source_titles": [
                            s.get("title", "") for s in (res.get("sources") or [])
                        ],
                        "reasoning_trace": res.get("reasoning_trace"),
                        "requirement_checks": res.get("requirement_checks") or [],
                        "recommended_actions": res.get("recommended_actions") or [],
                    }
                    out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                    out_f.flush()
                    pct = 100 * done / total
                    eta = (time.monotonic() - started) / done * (total - done)
                    print(
                        f"[{done:3d}/{total}] {pct:5.1f}%  {code}  {entry['id'][:50]:50s}"
                        f"  {elapsed:5.1f}s  ETA={eta/60:.1f}m"
                    )
                except Exception as exc:
                    failed += 1
                    print(f"[{done:3d}/{total}] FAIL {code} {entry['id']}: {exc}")

    elapsed_total = time.monotonic() - started
    print(
        f"\nDone. total={done} skipped={skipped} failed={failed} elapsed={elapsed_total/60:.1f}m"
    )
    print(f"Results: {RESULTS_PATH}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, help="Limit to first N test entries")
    parser.add_argument(
        "--approaches",
        default="a1,a2,a3",
        help="Comma-separated list: a1, a2, a3 (default: all)",
    )
    parser.add_argument("--type", help="Filter by question_type (factual, single_eligibility, ...)")
    parser.add_argument("--profile", help="Filter by profile label substring")
    parser.add_argument("--resume", action="store_true", help="Skip already-completed records")
    parser.add_argument(
        "--clear-cache", action="store_true", help="Clear RAG cache before starting"
    )
    parser.add_argument(
        "--order",
        choices=["entry", "interleave"],
        default="entry",
        help="entry = profile-major (default); interleave = question-major (cycles through profiles).",
    )
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
