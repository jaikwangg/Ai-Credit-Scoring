"""Comparative test for the 3 advisor reasoning approaches.

Runs the same edge-case profile + question through:
  A1) Vanilla profile-conditioned advisor (single-hop retrieval)
  A2) Profile-conditioned + multi-hop query decomposition
  A3) Profile-conditioned + multi-hop + Self-RAG IsSup reflection

Reports per-approach: verdict, num requirement checks, sub-questions used,
sources retrieved, IsSup score (if applicable), elapsed time.

This is the bench script behind the thesis comparison table.
"""
import json
import sys
import time

import requests

sys.stdout.reconfigure(encoding="utf-8")

PROFILE = {
    "salary_per_month": 18000,
    "occupation": "Office Worker",
    "employment_tenure_months": 8,
    "marriage_status": "Single",
    "has_coapplicant": False,
    "outstanding_debt": 50000,
    "overdue_amount": 0,
    "loan_amount_requested": 1500000,
    "loan_term_years": 25,
    "interest_rate": 4.5,
}
QUESTION = "จากโปรไฟล์ของฉัน มีโอกาสกู้บ้านได้ไหม และควรปรับปรุงอะไรบ้าง"

CONFIGS = [
    ("A1: Single-hop", {"use_multihop": False, "use_self_rag": False}),
    ("A2: Multi-hop",  {"use_multihop": True,  "use_self_rag": False}),
    ("A3: Multi+Self", {"use_multihop": True,  "use_self_rag": True}),
]

# Clear cache so each run starts cold
requests.delete("http://localhost:8001/api/v1/rag/cache")

results = []
for name, flags in CONFIGS:
    print(f"\n{'=' * 70}\n{name}\n{'=' * 70}")
    body = {"question": QUESTION, "profile": PROFILE, "top_k": 6, **flags}
    t0 = time.monotonic()
    r = requests.post(
        "http://localhost:8001/api/v1/rag/advisor",
        json=body,
        timeout=300,
    )
    elapsed = time.monotonic() - t0
    if r.status_code != 200:
        print(f"  HTTP {r.status_code}: {r.text[:200]}")
        continue
    d = r.json()
    trace = d.get("reasoning_trace") or {}
    summary = {
        "name": name,
        "elapsed_s": round(elapsed, 1),
        "verdict": d.get("verdict"),
        "n_checks": len(d.get("requirement_checks") or []),
        "n_pass": sum(1 for c in (d.get("requirement_checks") or []) if c.get("status") == "pass"),
        "n_fail": sum(1 for c in (d.get("requirement_checks") or []) if c.get("status") == "fail"),
        "n_unknown": sum(1 for c in (d.get("requirement_checks") or []) if c.get("status") == "unknown"),
        "n_actions": len(d.get("recommended_actions") or []),
        "n_sources": len(d.get("sources") or []),
        "sub_questions": trace.get("sub_questions") or [],
        "sources_per_hop": trace.get("sources_per_hop") or [],
        "issup_score": trace.get("issup_score"),
        "issup_passed": trace.get("issup_passed"),
        "self_rag_retried": trace.get("self_rag_retried"),
    }
    results.append(summary)
    print(f"  elapsed:        {summary['elapsed_s']}s")
    print(f"  verdict:        {summary['verdict']}")
    print(f"  checks:         {summary['n_checks']} (pass={summary['n_pass']} fail={summary['n_fail']} unknown={summary['n_unknown']})")
    print(f"  actions:        {summary['n_actions']}")
    print(f"  sources:        {summary['n_sources']}")
    if summary["sub_questions"]:
        print(f"  sub-questions:  {len(summary['sub_questions'])}")
        for i, sq in enumerate(summary["sub_questions"], 1):
            print(f"    {i}. {sq}")
        print(f"  per-hop counts: {summary['sources_per_hop']}")
    if summary["issup_score"] is not None:
        print(f"  IsSup score:    {summary['issup_score']}/5  (passed={summary['issup_passed']})")
    if summary["self_rag_retried"]:
        print(f"  Self-RAG retry: yes")
    print(f"  summary: {(d.get('verdict_summary') or '')[:200]}")

# Final comparison table
print(f"\n{'=' * 70}\nCOMPARISON TABLE\n{'=' * 70}")
print(f"{'Approach':18} {'Verdict':20} {'Checks':8} {'Sub-Q':6} {'Time':8} {'IsSup':6}")
print("-" * 70)
for r in results:
    sub = len(r.get("sub_questions") or []) or "-"
    issup = r.get("issup_score") or "-"
    print(f"{r['name']:18} {r['verdict']:20} {r['n_checks']:>3}/p{r['n_pass']}/f{r['n_fail']}  {str(sub):>6} {r['elapsed_s']:>5}s  {str(issup):>6}")
