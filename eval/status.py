"""Quick progress check for an in-flight eval run."""
import json
import sys
from collections import Counter
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).parent.parent
RESULTS = ROOT / "data" / "eval" / "advisor_results.jsonl"
TEST_SET = ROOT / "data" / "eval" / "advisor_test_set.jsonl"

if not RESULTS.exists():
    print("No results yet")
    sys.exit(0)

records = []
with RESULTS.open(encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if line:
            records.append(json.loads(line))

n_test = sum(1 for _ in TEST_SET.open(encoding="utf-8"))
total_planned = n_test * 3  # 3 approaches

print(f"Progress: {len(records)} / {total_planned}  ({100*len(records)/total_planned:.1f}%)")

by_approach = Counter(r.get("approach") for r in records)
for ap in sorted(by_approach):
    print(f"  {ap}: {by_approach[ap]} / {n_test}")

if records:
    elapsed_total = sum(r.get("elapsed_s") or 0 for r in records)
    avg = elapsed_total / len(records)
    remaining = total_planned - len(records)
    eta_min = (remaining * avg) / 60
    print(f"\navg per call: {avg:.1f}s")
    print(f"remaining:    {remaining}")
    print(f"ETA:          {eta_min:.0f} min ({eta_min/60:.1f} hours)")

    last = records[-1]
    print(f"\nLast result: {last.get('id')} ({last.get('approach')})")
    print(f"  verdict={last.get('verdict')} checks={last.get('n_checks')} elapsed={last.get('elapsed_s')}s")
