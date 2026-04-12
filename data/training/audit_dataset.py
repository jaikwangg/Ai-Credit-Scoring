"""Statistical audit of the loan training dataset.

Reports:
  - Row count, missing values, duplicates
  - Categorical value distributions
  - Numeric statistics (min, max, mean, std, quartiles)
  - Class balance (Approved/Rejected)
  - Per-category default rate
  - Per-grade default rate
  - Cross-tabs (Sex × loan_status, Occupation × loan_status, etc.)
  - Suspicious rows (e.g. perfect-credit rejected, terrible-credit approved)

Run::

    python data/training/audit_dataset.py
"""
import csv
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

CSV = Path(__file__).parent / "loan_dataset_sample.csv"

rows = []
with CSV.open(encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        rows.append(row)

n = len(rows)
print(f"=" * 70)
print(f"DATASET AUDIT — {CSV.name}")
print(f"=" * 70)
print(f"Total rows: {n}")
print(f"Columns: {list(rows[0].keys())}")
print()

# ── 1. Missing values ────────────────────────────────────────────────
print("=== 1. Missing values ===")
for col in rows[0]:
    missing = sum(1 for r in rows if r[col] in (None, "", "NA"))
    if missing > 0:
        print(f"  {col}: {missing}")
    else:
        print(f"  {col}: OK")
print()

# ── 2. Duplicates ────────────────────────────────────────────────────
print("=== 2. Duplicates ===")
seen = set()
dups = 0
for r in rows:
    key = tuple((k, v) for k, v in r.items() if k != "LoanId")
    if key in seen:
        dups += 1
    seen.add(key)
print(f"  Duplicate rows (excluding LoanId): {dups}")
print()

# ── 3. Class balance ─────────────────────────────────────────────────
print("=== 3. Class balance ===")
status_counts = Counter(r["loan_status"] for r in rows)
total = sum(status_counts.values())
for s, c in status_counts.most_common():
    print(f"  {s}: {c} ({100*c/total:.1f}%)")
print()

# ── 4. Categorical distributions ─────────────────────────────────────
print("=== 4. Categorical distributions + per-class default rate ===")
for col in ["Sex", "Marriage_Status", "Occupation", "credit_grade", "Coapplicant"]:
    print(f"\n  [{col}]")
    counts = Counter(r[col] for r in rows)
    by_status = defaultdict(lambda: {"Approved": 0, "Rejected": 0})
    for r in rows:
        by_status[r[col]][r["loan_status"]] += 1
    for val in sorted(counts):
        cnt = counts[val]
        appr = by_status[val]["Approved"]
        rej = by_status[val]["Rejected"]
        appr_rate = 100 * appr / (appr + rej) if (appr + rej) > 0 else 0
        print(f"    {val:32s} n={cnt:4d}  approval_rate={appr_rate:5.1f}%  ({appr}A/{rej}R)")

# ── 5. Numeric statistics ────────────────────────────────────────────
print()
print("=== 5. Numeric statistics ===")
numerics = ["Salary", "credit_score", "outstanding", "overdue", "loan_amount", "loan_term", "Interest_rate"]
for col in numerics:
    vals = [float(r[col]) for r in rows]
    vals.sort()
    p25 = vals[len(vals) // 4]
    p50 = vals[len(vals) // 2]
    p75 = vals[3 * len(vals) // 4]
    print(
        f"  {col:15s}  min={vals[0]:>14,.2f}  p25={p25:>14,.2f}  med={p50:>14,.2f}  "
        f"p75={p75:>14,.2f}  max={vals[-1]:>14,.2f}  std={statistics.pstdev(vals):>12,.2f}"
    )
print()

# ── 6. Discrete buckets for "overdue" ────────────────────────────────
print("=== 6. overdue distribution (should be discrete bucket) ===")
od = Counter(r["overdue"] for r in rows)
for v in sorted(od, key=lambda x: int(x)):
    cnt = od[v]
    pct = 100 * cnt / n
    print(f"  {v:>4s} วัน: {cnt:4d} ({pct:5.1f}%)")
print()

# ── 7. Approval rate by overdue bucket ───────────────────────────────
print("=== 7. Approval rate by overdue bucket ===")
od_status = defaultdict(lambda: {"Approved": 0, "Rejected": 0})
for r in rows:
    od_status[r["overdue"]][r["loan_status"]] += 1
for v in sorted(od_status, key=lambda x: int(x)):
    a = od_status[v]["Approved"]
    rj = od_status[v]["Rejected"]
    rate = 100 * a / (a + rj) if (a + rj) > 0 else 0
    print(f"  {v:>4s} วัน: approval_rate={rate:5.1f}%  ({a}A/{rj}R)")
print()

# ── 8. Approval rate by credit_grade ─────────────────────────────────
print("=== 8. Approval rate by credit_grade ===")
gr_status = defaultdict(lambda: {"Approved": 0, "Rejected": 0})
for r in rows:
    gr_status[r["credit_grade"]][r["loan_status"]] += 1
for g in sorted(gr_status):
    a = gr_status[g]["Approved"]
    rj = gr_status[g]["Rejected"]
    rate = 100 * a / (a + rj) if (a + rj) > 0 else 0
    print(f"  {g}: approval_rate={rate:5.1f}%  ({a}A/{rj}R)")
print()

# ── 9. credit_score range per grade ──────────────────────────────────
print("=== 9. credit_score range per grade (should be monotone) ===")
gr_scores = defaultdict(list)
for r in rows:
    gr_scores[r["credit_grade"]].append(int(r["credit_score"]))
for g in sorted(gr_scores):
    s = gr_scores[g]
    print(f"  {g}: min={min(s):3d}  med={statistics.median(s):.0f}  max={max(s):3d}  n={len(s)}")
print()

# ── 10. Suspicious / contradictory rows ──────────────────────────────
print("=== 10. Suspicious rows ===")
print("  Top-grade AA + high score + Rejected:")
weird_rejects = [
    r for r in rows
    if r["credit_grade"] == "AA"
    and int(r["credit_score"]) >= 800
    and r["loan_status"] == "Rejected"
]
print(f"    Count: {len(weird_rejects)}")
for r in weird_rejects[:5]:
    print(
        f"    LoanId={r['LoanId']:4s} salary={r['Salary']:>7s} loan={r['loan_amount']:>8s} "
        f"score={r['credit_score']} dti={int(r['outstanding'])/max(int(r['Salary']),1):.1f}"
    )

print()
print("  HH grade + low score + Approved:")
weird_approves = [
    r for r in rows
    if r["credit_grade"] == "HH"
    and int(r["credit_score"]) <= 600
    and r["loan_status"] == "Approved"
]
print(f"    Count: {len(weird_approves)}")
for r in weird_approves[:5]:
    print(
        f"    LoanId={r['LoanId']:4s} salary={r['Salary']:>7s} loan={r['loan_amount']:>8s} "
        f"score={r['credit_score']} overdue={r['overdue']}"
    )

print()
print("  Long overdue (≥120 days) + Approved (very surprising):")
weird_overdue = [
    r for r in rows
    if int(r["overdue"]) >= 120 and r["loan_status"] == "Approved"
]
print(f"    Count: {len(weird_overdue)}")
for r in weird_overdue[:5]:
    print(
        f"    LoanId={r['LoanId']:4s} grade={r['credit_grade']} score={r['credit_score']} "
        f"salary={r['Salary']:>7s} loan={r['loan_amount']:>8s}"
    )

# ── 11. Approval rate vs Interest_rate (check if rate has any signal) ─
print()
print("=== 11. Approval rate by Interest_rate bucket ===")
for low, high in [(5.69, 5.74), (5.74, 5.79), (5.79, 5.84), (5.84, 5.90)]:
    bucket = [r for r in rows if low <= float(r["Interest_rate"]) < high]
    if bucket:
        a = sum(1 for r in bucket if r["loan_status"] == "Approved")
        rate = 100 * a / len(bucket)
        print(f"  [{low}, {high}): n={len(bucket):3d}  approval_rate={rate:5.1f}%")
print()

# ── 12. DTI (debt/income) correlation with status ────────────────────
print("=== 12. DTI quartiles vs approval rate ===")
dti_rows = [
    (int(r["outstanding"]) / max(int(r["Salary"]), 1), r["loan_status"])
    for r in rows
]
dti_rows.sort(key=lambda x: x[0])
chunks = [dti_rows[i::4] for i in range(4)]  # not actually quartiles but buckets
for i, label in enumerate(["very low", "low", "high", "very high"]):
    pass
quartile_size = len(dti_rows) // 4
for i, label in enumerate(["Q1 (lowest DTI)", "Q2", "Q3", "Q4 (highest DTI)"]):
    chunk = dti_rows[i * quartile_size : (i + 1) * quartile_size]
    if chunk:
        dti_min = chunk[0][0]
        dti_max = chunk[-1][0]
        a = sum(1 for x in chunk if x[1] == "Approved")
        rate = 100 * a / len(chunk)
        print(f"  {label:18s}: dti=[{dti_min:5.1f}, {dti_max:5.1f}]  approval_rate={rate:5.1f}%")

# ── 13. LTI (loan/income) ────────────────────────────────────────────
print()
print("=== 13. LTI (loan/income) quartiles vs approval rate ===")
lti_rows = sorted(
    [(int(r["loan_amount"]) / max(int(r["Salary"]), 1), r["loan_status"]) for r in rows],
    key=lambda x: x[0],
)
quartile_size = len(lti_rows) // 4
for i, label in enumerate(["Q1 (lowest LTI)", "Q2", "Q3", "Q4 (highest LTI)"]):
    chunk = lti_rows[i * quartile_size : (i + 1) * quartile_size]
    if chunk:
        a = sum(1 for x in chunk if x[1] == "Approved")
        rate = 100 * a / len(chunk)
        print(
            f"  {label:18s}: lti=[{chunk[0][0]:5.1f}, {chunk[-1][0]:5.1f}]"
            f"  approval_rate={rate:5.1f}%"
        )
