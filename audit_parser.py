"""Audit script: inspect chunk quality in the Chroma collection.

Run: python audit_parser.py
"""
import re
import sys
import chromadb
from dotenv import load_dotenv
load_dotenv(".env")
from config.settings import settings

sys.stdout.reconfigure(encoding="utf-8")

client = chromadb.PersistentClient(path=settings.CHROMA_PERSIST_DIR)
coll = client.get_collection(settings.CHROMA_COLLECTION)

res = coll.get(limit=coll.count())
docs = res["documents"]
metas = res["metadatas"]
print(f"Loaded {len(docs)} chunks from collection={settings.CHROMA_COLLECTION}\n")

# === 1. Length distribution ===
lengths = [len(d) for d in docs]
print("=== AUDIT 1: chunk length distribution ===")
print(f"  total chunks: {len(docs)}")
print(f"  min={min(lengths)} max={max(lengths)} avg={sum(lengths)//len(lengths)}")
too_short = [i for i, l in enumerate(lengths) if l < 100]
too_long = [i for i, l in enumerate(lengths) if l > 2500]
print(f"  < 100 chars: {len(too_short)}")
print(f"  > 2500 chars: {len(too_long)}")
if too_short[:3]:
    print("  SHORT examples:")
    for i in too_short[:3]:
        print(f"    [{lengths[i]}c] {docs[i][:140]!r}")
if too_long[:2]:
    print("  LONG examples:")
    for i in too_long[:2]:
        print(f"    [{lengths[i]}c] title={metas[i].get('title','?')[:50]}")

# === 2. Broken sentence boundaries ===
print("\n=== AUDIT 2: broken boundaries ===")
broken_end = 0
broken_start = 0
for d in docs:
    s = d.strip()
    if not s:
        continue
    last = s[-1]
    if not re.search(r"[.!?)\]}]$|[\u0E01-\u0E5B]$|\d$", s):
        broken_end += 1
    # Starts mid-sentence (with lowercase latin or orphan connector)
    if re.match(r"^[a-z]|^[,.)\]]|^และ\s|^หรือ\s|^ที่\s", s):
        broken_start += 1
print(f"  chunks not ending with terminator: {broken_end}/{len(docs)} ({100*broken_end/len(docs):.0f}%)")
print(f"  chunks starting mid-sentence:      {broken_start}/{len(docs)} ({100*broken_start/len(docs):.0f}%)")

# === 3. Incomplete table fragments ===
print("\n=== AUDIT 3: table fragments ===")
table_cues = ["อาชีพ สถานที่", "กู้เดี่ยว", "ผู้กู้หลัก", "กรุงเทพฯ และปริมณฑล"]
table_orphans = 0
for d in docs:
    hits = sum(1 for c in table_cues if c in d)
    has_number = bool(re.search(r"\d{1,3}(,\d{3})+", d))
    if hits >= 2 and not has_number:
        table_orphans += 1
print(f"  table headers without numeric data: {table_orphans}")

# === 4. Metadata sanity ===
print("\n=== AUDIT 4: category vs content sanity ===")
expected_map = {
    "interest_structure": ["ดอกเบี้ย", "interest", "rate", "mrr", "fixed", "floating"],
    "fee_structure": ["ค่าธรรมเนียม", "fee", "prepayment", "ค่าปรับ"],
    "hardship_support": ["ช่วยเหลือ", "ปรับโครงสร้าง", "น้ำท่วม", "covid", "พักชำระ", "ขยายระยะเวลา"],
    "refinance": ["รีไฟแนนซ์", "refinance"],
    "policy_requirement": ["คุณสมบัติ", "เอกสาร", "faq", "คำถามที่พบบ่อย", "รายได้", "อายุผู้กู้", "สัญชาติ"],
}
mismatch = 0
mismatch_examples = []
for d, m in zip(docs, metas):
    cat = m.get("category", "")
    title = m.get("title", "")
    kws = expected_map.get(cat, [])
    if kws:
        text = f"{title}\n{d}".lower()
        if not any(k in text for k in kws):
            mismatch += 1
            if len(mismatch_examples) < 3:
                mismatch_examples.append((cat, title[:50], d[:100]))
print(f"  category/content mismatches: {mismatch}/{len(docs)}")
for cat, title, snippet in mismatch_examples:
    print(f"    cat={cat:20s} title={title!r}")
    print(f"       snippet={snippet!r}")

# === 5. Critical fact coverage ===
print("\n=== AUDIT 5: critical fact coverage ===")
facts = {
    "รายได้ 15,000 (พนักงาน)": "15,000",
    "รายได้ 30,000 (ธุรกิจ)": "30,000",
    "DSR 40-50%": "40-50%",
    "อายุ 21-62": "21-62",
    "LTV 90%": "90%",
    "น้ำท่วม มาตรการ": "น้ำท่วม",
    "Mortgage Power 30 ล้าน": "30 ล้าน",
    "ค่าปรับ prepayment": "ค่าปรับ",
    "คนไทยต่างประเทศ (expat)": "expat",
    "3 เดือน พักชำระ": "3 เดือน",
}
for label, needle in facts.items():
    n = sum(1 for d in docs if needle in d)
    status = "OK " if n > 0 else "MISS"
    print(f"  [{status}] {label:30s} → {n:3d} chunks contain {needle!r}")

# === 6. Noise / URL / leaked metadata ===
print("\n=== AUDIT 6: noise and metadata leakage ===")
url_noise = sum(1 for d in docs if ("/help-support/" in d or ".html" in d))
# Real parser header leak: parser-injected metadata fields. The "=== ... ==="
# markers found in the early audit are legitimate FAQ section dividers from
# the source documents themselves, NOT parser leakage — exclude them.
parser_header_leak = sum(
    1 for d in docs
    if "CLEANING_VERSION:" in d
    or "SOURCE URL:" in d
    or "PUBLICATION DATE:" in d
    or d.startswith("TITLE:")
)
empty_summary = sum(1 for d in docs if len(d.strip()) < 50)
print(f"  chunks containing URLs:          {url_noise}")
print(f"  chunks with parser header leaked: {parser_header_leak}")
print(f"  chunks with < 50 chars of body:   {empty_summary}")

# === 7. Duplicate detection ===
print("\n=== AUDIT 7: duplicate chunks ===")
seen = {}
dupes = 0
for i, d in enumerate(docs):
    sig = d.strip()[:200]
    if sig in seen:
        dupes += 1
    else:
        seen[sig] = i
print(f"  near-duplicate chunks (first 200 chars match): {dupes}/{len(docs)}")

# === 8. Per-category chunk size distribution ===
print("\n=== AUDIT 8: size distribution per category ===")
from collections import defaultdict
per_cat = defaultdict(list)
for d, m in zip(docs, metas):
    per_cat[m.get("category", "?")].append(len(d))
for cat in sorted(per_cat):
    ls = per_cat[cat]
    print(f"  {cat:22s}: n={len(ls):3d}  avg={sum(ls)//len(ls):5d}  min={min(ls):4d}  max={max(ls):5d}")
