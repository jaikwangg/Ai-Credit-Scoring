"""Generate the advisor evaluation test set as JSONL.

Produces ``data/eval/advisor_test_set.jsonl`` with 100 entries that
systematically cover the policy space:

  - 10 profile templates (strong → thin-file → edge cases)
  - 10 question templates per profile (factual + eligibility + advice)
  = 100 unique (profile, question) test cases

Each entry carries:
  - id, question, question_type, profile
  - expected_keywords: phrases the answer should mention (recall metric)
  - expected_verdict: weak ground-truth label (eligible/ineligible/None)
  - ground_truth_requirements: list of policy facts the case should test

Re-run this script if you change the templates. The eval runner consumes
the JSONL file and ignores templates entirely.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

OUT_PATH = Path(__file__).parent.parent / "data" / "eval" / "advisor_test_set.jsonl"

# ────────────────────────────────────────────────────────────────────────────
# Profile templates — covers 10 archetypes the advisor should handle
# ────────────────────────────────────────────────────────────────────────────
PROFILES: List[Dict[str, Any]] = [
    {
        "label": "P1_strong_doctor",
        "description": "High-income professional with clean credit",
        "expected_class": "eligible",
        "data": {
            "salary_per_month": 200000,
            "occupation": "Doctor",
            "employment_tenure_months": 60,
            "marriage_status": "Married",
            "has_coapplicant": False,
            "credit_score": 800,
            "credit_grade": "AA",
            "outstanding_debt": 0,
            "overdue_amount": 0,
            "loan_amount_requested": 3000000,
            "loan_term_years": 20,
            "interest_rate": 3.5,
        },
    },
    {
        "label": "P2_solid_engineer",
        "description": "Salaried engineer with mid-tier credit",
        "expected_class": "eligible",
        "data": {
            "salary_per_month": 50000,
            "occupation": "Engineer",
            "employment_tenure_months": 36,
            "marriage_status": "Single",
            "has_coapplicant": False,
            "credit_score": 720,
            "credit_grade": "BB",
            "outstanding_debt": 100000,
            "overdue_amount": 0,
            "loan_amount_requested": 1500000,
            "loan_term_years": 25,
            "interest_rate": 4.0,
        },
    },
    {
        "label": "P3_average_office",
        "description": "Average office worker, BKK",
        "expected_class": "partially_eligible",
        "data": {
            "salary_per_month": 30000,
            "occupation": "Office Worker",
            "employment_tenure_months": 24,
            "marriage_status": "Single",
            "has_coapplicant": False,
            "credit_score": 650,
            "credit_grade": "CC",
            "outstanding_debt": 80000,
            "overdue_amount": 0,
            "loan_amount_requested": 1500000,
            "loan_term_years": 25,
            "interest_rate": 4.5,
        },
    },
    {
        "label": "P4_borderline_short_tenure",
        "description": "Income meets minimum but tenure too short",
        "expected_class": "ineligible",
        "data": {
            "salary_per_month": 18000,
            "occupation": "Office Worker",
            "employment_tenure_months": 8,
            "marriage_status": "Single",
            "has_coapplicant": False,
            "credit_score": 600,
            "credit_grade": "CC",
            "outstanding_debt": 50000,
            "overdue_amount": 0,
            "loan_amount_requested": 1500000,
            "loan_term_years": 25,
            "interest_rate": 5.0,
        },
    },
    {
        "label": "P5_thin_file_fresh_grad",
        "description": "Fresh graduate with no credit history",
        "expected_class": "ineligible",
        "data": {
            "salary_per_month": 20000,
            "occupation": "Office Worker",
            "employment_tenure_months": 4,
            "marriage_status": "Single",
            "has_coapplicant": False,
            "outstanding_debt": 0,
            "overdue_amount": 0,
            "loan_amount_requested": 2000000,
            "loan_term_years": 30,
            "interest_rate": 5.0,
        },
    },
    {
        "label": "P6_self_employed_strong",
        "description": "Established business owner",
        "expected_class": "eligible",
        "data": {
            "salary_per_month": 80000,
            "occupation": "Business Owner",
            "employment_tenure_months": 48,
            "marriage_status": "Married",
            "has_coapplicant": True,
            "coapplicant_income": 30000,
            "credit_score": 700,
            "credit_grade": "BB",
            "outstanding_debt": 200000,
            "overdue_amount": 0,
            "loan_amount_requested": 2500000,
            "loan_term_years": 20,
            "interest_rate": 4.5,
        },
    },
    {
        "label": "P7_self_employed_weak",
        "description": "New business owner, low income, outside BKK",
        "expected_class": "ineligible",
        "data": {
            "salary_per_month": 25000,
            "occupation": "Business Owner",
            "employment_tenure_months": 18,
            "marriage_status": "Married",
            "has_coapplicant": False,
            "credit_score": 580,
            "credit_grade": "DD",
            "outstanding_debt": 30000,
            "overdue_amount": 0,
            "loan_amount_requested": 1200000,
            "loan_term_years": 30,
            "interest_rate": 5.5,
        },
    },
    {
        "label": "P8_has_overdue",
        "description": "Has past-due payments — major red flag",
        "expected_class": "ineligible",
        "data": {
            "salary_per_month": 35000,
            "occupation": "Teacher",
            "employment_tenure_months": 60,
            "marriage_status": "Married",
            "has_coapplicant": False,
            "credit_score": 600,
            "credit_grade": "CC",
            "outstanding_debt": 100000,
            "overdue_amount": 8000,
            "loan_amount_requested": 1000000,
            "loan_term_years": 20,
            "interest_rate": 5.0,
        },
    },
    {
        "label": "P9_high_lti",
        "description": "Decent income but loan ratio too high",
        "expected_class": "ineligible",
        "data": {
            "salary_per_month": 50000,
            "occupation": "Engineer",
            "employment_tenure_months": 36,
            "marriage_status": "Single",
            "has_coapplicant": False,
            "credit_score": 720,
            "credit_grade": "BB",
            "outstanding_debt": 0,
            "overdue_amount": 0,
            "loan_amount_requested": 5000000,
            "loan_term_years": 30,
            "interest_rate": 4.0,
        },
    },
    {
        "label": "P10_coapplicant_combined",
        "description": "Low individual income but co-applicant lifts combined",
        "expected_class": "partially_eligible",
        "data": {
            "salary_per_month": 15000,
            "occupation": "Office Worker",
            "employment_tenure_months": 18,
            "marriage_status": "Married",
            "has_coapplicant": True,
            "coapplicant_income": 18000,
            "credit_score": 650,
            "credit_grade": "CC",
            "outstanding_debt": 20000,
            "overdue_amount": 0,
            "loan_amount_requested": 1200000,
            "loan_term_years": 25,
            "interest_rate": 4.5,
        },
    },
]


# ────────────────────────────────────────────────────────────────────────────
# Question templates — 10 per profile, mixing question types
# ────────────────────────────────────────────────────────────────────────────
QUESTION_TEMPLATES: List[Dict[str, Any]] = [
    {
        "key": "Q01_overall_eligibility",
        "type": "multi_eligibility",
        "text": "จากโปรไฟล์ของฉัน มีโอกาสได้รับการอนุมัติสินเชื่อบ้านหรือไม่ และเพราะเหตุใด",
        "expected_keywords": ["รายได้", "อายุงาน", "เครดิต"],
        "ground_truth_requirements": ["รายได้ขั้นต่ำ", "อายุงาน", "DSR"],
    },
    {
        "key": "Q02_min_income_check",
        "type": "single_eligibility",
        "text": "รายได้ของฉันผ่านเกณฑ์รายได้ขั้นต่ำของธนาคารหรือไม่",
        "expected_keywords": ["15,000", "30,000", "รายได้ขั้นต่ำ"],
        "ground_truth_requirements": ["รายได้ขั้นต่ำ"],
    },
    {
        "key": "Q03_dsr_check",
        "type": "single_eligibility",
        "text": "ภาระหนี้ต่อรายได้ (DSR) ของฉันอยู่ในเกณฑ์ที่ธนาคารยอมรับหรือไม่",
        "expected_keywords": ["40", "50%", "DSR"],
        "ground_truth_requirements": ["DSR"],
    },
    {
        "key": "Q04_documents_needed",
        "type": "factual",
        "text": "ฉันต้องเตรียมเอกสารอะไรบ้างในการสมัครสินเชื่อบ้าน",
        "expected_keywords": ["บัตรประชาชน", "สลิป", "ทะเบียนบ้าน"],
        "ground_truth_requirements": [],
    },
    {
        "key": "Q05_improvement_advice",
        "type": "advice",
        "text": "ฉันควรปรับปรุงอะไรในโปรไฟล์ของฉันเพื่อเพิ่มโอกาสอนุมัติ",
        "expected_keywords": [],
        "ground_truth_requirements": [],
    },
    {
        "key": "Q06_payment_relief",
        "type": "factual",
        "text": "ถ้าฉันผ่อนไม่ไหว ธนาคารมีมาตรการช่วยเหลืออะไรบ้าง",
        "expected_keywords": ["พักชำระ", "ปรับโครงสร้าง", "3 เดือน"],
        "ground_truth_requirements": [],
    },
    {
        "key": "Q07_max_loan_amount",
        "type": "single_eligibility",
        "text": "ตามรายได้ของฉัน ฉันน่าจะกู้ได้สูงสุดประมาณเท่าไหร่",
        "expected_keywords": ["DSR", "40", "50%"],
        "ground_truth_requirements": ["DSR", "วงเงิน"],
    },
    {
        "key": "Q08_max_loan_term",
        "type": "factual",
        "text": "ระยะเวลาสูงสุดที่กู้บ้านได้คือกี่ปี และมีเงื่อนไขเรื่องอายุผู้กู้อย่างไร",
        "expected_keywords": ["70", "อายุ"],
        "ground_truth_requirements": ["อายุผู้กู้"],
    },
    {
        "key": "Q09_coapplicant_advice",
        "type": "advice",
        "text": "ถ้าฉันหาผู้กู้ร่วมมาช่วย จะเพิ่มโอกาสอนุมัติได้ไหม และผู้กู้ร่วมต้องมีคุณสมบัติอย่างไร",
        "expected_keywords": ["ผู้กู้ร่วม", "รายได้"],
        "ground_truth_requirements": ["ผู้กู้ร่วม"],
    },
    {
        "key": "Q10_tenure_check",
        "type": "single_eligibility",
        "text": "อายุงานปัจจุบันของฉันถึงเกณฑ์ที่ธนาคารกำหนดหรือยัง",
        "expected_keywords": ["6 เดือน", "1 ปี", "อายุงาน"],
        "ground_truth_requirements": ["อายุงาน"],
    },
]


def build_entries() -> List[Dict[str, Any]]:
    entries: List[Dict[str, Any]] = []
    for p_idx, profile in enumerate(PROFILES, start=1):
        for q_idx, qt in enumerate(QUESTION_TEMPLATES, start=1):
            qid = f"{profile['label']}_{qt['key']}"
            entry = {
                "id": qid,
                "question": qt["text"],
                "question_type": qt["type"],
                "profile_label": profile["label"],
                "profile_description": profile["description"],
                "profile": profile["data"],
                "expected_keywords": qt["expected_keywords"],
                "ground_truth_requirements": qt["ground_truth_requirements"],
                # Verdict label only for overall-eligibility-style questions where
                # the profile-level expected_class is meaningful. Other questions
                # are scored on retrieval/keyword metrics, not verdict agreement.
                "expected_verdict": (
                    profile["expected_class"]
                    if qt["type"] in ("multi_eligibility", "single_eligibility")
                    else None
                ),
            }
            entries.append(entry)
    return entries


def main() -> None:
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    entries = build_entries()
    with OUT_PATH.open("w", encoding="utf-8") as f:
        for e in entries:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")
    print(f"Wrote {len(entries)} test cases to {OUT_PATH}")
    # Print distribution by type
    from collections import Counter
    types = Counter(e["question_type"] for e in entries)
    print("By type:", dict(types))
    profiles = Counter(e["profile_label"] for e in entries)
    print(f"By profile: {len(profiles)} profiles × {len(entries) // len(profiles)} questions each")


if __name__ == "__main__":
    main()
