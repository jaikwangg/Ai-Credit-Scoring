#!/usr/bin/env python3
"""
CIMB Thai Home Loan Corpus Cleaner
====================================
Cleans scraped .txt documents and produces:
  1. Cleaned .txt files (nav/footer stripped, deduped)
  2. Cleaned JSON manifest (fixed titles, dates flagged, irrelevant docs removed)
  3. A cleaning report
"""

import os
import json
import re
import shutil
from pathlib import Path
from datetime import datetime

# ─── Paths ───────────────────────────────────────────────────────────────────
SRC_DIR   = Path("/sessions/inspiring-focused-johnson/mnt/data/documents")
OUT_DIR   = Path("/sessions/inspiring-focused-johnson/mnt/data/documents_cleaned")
REPORT_PATH = Path("/sessions/inspiring-focused-johnson/mnt/data/cleaning_report.json")
MANIFEST_IN  = None   # We'll embed the manifest below
MANIFEST_OUT = Path("/sessions/inspiring-focused-johnson/mnt/data/manifest_cleaned.json")

OUT_DIR.mkdir(parents=True, exist_ok=True)

# ─── Documents to REMOVE from corpus ────────────────────────────────────────
# Not relevant to home loan Q&A
REMOVE_IDS = {
    "privacy-notice-personal-th",          # General PDPA privacy notice
    "privacy-notice-announcement-th-2022", # Duplicate/outdated privacy
    "customer-profiling-notice-th-2025",   # Online banking security feature, not loan
    "customer-support-measures-covid-th-2021",          # COVID 2021, outdated
    "special-relief-assistance-measures-phase2-th-2020", # COVID 2020, outdated
}

# ─── Known-wrong publication dates (crawl date used instead of actual date) ──
# These documents show 2026-02-27 but their actual dates are from their IDs/content
DATE_OVERRIDES = {
    "debtor-management-policies-notice-th-2020":         "2020-01-01T00:00:00+00:00",
    "feedback-or-complaint-notice-th-2020":              "2020-01-01T00:00:00+00:00",
    "special-relief-assistance-measures-phase2-th-2020": "2020-01-01T00:00:00+00:00",
    "customer-support-measures-covid-th-2021":           "2021-01-01T00:00:00+00:00",
    "privacy-notice-announcement-th-2022":               "2022-05-30T08:40:02+00:00",
}

# ─── Nav/footer noise patterns ───────────────────────────────────────────────
# These lines appear as the repeating CIMB site nav menu
NAV_LINES = {
    "ปิด", "Search", "เลือกเพื่อเข้าสู่ระบบ", "BizChannel@CIMB",
    "เกี่ยวกับเรา", "ผลิตภัณฑ์ธนาคาร", "การบริหารความมั่งคั่ง",
    "โปรโมชั่น", "ช่องทางบริการธนาคาร", "บริการช่วยเหลือ",
    "การกำกับดูแล", "ทีมผู้บริหาร", "รางวัล", "นักลงทุนสัมพันธ์",
    "ความยั่งยืน", "ข่าวและกิจกรรม",
    "เงินฝาก", "บัตร", "ประกัน", "สินเชื่อ", "บริการโอนเงินระหว่างประเทศ",
    "การลงทุน", "โปรโมชั่นล่าสุด",
    "we love bond concert", "CIMB My Bond ผู้ช่วยส่วนตัว 24 ชั่วโมง",
    "CIMB THAI App", "CIMB THAI Connect", "บริการแจ้งเตือนผ่าน SMS", "พร้อมเพย์",
    "ติดต่อเรา", "สาขาธนาคาร", "ข้อมูลคุณภาพการให้บริการ",
    "อัตราและค่าธรรมเนียม", "Form Download Center",
    "โครงสร้างการกำกับดูแลกิจการ", "นโยบายการกำกับดูแลกิจการ",
    "นโยบายการต่อต้านการคอร์รัปชั่น", "หนังสือ บริคณห์สนธิ",
    "ข้อบังคับ", "จรรยาบรรณธนาคาร", "กฎบัตรคณะกรรมการ", "จรรยาบรรณกรรมการ",
    "คณะกรรมการ", "ผู้บริหารระดับสูง",
    "ข้อมูลทางการเงิน", "ข่าวแจ้งตลาดหลักทรัพย์", "บริการผูัถือหุ้น",
    "การกำกับดูแลและความเสี่ยง",
    "ผลิตภัณฑ์เพื่อธุรกิจและการธนาคารที่ยั่งยืน",
    "การขับเคลื่อนด้วยการดำเนินการอย่างยั่งยืน",
    "การมีส่วนร่วมและการสนับสนุนของผู้มีส่วนได้ส่วนเสีย",
    "บัญชีเงินฝากออมทรัพย์", "บัญชีเงินฝากประจำ", "บัญชีเงินฝากกระแสรายวัน",
    "บัญชีเงินฝากเงินตราต่างประเทศ", "ตารางเปรียบเทียบผลิตภัณฑ์",
    "บัตรเดบิต ซีไอเอ็มบี ไทย (รองรับมาตรฐานชิปการ์ดไทย)",
    "ประกันชีวิต", "ประกันวินาศภัย", "สินเชื่อบุคคล",
    "สินเชื่อบ้านแลกเงินและสินเชื่ออเนกประสงค์", "ผลิตภัณฑ์การลงทุน",
    "อัตราแลกเปลี่ยนเงินตราต่างประเทศ", "อัตราดอกเบี้ยเงินฝาก",
    "อัตราดอกเบี้ยเงินฝากลูกค้าสถาบัน", "อัตราดอกเบี้ยบัญชีเงินฝากเงินตราต่างประเทศ",
    "อัตราดอกเบี้ยเงินกู้",
    "กำหนดระยะเวลาการขายหรือฝากเงินได้ที่เป็นเงินตราต่างประเทศ",
    "ค่าธรรมเนียม", "อัตราค่าธรรมเนียมการฝากถอนบัญชีเงินฝากเงินตราต่างประเทศ",
    "ข้อกำหนดบัญชีเงินฝาก",
    "เงื่อนไขและค่าธรรมเนียมที่เกี่ยวกับการให้บริการบัญชีเงินฝากเงินตราต่างประเทศ",
    "ข้อกำหนดและเงื่อนไขการใช้บริการ Debit card",
    "บริการฝากเงินเข้าบัญชีธนาคาร ซีไอเอ็มบี ไทย ที่ตู้บุญเติม",
    "Quicklinks", "ประกาศสำคัญ", "สินทรัพทย์ธนาคาร", "ร่วมงานกับเรา",
    "Back", "Personal", "ลูกค้าบุคคล", "Preferred", "Business Banking",
    "ลูกค้าธุรกิจ", "Group", "กลุ่มซีไอเอ็มบี", "Language", "ไทย", "EN",
    "Countries", "CIMB Malaysia", "CIMB Indonesia", "CIMB Singapore",
    "CIMB Thailand", "CIMB Cambodia", "CIMB Vietnam", "CIMB Philippines",
    "See All เกี่ยวกับเรา", "See All รางวัล", "See All ข่าวและกิจกรรม",
    "See All บริการโอนเงินระหว่างประเทศ", "See All โปรโมชั่นล่าสุด",
    "See All we love bond concert", "See All CIMB My Bond ผู้ช่วยส่วนตัว 24 ชั่วโมง",
    "See All CIMB THAI App", "See All CIMB THAI Connect",
    "See All บริการแจ้งเตือนผ่าน SMS", "See All พร้อมเพย์",
    "See All บริการเปิดบัญชีด้วยการยืนยันตัวตนรูปแบบดิจิทัล (NDID)",
    "See All ติดต่อเรา", "See All สาขาธนาคาร",
    "See All ข้อมูลคุณภาพการให้บริการ",
    "See All คำมั่นสัญญาการให้บริการลูกค้าธนาคาร ซีไอเอ็มบี ไทย",
    "See All Form Download Center",
    "You're viewing:", "Other Sites", "TH",
    "CIMB Preferred", "สินทรัพย์ธนาคาร", "WORKING AT CIMB",
    "ความรับผิดชอบต่อสังคม", "ประกาศความเป็นส่วนตัว",
    "ประกาศการเก็บข้อมูลคุกกี้", "ข้อตกลงการใช้บริการ",
    "ปฏิทินธนาคาร ซีไอเอ็มบี ไทย", "Sitemap", "สำนักงานทั่วโลก",
    "CIMB", "CIMB Islamic", "CIMB Bank (MY)", "CIMB Bank (SG)",
    "CIMB Bank (KH)", "CIMB Niaga", "CIMB Bank (VN)", "CIMB Bank (PH)",
    "All rights reserved. Copyright © {0} CIMB THAI Bank",
    # Channel items with "See All" prefix
    "See All การขอและรับส่งข้อมูลรายการเคลื่อนไหวบัญชีเงินฝาก ในรูปแบบข้อมูลดิจิทัลระหว่างธนาคาร (dStatement)",
    "See All บริการยืนยันตัวตนรูปแบบดิจิทัล (NDID) เพื่อทำธรุกรรมออนไลน์กับกรมสรรพากร",
    "See All อัตราและค่าธรรมเนียม",
}

# Regex patterns for nav lines (prefix-based)
NAV_PREFIXES = [
    r"^See All ",
    r"^CIMB Bank \(",
]

# Lines that mark start of actual page content (after nav block)
# Pattern: "ประกาศสำคัญ\nสินทรัพทย์ธนาคาร\nร่วมงานกับเรา\nTH" then skip any of CONTENT_PRE_SKIP lines
CONTENT_START_SEQUENCE = ["ประกาศสำคัญ", "สินทรัพทย์ธนาคาร", "ร่วมงานกับเรา", "TH"]

# Lines to skip right after the end of the start sequence (breadcrumbs / lang selector)
CONTENT_PRE_SKIP = {"-", "ไทย", "ประกาศ", "ประกาศสำคัญ", "TH"}

# Lines that mark start of footer (end of real content)
FOOTER_START_LINES = {
    "บัญชีเงินฝากออมทรัพย์",  # part of product footer nav
}


# ─── Helpers ─────────────────────────────────────────────────────────────────

def clean_title(title: str) -> str:
    """Remove ' | CIMB TH', ' | สินเชื่อบ้าน | CIMB TH', etc. from titles."""
    # Remove trailing ' | CIMB TH' variants
    title = re.sub(r'\s*\|\s*CIMB TH\s*$', '', title)
    title = re.sub(r'\s*\|\s*CIMB Thai\s*$', '', title)
    # Remove trailing ' | Important Notice 2021' etc.
    title = re.sub(r'\s*\|\s*Important Notice \d+\s*$', '', title)
    # Collapse multiple pipes - take first meaningful segment if still has pipes
    if ' | ' in title:
        parts = [p.strip() for p in title.split(' | ') if p.strip()]
        # Keep all parts that are not generic breadcrumbs
        GENERIC_PARTS = {"สินเชื่อบ้าน", "CIMB TH", "CIMB Thai", "อัตราและค่าธรรมเนียม",
                         "บริการช่วยเหลือ  CIMB TH", "บริการช่วยเหลือ"}
        parts = [p for p in parts if p not in GENERIC_PARTS]
        title = ' | '.join(parts) if parts else title
    return title.strip()


def is_nav_line(line: str) -> bool:
    """Check if a line is navigation noise."""
    stripped = line.strip()
    if stripped in NAV_LINES:
        return True
    for pat in NAV_PREFIXES:
        if re.match(pat, stripped):
            return True
    # Long channel service lines
    if "รูปแบบข้อมูลดิจิทัลระหว่างธนาคาร" in stripped:
        return True
    if "ยืนยันตัวตนรูปแบบดิจิทัล (NDID)" in stripped:
        return True
    return False


def extract_content_from_webpage(raw_lines: list[str]) -> str:
    """
    For web-scraped docs:
    1. Find content start: after the sequence ending with 'TH\nไทย\n'
    2. Find content end: before the footer nav starting with specific product nav
    3. Remove remaining nav lines within content
    """
    # Find content start: look for CONTENT_START_SEQUENCE
    start_idx = 0
    seq_len = len(CONTENT_START_SEQUENCE)
    for i in range(len(raw_lines) - seq_len):
        window = [raw_lines[j].strip() for j in range(i, i + seq_len)]
        if window == CONTENT_START_SEQUENCE:
            start_idx = i + seq_len  # content starts after "TH"
            # Skip any leading breadcrumb/lang-selector lines
            while start_idx < len(raw_lines) and raw_lines[start_idx].strip() in CONTENT_PRE_SKIP:
                start_idx += 1
            break

    if start_idx == 0:
        # No nav pattern found - probably already clean (PDF-extracted)
        content_lines = raw_lines
    else:
        content_lines = raw_lines[start_idx:]

    # Find content end: detect footer nav by seeing product nav pattern
    # Footer starts with a breadcrumb of current page followed by "เงินฝาก\nบัญชีเงินฝากออมทรัพย์"
    # Look for "เงินฝาก" line followed shortly by "บัญชีเงินฝากออมทรัพย์"
    end_idx = len(content_lines)
    for i in range(len(content_lines) - 1):
        a = content_lines[i].strip()
        b = content_lines[i + 1].strip()
        if a == "เงินฝาก" and b == "บัญชีเงินฝากออมทรัพย์":
            # Walk back to find start of this footer block (may have breadcrumb line)
            end_idx = i
            # Check if line before is current page breadcrumb
            if i > 0 and content_lines[i-1].strip() not in NAV_LINES:
                # Could be "โฮมโลนฟอร์ยู" or similar - include in cut
                end_idx = i - 1
            break
        # Also catch "Chat กับเรา\nลงทะเบียนให้ติดต่อกลับ" repeated pattern
        if a == "Chat กับเรา" and b == "ลงทะเบียนให้ติดต่อกลับ":
            # This marks end of real content / start of widget + footer
            # Find the SECOND occurrence (first is real CTA, second is duplicate)
            # Actually just keep the first one, cut at third instance
            pass

    content_lines = content_lines[:end_idx]

    # Remove remaining nav/boilerplate lines from within content
    cleaned = []
    prev_blank = False
    for line in content_lines:
        stripped = line.strip()
        if is_nav_line(stripped):
            continue
        # Remove template variables
        if re.match(r'^\{\{.*\}\}$', stripped):
            continue
        # Remove "ค้นหาเพิ่มเติม" standalone (navigation element)
        if stripped == "ค้นหาเพิ่มเติม":
            continue
        # Collapse multiple blank lines
        if stripped == "":
            if prev_blank:
                continue
            prev_blank = True
        else:
            prev_blank = False
        cleaned.append(stripped)

    # Remove leading/trailing blank lines
    while cleaned and cleaned[0] == "":
        cleaned.pop(0)
    while cleaned and cleaned[-1] == "":
        cleaned.pop()

    return "\n".join(cleaned)


def parse_doc_file(filepath: Path) -> dict:
    """Parse header metadata and body from a .txt doc file."""
    text = filepath.read_text(encoding="utf-8")
    lines = text.splitlines()

    meta = {}
    summary = ""
    content_lines = []
    state = "header"  # header → summary → content

    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()

        if state == "header":
            if stripped == "---":
                state = "summary_header"
            elif ": " in stripped:
                key, _, val = stripped.partition(": ")
                meta[key] = val
        elif state == "summary_header":
            if stripped.startswith("SUMMARY"):
                state = "summary"
            elif stripped == "FULL CLEANED TEXT CONTENT":
                state = "content"
        elif state == "summary":
            if stripped == "---":
                state = "content_header"
            else:
                summary += (" " if summary else "") + stripped
        elif state == "content_header":
            if stripped == "FULL CLEANED TEXT CONTENT":
                state = "content"
        elif state == "content":
            if stripped == "---":
                break  # end of file marker
            content_lines.append(line)
        i += 1

    return {
        "meta": meta,
        "summary": summary.strip(),
        "raw_content_lines": content_lines,
    }


def clean_doc_file(doc_id: str, filepath: Path) -> tuple[str, dict]:
    """Clean a single .txt document. Returns (cleaned_text, report_entry)."""
    parsed = parse_doc_file(filepath)
    raw_lines = parsed["raw_content_lines"]

    # Determine if this is a web-scraped doc (has nav) or PDF (clean)
    # Heuristic: web-scraped docs have "You're viewing:" in raw content
    is_web = any("You're viewing:" in l or "BizChannel@CIMB" in l for l in raw_lines)

    if is_web:
        cleaned_content = extract_content_from_webpage(raw_lines)
    else:
        # PDF-extracted: mostly clean, just normalize whitespace
        cleaned = []
        prev_blank = False
        for line in raw_lines:
            stripped = line.strip()
            if stripped == "":
                if prev_blank:
                    continue
                prev_blank = True
            else:
                prev_blank = False
            cleaned.append(stripped)
        while cleaned and cleaned[0] == "":
            cleaned.pop(0)
        while cleaned and cleaned[-1] == "":
            cleaned.pop()
        cleaned_content = "\n".join(cleaned)

    # Check if content is empty or just template vars
    is_empty = (
        not cleaned_content.strip() or
        re.match(r'^(\{\{[^}]+\}\}\s*)+$', cleaned_content.strip())
    )

    # Rebuild the file with clean format
    meta = parsed["meta"]
    summary = parsed["summary"]

    title = meta.get("TITLE", "")
    source_url = meta.get("SOURCE URL", "")
    institution = meta.get("INSTITUTION", "")
    pub_date = meta.get("PUBLICATION DATE", "")
    category = meta.get("CATEGORY", "")

    output = f"TITLE: {title}\n"
    output += f"SOURCE URL: {source_url}\n"
    output += f"INSTITUTION: {institution}\n"
    output += f"PUBLICATION DATE: {pub_date}\n"
    output += f"CATEGORY: {category}\n"
    output += "---\n"
    if summary:
        output += f"SUMMARY\n{summary}\n---\n"
    output += "FULL CLEANED TEXT CONTENT\n"
    output += cleaned_content + "\n"
    output += "---\n"

    raw_lines_count = len([l for l in raw_lines if l.strip()])
    clean_lines_count = len([l for l in cleaned_content.splitlines() if l.strip()])

    report_entry = {
        "id": doc_id,
        "is_web_scraped": is_web,
        "raw_content_lines": raw_lines_count,
        "cleaned_content_lines": clean_lines_count,
        "reduction_pct": round((1 - clean_lines_count / max(raw_lines_count, 1)) * 100, 1),
        "is_empty_after_clean": is_empty,
    }

    return output, report_entry


# ─── Manifest JSON embedded ──────────────────────────────────────────────────
MANIFEST = {
  "corpus": "thai_home_loan_ai/cimb",
  "document_count": 29,
  "documents": [
    {"id": "home-loan-overview-th", "title": "โฮมโลน | สินเชื่อบ้าน | CIMB TH", "source_url": "https://www.cimbthai.com/th/personal/products/loans/home-loan.html", "institution": "CIMB Thai", "publication_date": "2025-10-07T09:36:20+00:00", "category": "eligibility_rule", "file_path": "corpus/thai_home_loan_ai/cimb/home-loan-overview-th.txt"},
    {"id": "home-loan-4u-th", "title": "โฮมโลนฟอร์ยู | สินเชื่อบ้าน | CIMB TH", "source_url": "https://www.cimbthai.com/th/personal/products/loans/home-loan/home-loan-4u.html", "institution": "CIMB Thai", "publication_date": "2026-02-23T07:36:18+00:00", "category": "interest_structure", "file_path": "corpus/thai_home_loan_ai/cimb/home-loan-4u-th.txt"},
    {"id": "home-loan-refinance-th", "title": "รีไฟแนนซ์ | สินเชื่อบ้าน | CIMB TH", "source_url": "https://www.cimbthai.com/th/personal/products/loans/home-loan/refinance.html", "institution": "CIMB Thai", "publication_date": "2026-01-05T13:26:17+00:00", "category": "bank_policy", "file_path": "corpus/thai_home_loan_ai/cimb/home-loan-refinance-th.txt"},
    {"id": "loan-interest-rates-th", "title": "อัตราดอกเบี้ยเงินกู้ | อัตราและค่าธรรมเนียม | CIMB TH", "source_url": "https://www.cimbthai.com/th/personal/help-support/rates-charges/loan-interest-rates.html", "institution": "CIMB Thai", "publication_date": "2025-12-29T08:31:17+00:00", "category": "interest_structure", "file_path": "corpus/thai_home_loan_ai/cimb/loan-interest-rates-th.txt"},
    {"id": "service-fees-th", "title": "ค่าธรรมเนียมบริการ", "source_url": "https://www.cimbthai.com/th/personal/help-support/rates-charges/fees/service-fees.html", "institution": "CIMB Thai", "publication_date": "2026-01-30T08:39:35+00:00", "category": "bank_policy", "file_path": "corpus/thai_home_loan_ai/cimb/service-fees-th.txt"},
    {"id": "form-download-center-th", "title": "Form Download Center", "source_url": "https://www.cimbthai.com/th/personal/help-support/form_download_center.html", "institution": "CIMB Thai", "publication_date": "2026-01-23T09:13:38+00:00", "category": "consumer_guideline", "file_path": "corpus/thai_home_loan_ai/cimb/form-download-center-th.txt"},
    {"id": "fire-insurance-secured-loans-th", "title": "ประกันอัคคีภัย สำหรับสินเชื่อมีหลักประกัน | CIMB TH", "source_url": "https://www.cimbthai.com/th/personal/products/insurance/non-life-insurances/cimb-thai-fire-insurance-for-secured-loans.html", "institution": "CIMB Thai", "publication_date": "2025-06-04T04:23:18+00:00", "category": "bank_policy", "file_path": "corpus/thai_home_loan_ai/cimb/fire-insurance-secured-loans-th.txt"},
    {"id": "service-sla-th", "title": "ข้อมูลคุณภาพการให้บริการ | บริการช่วยเหลือ  CIMB TH", "source_url": "https://www.cimbthai.com/th/personal/help-support/service-sla.html", "institution": "CIMB Thai", "publication_date": "2026-01-26T03:12:21+00:00", "category": "consumer_guideline", "file_path": "corpus/thai_home_loan_ai/cimb/service-sla-th.txt"},
    {"id": "tcf-commitment-th", "title": "คำมั่นสัญญาการให้บริการลูกค้าธนาคาร ซีไอเอ็มบี ไทย", "source_url": "https://www.cimbthai.com/th/personal/help-support/cimb-thai-s-treating-customers-fairly-commitment.html", "institution": "CIMB Thai", "publication_date": "2021-04-02T03:50:24+00:00", "category": "consumer_guideline", "file_path": "corpus/thai_home_loan_ai/cimb/tcf-commitment-th.txt"},
    {"id": "responsible-lending-notice-th", "title": "การให้สินเชื่ออย่างรับผิดชอบและเป็นธรรม (Responsible Lending)", "source_url": "https://www.cimbthai.com/th/personal/important-notices/2024/responsible-lending.html", "institution": "CIMB Thai", "publication_date": "2024-07-02T04:59:13+00:00", "category": "regulation", "file_path": "corpus/thai_home_loan_ai/cimb/responsible-lending-notice-th.txt"},
    {"id": "flood-relief-measures-th-2024", "title": "มาตรการช่วยเหลือผู้ประสบภัยน้ำท่วมของธนาคาร ซีไอเอ็มบี ไทย จำกัด (มหาชน)", "source_url": "https://www.cimbthai.com/th/personal/important-notices/2024/flood-relief-measures.html", "institution": "CIMB Thai", "publication_date": "2024-09-30T08:41:17+00:00", "category": "consumer_guideline", "file_path": "corpus/thai_home_loan_ai/cimb/flood-relief-measures-th-2024.txt"},
    {"id": "relief-measures-border-flooding-th-2025", "title": "มาตรการช่วยเหลือผู้ประสบภัยชายแดนไทย-กัมพูชา และอุทกภัยภาคเหนือ", "source_url": "https://www.cimbthai.com/th/personal/important-notices/2025/relief-measures-border-areas-and-flooding.html", "institution": "CIMB Thai", "publication_date": "2025-07-30T03:24:30+00:00", "category": "consumer_guideline", "file_path": "corpus/thai_home_loan_ai/cimb/relief-measures-border-flooding-th-2025.txt"},
    {"id": "privacy-notice-personal-th", "title": "ประกาศความเป็นส่วนตัว  |  ซีไอเอ็มบี ไทย | CIMB TH", "source_url": "https://www.cimbthai.com/th/personal/privacy-notice.html", "institution": "CIMB Thai", "publication_date": "2025-03-26T09:04:12+00:00", "category": "regulation", "file_path": "corpus/thai_home_loan_ai/cimb/privacy-notice-personal-th.txt"},
    {"id": "privacy-notice-announcement-th-2022", "title": "การประมวลผลข้อมูลส่วนบุคคลของท่าน", "source_url": "https://www.cimbthai.com/th/personal/important-notices/2022/annoucement-privacynotices.html", "institution": "CIMB Thai", "publication_date": "2022-05-30T08:40:02+00:00", "category": "regulation", "file_path": "corpus/thai_home_loan_ai/cimb/privacy-notice-announcement-th-2022.txt"},
    {"id": "customer-profiling-notice-th-2025", "title": "Customer Profiling ฟีเจอร์ใหม่เพิ่มความปลอดภัยให้ธุรกรรมออนไลน์", "source_url": "https://www.cimbthai.com/th/personal/important-notices/2025/customer-profiling.html", "institution": "CIMB Thai", "publication_date": "2025-09-26T10:29:54+00:00", "category": "bank_policy", "file_path": "corpus/thai_home_loan_ai/cimb/customer-profiling-notice-th-2025.txt"},
    {"id": "home-loan-certification-app-th", "title": "การขอหนังสือรับรองดอกเบี้ยบ้านผ่าน CIMB THAI App | CIMB TH", "source_url": "https://www.cimbthai.com/th/personal/ways-to-bank/cimb-thai-digital-banking/how-to-request-home-loan-certification-through-app-cimb-thai-digital-banking.html", "institution": "CIMB Thai", "publication_date": "2025-11-13T04:24:16+00:00", "category": "consumer_guideline", "file_path": "corpus/thai_home_loan_ai/cimb/home-loan-certification-app-th.txt"},
    {"id": "home-loan-generic-rate-th-2026", "title": "อัตราดอกเบี้ยสินเชื่อบ้านใหม่ (Generic) ปี 2568/2569", "source_url": "https://www.cimbthai.com/content/dam/cimbth/personal/documents/loan/homeloan/2026/%E0%B8%94%E0%B8%AD%E0%B8%81%E0%B9%80%E0%B8%9A%E0%B8%B5%E0%B9%89%E0%B8%A2%20Generic%20-%20%E0%B8%9A%E0%B9%89%E0%B8%B2%E0%B8%99%E0%B9%83%E0%B8%AB%E0%B8%A1%E0%B9%881-Th_261268.pdf", "institution": "CIMB Thai", "publication_date": "2026-02-27T22:34:11+00:00", "category": "interest_structure", "file_path": "corpus/thai_home_loan_ai/cimb/home-loan-generic-rate-th-2026.txt"},
    {"id": "refinance-generic-rate-plan1-th-2026", "title": "อัตราดอกเบี้ยรีไฟแนนซ์ Generic แผน 1 ปี 2568/2569", "source_url": "https://www.cimbthai.com/content/dam/cimbth/personal/documents/loan/refinance/2026/%E0%B8%94%E0%B8%AD%E0%B8%81%E0%B9%80%E0%B8%9A%E0%B8%B5%E0%B9%89%E0%B8%A2%20Generic%20-%20%E0%B8%A3%E0%B8%B5%E0%B9%84%E0%B8%9F%E0%B9%81%E0%B8%99%E0%B8%99%E0%B8%8B%E0%B9%8C1-Th_261268.pdf", "institution": "CIMB Thai", "publication_date": "2026-02-27T22:34:11+00:00", "category": "interest_structure", "file_path": "corpus/thai_home_loan_ai/cimb/refinance-generic-rate-plan1-th-2026.txt"},
    {"id": "refinance-generic-rate-plan2-th-2026", "title": "อัตราดอกเบี้ยรีไฟแนนซ์ Generic แผน 2 ปี 2568/2569", "source_url": "https://www.cimbthai.com/content/dam/cimbth/personal/documents/loan/refinance/2026/%E0%B8%94%E0%B8%AD%E0%B8%81%E0%B9%80%E0%B8%9A%E0%B8%B5%E0%B9%89%E0%B8%A2%20Generic%20-%20%E0%B8%A3%E0%B8%B5%E0%B9%84%E0%B8%9F%E0%B9%81%E0%B8%99%E0%B8%99%E0%B8%8B%E0%B9%8C2-Th_261268.pdf", "institution": "CIMB Thai", "publication_date": "2026-02-27T22:34:11+00:00", "category": "interest_structure", "file_path": "corpus/thai_home_loan_ai/cimb/refinance-generic-rate-plan2-th-2026.txt"},
    {"id": "mortgage-power-generic-rate-th-2026", "title": "อัตราดอกเบี้ย Mortgage Power (สินเชื่อบ้านแลกเงิน) ปี 2569", "source_url": "https://www.cimbthai.com/content/dam/cimbth/personal/documents/loan/mortgage/feb/%E0%B8%94%E0%B8%AD%E0%B8%81%E0%B9%80%E0%B8%9A%E0%B8%B5%E0%B9%89%E0%B8%A2%20Generic%20-%20%E0%B8%A1%E0%B8%AD%E0%B8%A3%E0%B9%8C%E0%B9%80%E0%B8%81%E0%B8%88%E0%B8%9E%E0%B8%B2%E0%B8%A7%E0%B9%80%E0%B8%A7%E0%B8%AD%E0%B8%A3%E0%B9%8C-Th_090269.pdf", "institution": "CIMB Thai", "publication_date": "2026-02-27T22:34:12+00:00", "category": "interest_structure", "file_path": "corpus/thai_home_loan_ai/cimb/mortgage-power-generic-rate-th-2026.txt"},
    {"id": "property-loan-generic-rate-th-2026", "title": "อัตราดอกเบี้ย Property Loan ปี 2568/2569", "source_url": "https://www.cimbthai.com/content/dam/cimbth/personal/documents/loan/Poperty-loan/2026/%E0%B8%94%E0%B8%AD%E0%B8%81%E0%B9%80%E0%B8%9A%E0%B8%B5%E0%B9%89%E0%B8%A2%20Generic%20-%20%E0%B8%9E%E0%B8%A3%E0%B9%87%E0%B8%AD%E0%B8%9E%E0%B9%80%E0%B8%9E%E0%B8%AD%E0%B8%A3%E0%B9%8C%E0%B8%95%E0%B8%B5%E0%B9%89-Th_261268.pdf", "institution": "CIMB Thai", "publication_date": "2026-02-27T22:34:12+00:00", "category": "interest_structure", "file_path": "corpus/thai_home_loan_ai/cimb/property-loan-generic-rate-th-2026.txt"},
    {"id": "property-loan-sales-sheet-th-2026", "title": "Sales Sheet Property Loan (เงื่อนไขผลิตภัณฑ์) ปี 2568", "source_url": "https://www.cimbthai.com/content/dam/cimbth/personal/documents/loan/Poperty-loan/2026/Sales%20sheet%20PY_Eff%20180868_080868.pdf", "institution": "CIMB Thai", "publication_date": "2026-02-27T22:34:14+00:00", "category": "consumer_guideline", "file_path": "corpus/thai_home_loan_ai/cimb/property-loan-sales-sheet-th-2026.txt"},
    {"id": "loan-close-account-collateral-release-form-th-2025", "title": "แบบฟอร์มยกเลิกวงเงิน/ปิดบัญชี/ไถ่ถอนหลักประกัน (สินเชื่อ)", "source_url": "https://www.cimbthai.com/content/dam/cimbth/personal/documents/form-download-center/2025/...", "institution": "CIMB Thai", "publication_date": "2026-02-27T22:34:14+00:00", "category": "consumer_guideline", "file_path": "corpus/thai_home_loan_ai/cimb/loan-close-account-collateral-release-form-th-2025.txt"},
    {"id": "debt-restructuring-extension-form-th-2025", "title": "ใบคำขอปรับปรุงโครงสร้างหนี้ (ขยายระยะเวลาผ่อน)", "source_url": "https://www.cimbthai.com/content/dam/cimbth/personal/documents/form-download-center/2025/...", "institution": "CIMB Thai", "publication_date": "2026-02-27T22:34:14+00:00", "category": "consumer_guideline", "file_path": "corpus/thai_home_loan_ai/cimb/debt-restructuring-extension-form-th-2025.txt"},
    {"id": "loan-interest-disclosure-consent-form-th-2026", "title": "หนังสือยินยอมเปิดเผยข้อมูลดอกเบี้ยเงินกู้ (CIMB THAI)", "source_url": "https://www.cimbthai.com/content/dam/cimbth/personal/documents/form-download-center/2026/...", "institution": "CIMB Thai", "publication_date": "2026-02-27T22:34:15+00:00", "category": "regulation", "file_path": "corpus/thai_home_loan_ai/cimb/loan-interest-disclosure-consent-form-th-2026.txt"},
    {"id": "debtor-management-policies-notice-th-2020", "title": "ประกาศชี้แจงนโยบายและขั้นตอนการบริหารจัดการลูกหนี้ของธนาคาร", "source_url": "https://www.cimbthai.com/th/personal/important-notices/2020/clarification-of-cimb-thai-bank-plcs-debtor-management-policies-and-procedures.html", "institution": "CIMB Thai", "publication_date": "2026-02-27T22:34:15+00:00", "category": "bank_policy", "file_path": "corpus/thai_home_loan_ai/cimb/debtor-management-policies-notice-th-2020.txt"},
    {"id": "feedback-or-complaint-notice-th-2020", "title": "ช่องทางเสนอแนะหรือร้องเรียน | CIMB Thai Important Notice", "source_url": "https://www.cimbthai.com/th/personal/important-notices/2020/feedback-or-complaint.html", "institution": "CIMB Thai", "publication_date": "2026-02-27T22:34:16+00:00", "category": "consumer_guideline", "file_path": "corpus/thai_home_loan_ai/cimb/feedback-or-complaint-notice-th-2020.txt"},
    {"id": "special-relief-assistance-measures-phase2-th-2020", "title": "มาตรการช่วยเหลือลูกหนี้ระยะที่ 2 (Special Relief Assistance Measures)", "source_url": "https://www.cimbthai.com/th/personal/important-notices/2020/special-relief-assistance-measures-phase-2.html", "institution": "CIMB Thai", "publication_date": "2026-02-27T22:34:16+00:00", "category": "consumer_guideline", "file_path": "corpus/thai_home_loan_ai/cimb/special-relief-assistance-measures-phase2-th-2020.txt"},
    {"id": "customer-support-measures-covid-th-2021", "title": "มาตรการช่วยเหลือลูกค้า (COVID-19) | Important Notice 2021", "source_url": "https://www.cimbthai.com/th/personal/important-notices/2021/customer-support-measures-covid.html", "institution": "CIMB Thai", "publication_date": "2026-02-27T22:34:16+00:00", "category": "consumer_guideline", "file_path": "corpus/thai_home_loan_ai/cimb/customer-support-measures-covid-th-2021.txt"},
  ]
}

# Category fixes based on content analysis
CATEGORY_FIXES = {
    "home-loan-refinance-th": "interest_structure",   # was bank_policy, but it's a product page with rates
    "service-fees-th": "bank_policy",                 # OK as-is
    "debtor-management-policies-notice-th-2020": "regulation",  # was bank_policy, more of a regulatory notice
}


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    report = {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "removed_docs": [],
        "empty_after_clean": [],
        "date_fixes": [],
        "category_fixes": [],
        "title_fixes": [],
        "cleaned_docs": [],
    }

    cleaned_manifest_docs = []

    for doc in MANIFEST["documents"]:
        doc_id = doc["id"]

        # 1. Remove irrelevant docs
        if doc_id in REMOVE_IDS:
            report["removed_docs"].append({"id": doc_id, "reason": "not relevant to home loan corpus"})
            continue

        # 2. Process the .txt file
        src_file = SRC_DIR / f"{doc_id}.txt"
        if not src_file.exists():
            print(f"WARNING: {src_file} not found, skipping")
            continue

        cleaned_text, file_report = clean_doc_file(doc_id, src_file)
        report["cleaned_docs"].append(file_report)

        if file_report["is_empty_after_clean"]:
            report["empty_after_clean"].append(doc_id)
            print(f"  ⚠️  {doc_id}: EMPTY after cleaning")

        # Write cleaned file
        out_file = OUT_DIR / f"{doc_id}.txt"
        out_file.write_text(cleaned_text, encoding="utf-8")

        # 3. Build cleaned manifest entry
        new_doc = dict(doc)

        # Fix title
        clean = clean_title(doc["title"])
        # Additional cleanup: strip trailing " | CIMB Thai Important Notice" variants
        clean = re.sub(r'\s*\|\s*CIMB Thai Important Notice.*$', '', clean).strip()
        if clean != doc["title"]:
            report["title_fixes"].append({"id": doc_id, "before": doc["title"], "after": clean})
            new_doc["title"] = clean

        # Fix publication date
        if doc_id in DATE_OVERRIDES:
            old = doc["publication_date"]
            new_doc["publication_date"] = DATE_OVERRIDES[doc_id]
            report["date_fixes"].append({"id": doc_id, "before": old, "after": DATE_OVERRIDES[doc_id]})

        # Fix category
        if doc_id in CATEGORY_FIXES:
            old = doc["category"]
            new_doc["category"] = CATEGORY_FIXES[doc_id]
            report["category_fixes"].append({"id": doc_id, "before": old, "after": CATEGORY_FIXES[doc_id]})

        # Flag home-loan-overview as needing re-scrape (content is unrendered template vars)
        if file_report["is_empty_after_clean"] or doc_id == "home-loan-overview-th":
            new_doc["needs_rescrape"] = True
            if doc_id == "home-loan-overview-th":
                report.setdefault("needs_rescrape", []).append({
                    "id": doc_id,
                    "reason": "page content is unrendered template variables (JS-rendered page); needs re-scrape with headless browser"
                })

        # Update file_path to point to cleaned dir
        new_doc["file_path"] = f"corpus/thai_home_loan_ai/cimb/cleaned/{doc_id}.txt"

        cleaned_manifest_docs.append(new_doc)

    # 4. Write cleaned manifest
    cleaned_manifest = {
        "corpus": MANIFEST["corpus"],
        "document_count": len(cleaned_manifest_docs),
        "documents": cleaned_manifest_docs,
    }
    MANIFEST_OUT.write_text(json.dumps(cleaned_manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    # 5. Write report
    REPORT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    # 6. Print summary
    print("\n" + "="*60)
    print("CLEANING SUMMARY")
    print("="*60)
    print(f"Total input docs:    {len(MANIFEST['documents'])}")
    print(f"Removed (irrelevant): {len(report['removed_docs'])}")
    print(f"Output docs:         {len(cleaned_manifest_docs)}")
    print(f"Empty after clean:   {len(report['empty_after_clean'])}")
    print(f"Title fixes:         {len(report['title_fixes'])}")
    print(f"Date fixes:          {len(report['date_fixes'])}")
    print(f"Category fixes:      {len(report['category_fixes'])}")
    print()

    if report["removed_docs"]:
        print("Removed docs:")
        for d in report["removed_docs"]:
            print(f"  - {d['id']}")

    if report["empty_after_clean"]:
        print("\nEmpty docs (consider manual review or re-scrape):")
        for d in report["empty_after_clean"]:
            print(f"  ⚠️  {d}")

    print("\nNoise reduction per doc:")
    for d in report["cleaned_docs"]:
        marker = "⚠️ EMPTY" if d["is_empty_after_clean"] else ""
        print(f"  {d['id']:55s} {d['raw_content_lines']:4d} → {d['cleaned_content_lines']:4d} lines  (-{d['reduction_pct']}%) {marker}")

    print(f"\nOutput written to: {OUT_DIR}")
    print(f"Manifest: {MANIFEST_OUT}")
    print(f"Report:   {REPORT_PATH}")


if __name__ == "__main__":
    main()
