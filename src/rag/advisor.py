"""Profile-conditioned RAG advisor.

The vanilla /rag/query endpoint paraphrases retrieved chunks. This module
implements *reasoning* on top of retrieval: it asks the LLM to extract
eligibility requirements from policy chunks and then evaluate them against
a concrete user profile, producing a structured verdict instead of prose.

Why this matters for the thesis:
- Vanilla RAG is "retrieval-augmented summarisation".
- Profile-conditioned RAG is the smallest hop into "retrieval-augmented
  reasoning" — measurable as a contribution.

Output is a structured AdvisorResponse so the frontend can render it as a
checklist of pass/fail rows rather than as wall-of-text.
"""
from __future__ import annotations

import json
import logging
import re
import time
from typing import Any, Dict, List, Optional

from llama_index.core.settings import Settings

from src.api.schemas.payload import (
    AdvisorProfile,
    AdvisorReasoningTrace,
    AdvisorRequirementCheck,
    AdvisorResponse,
    RAGSource,
)
from src.rag.multihop import multihop_retrieve

logger = logging.getLogger(__name__)


# IsSup reflection prompt — borrowed from src/rag/self_rag.py and adapted for
# advisor JSON output. Returns 1-5 grounding score.
ISSUP_PROMPT = """\
ประเมินว่าคำตอบ JSON ของระบบแอดไวเซอร์ด้านล่างมี "หลักฐานสนับสนุน" จากเอกสารบริบทมากแค่ไหน

คำถาม: {question}

บริบท (เอกสารที่ retrieve ได้):
{context}

JSON ที่ระบบสร้างขึ้น:
{answer}

ให้คะแนน 1-5:
- 5 = ทุกการอ้างอิงและตัวเลขในคำตอบสามารถยืนยันได้จากบริบท
- 4 = เกือบทั้งหมดยืนยันได้ มีจุดเล็กน้อยที่อนุมาน
- 3 = ครึ่งหนึ่งยืนยันได้ ครึ่งหนึ่งเป็นการอนุมาน
- 2 = ส่วนใหญ่อนุมาน หลักฐานน้อย
- 1 = ไม่มีหลักฐานเลย ตอบจากความรู้ทั่วไป

ตอบกลับเป็นตัวเลขตัวเดียว (1, 2, 3, 4, หรือ 5) เท่านั้น
"""


# Map English profile field names to Thai display labels used in the prompt.
_PROFILE_LABELS_TH: Dict[str, str] = {
    "salary_per_month": "รายได้ต่อเดือน (บาท)",
    "occupation": "อาชีพ",
    "employment_tenure_months": "อายุงานปัจจุบัน (เดือน)",
    "marriage_status": "สถานภาพสมรส",
    "has_coapplicant": "มีผู้กู้ร่วม",
    "coapplicant_income": "รายได้ผู้กู้ร่วม (บาท)",
    "credit_score": "คะแนนเครดิต",
    "credit_grade": "เกรดเครดิต",
    "outstanding_debt": "หนี้คงค้าง (บาท)",
    "overdue_amount": "ยอดค้างชำระ (บาท)",
    "loan_amount_requested": "วงเงินขอกู้ (บาท)",
    "loan_term_years": "ระยะเวลากู้ (ปี)",
    "interest_rate": "อัตราดอกเบี้ย (%)",
}


def _format_profile_for_prompt(profile: AdvisorProfile) -> str:
    """Render the profile as a Thai bullet list, skipping unset fields."""
    lines: List[str] = []
    data = profile.model_dump()
    for field, label in _PROFILE_LABELS_TH.items():
        value = data.get(field)
        if value is None or value == "":
            continue
        if isinstance(value, bool):
            value = "มี" if value else "ไม่มี"
        if isinstance(value, float) and value.is_integer():
            value = int(value)
        lines.append(f"- {label}: {value}")
    if not lines:
        return "(ไม่มีข้อมูลผู้สมัคร — โปรดให้คำตอบทั่วไปจากเอกสารอย่างเดียว)"
    return "\n".join(lines)


def _build_context_block(sources: List[Dict[str, Any]], max_chars_per_source: int = 1200) -> str:
    """Render retrieved sources as a numbered context block for the LLM."""
    if not sources:
        return "(ไม่พบข้อมูลในเอกสารที่เกี่ยวข้อง)"
    blocks: List[str] = []
    for i, src in enumerate(sources, 1):
        meta = src.get("metadata", {}) or {}
        title = meta.get("title", "ไม่ระบุชื่อเอกสาร")
        category = meta.get("category", "")
        content = (src.get("content") or "").strip()
        if len(content) > max_chars_per_source:
            content = content[: max_chars_per_source - 3].rstrip() + "..."
        blocks.append(f"[เอกสาร {i}] {title} (หมวด: {category})\n{content}")
    return "\n\n".join(blocks)


PROMPT_TEMPLATE = """\
คุณเป็นที่ปรึกษาสินเชื่อมืออาชีพที่วิเคราะห์โปรไฟล์ผู้ขอสินเชื่อโดยอ้างอิงเอกสารนโยบายธนาคารจริง

คำถามของผู้ใช้:
{question}

โปรไฟล์ผู้ขอสินเชื่อ:
{profile}

เอกสารนโยบายที่เกี่ยวข้อง (จาก RAG retrieval):
{context}

คำสั่ง:
1. สกัด "เงื่อนไข/คุณสมบัติ" ที่เกี่ยวข้องจากเอกสารข้างต้น (เช่น รายได้ขั้นต่ำ, อายุ, อายุงาน, DSR, เครดิตบูโร ฯลฯ) — เฉพาะที่เกี่ยวกับคำถาม
2. สำหรับแต่ละเงื่อนไข ให้เปรียบเทียบกับโปรไฟล์ผู้ขอสินเชื่อจริง:
   - "pass" ถ้าผู้ใช้ผ่านเงื่อนไขนั้นชัดเจน
   - "fail" ถ้าผู้ใช้ไม่ผ่านเงื่อนไขนั้นชัดเจน
   - "unknown" ถ้าผู้ใช้ไม่ได้ระบุข้อมูลที่จำเป็น
   - "not_applicable" ถ้าเงื่อนไขนั้นไม่เกี่ยวกับโปรไฟล์ผู้ใช้นี้
3. ตัดสิน verdict ภาพรวม:
   - "eligible" ถ้า pass ทุกเงื่อนไขสำคัญ
   - "partially_eligible" ถ้า pass บางส่วน fail บางส่วน
   - "ineligible" ถ้า fail เงื่อนไขสำคัญ
   - "needs_more_info" ถ้า unknown มากกว่า pass+fail
4. แนะนำ 2-4 action ที่ผู้ใช้ทำได้จริงเพื่อปรับปรุงโอกาสอนุมัติ
5. ตอบเป็น JSON เท่านั้น ห้ามมีข้อความอื่นใด ห้ามใส่ markdown code fence

โครงสร้าง JSON ที่ต้องส่งกลับ:
{{
  "verdict": "eligible | partially_eligible | ineligible | needs_more_info",
  "verdict_summary": "สรุปสั้น 1-2 ประโยคเป็นภาษาไทย ระบุเหตุผลหลัก",
  "requirement_checks": [
    {{
      "requirement": "ชื่อเงื่อนไข เช่น รายได้ขั้นต่ำสำหรับพนักงานเอกชน",
      "user_value": "ค่าจริงของผู้ใช้ เช่น 18,000 บาท หรือ ไม่ระบุ",
      "status": "pass | fail | unknown | not_applicable",
      "explanation": "อธิบายสั้น 1 ประโยคว่าทำไม พร้อมอ้างเลขเอกสาร [เอกสาร N]"
    }}
  ],
  "recommended_actions": [
    "action ที่ 1",
    "action ที่ 2"
  ]
}}
"""


_VERDICT_VALUES = {"eligible", "partially_eligible", "ineligible", "needs_more_info"}
_STATUS_VALUES = {"pass", "fail", "unknown", "not_applicable"}


def _extract_json(text: str) -> Optional[dict]:
    """Best-effort extraction of a JSON object from an LLM response."""
    if not text:
        return None
    text = text.strip()
    # strip markdown code fences if present
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    # find first { ... last }
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    snippet = text[start : end + 1]
    try:
        return json.loads(snippet)
    except json.JSONDecodeError:
        # Try to repair common issues: trailing commas
        snippet2 = re.sub(r",(\s*[}\]])", r"\1", snippet)
        try:
            return json.loads(snippet2)
        except json.JSONDecodeError as exc:
            logger.warning("Advisor JSON parse failed: %s", exc)
            return None


def _normalize_verdict(value: str) -> str:
    v = (value or "").strip().lower().replace("-", "_")
    if v in _VERDICT_VALUES:
        return v
    if "ineligible" in v or "ไม่ผ่าน" in v:
        return "ineligible"
    if "partial" in v:
        return "partially_eligible"
    if "eligible" in v or "ผ่าน" in v:
        return "eligible"
    return "needs_more_info"


def _normalize_status(value: str) -> str:
    v = (value or "").strip().lower().replace("-", "_")
    if v in _STATUS_VALUES:
        return v
    if "pass" in v or "ผ่าน" in v:
        return "pass"
    if "fail" in v or "ไม่ผ่าน" in v:
        return "fail"
    if "n/a" in v or "ไม่เกี่ยว" in v:
        return "not_applicable"
    return "unknown"


def _issup_score(question: str, context_text: str, advisor_json: str) -> Optional[int]:
    """Run a single-shot Self-RAG IsSup reflection on the advisor output."""
    llm = getattr(Settings, "llm", None)
    if llm is None:
        return None
    prompt = ISSUP_PROMPT.format(
        question=question.strip(),
        context=context_text[:4000],
        answer=advisor_json[:3000],
    )
    try:
        raw = str(llm.complete(prompt)).strip()
    except Exception as exc:
        logger.warning("IsSup reflection failed: %s", exc)
        return None
    match = re.search(r"[1-5]", raw)
    return int(match.group()) if match else None


def run_advisor(
    question: str,
    profile: AdvisorProfile,
    rag_manager: Any,
    top_k: int = 6,
    use_multihop: bool = False,
    use_self_rag: bool = False,
) -> AdvisorResponse:
    """Run profile-conditioned advisory reasoning over the RAG index.

    Steps:
      1. Retrieve relevant policy chunks (single-hop or multi-hop).
      2. Build a structured prompt with question + profile + context.
      3. Ask LLM to return JSON with per-requirement pass/fail.
      4. Parse + normalise JSON. Fall back gracefully if LLM goes off-format.
      5. (Optional) Run Self-RAG IsSup reflection. If score < 3, retry with
         broader retrieval (top_k * 2) and re-synthesise.
    """
    t_start = time.monotonic()
    trace = AdvisorReasoningTrace(used_multihop=use_multihop, used_self_rag=use_self_rag)

    # Step 1: retrieve — single-hop or multi-hop
    if use_multihop:
        profile_text = _format_profile_for_prompt(profile)
        hop_result = multihop_retrieve(
            question=question,
            rag_manager=rag_manager,
            profile_text=profile_text,
            top_k_per_hop=max(3, top_k // 2),
            max_total_sources=max(top_k * 2, 8),
        )
        raw_sources: List[Dict[str, Any]] = hop_result["sources"]
        trace.sub_questions = hop_result["sub_questions"]
        trace.sources_per_hop = hop_result["per_hop_counts"]
        trace.total_sources_after_dedup = len(raw_sources)
    else:
        rag_result = rag_manager.query(question, similarity_top_k=top_k, include_sources=True)
        raw_sources = rag_result.get("sources", []) or []
        trace.total_sources_after_dedup = len(raw_sources)

    # Build sources list for the response
    response_sources: List[RAGSource] = []
    for src in raw_sources:
        meta = src.get("metadata", {}) or {}
        response_sources.append(
            RAGSource(
                title=meta.get("title", "Unknown"),
                category=meta.get("category", "Uncategorized"),
                institution=meta.get("institution"),
                score=src.get("score"),
            )
        )

    # Step 2: build prompt
    profile_block = _format_profile_for_prompt(profile)
    context_block = _build_context_block(raw_sources)
    prompt = PROMPT_TEMPLATE.format(
        question=question.strip(),
        profile=profile_block,
        context=context_block,
    )

    # Step 3: LLM call
    llm = getattr(Settings, "llm", None)
    if llm is None:
        return AdvisorResponse(
            question=question,
            verdict="needs_more_info",
            verdict_summary="LLM ไม่พร้อมใช้งาน — ไม่สามารถวิเคราะห์โปรไฟล์ได้",
            sources=response_sources,
        )

    try:
        raw_answer = str(llm.complete(prompt))
    except Exception as exc:
        logger.error("Advisor LLM call failed: %s", exc)
        return AdvisorResponse(
            question=question,
            verdict="needs_more_info",
            verdict_summary=f"เกิดข้อผิดพลาดระหว่างวิเคราะห์: {exc}",
            sources=response_sources,
        )

    # Step 4: parse JSON
    parsed = _extract_json(raw_answer)
    if not parsed:
        # Fallback: return the raw text as a single summary
        return AdvisorResponse(
            question=question,
            verdict="needs_more_info",
            verdict_summary="ระบบไม่สามารถสกัดผลการวิเคราะห์เป็นโครงสร้างได้",
            sources=response_sources,
            raw_answer=raw_answer.strip()[:2000],
        )

    verdict = _normalize_verdict(str(parsed.get("verdict", "")))
    summary = str(parsed.get("verdict_summary", "")).strip() or "ไม่มีสรุป"

    checks_raw = parsed.get("requirement_checks") or []
    checks: List[AdvisorRequirementCheck] = []
    if isinstance(checks_raw, list):
        for item in checks_raw[:10]:
            if not isinstance(item, dict):
                continue
            checks.append(
                AdvisorRequirementCheck(
                    requirement=str(item.get("requirement", "")).strip() or "ไม่ระบุ",
                    user_value=str(item.get("user_value", "")).strip() or "ไม่ระบุ",
                    status=_normalize_status(str(item.get("status", ""))),
                    explanation=str(item.get("explanation", "")).strip(),
                )
            )

    actions_raw = parsed.get("recommended_actions") or []
    actions: List[str] = []
    if isinstance(actions_raw, list):
        for a in actions_raw[:6]:
            text = str(a).strip()
            if text:
                actions.append(text)

    response = AdvisorResponse(
        question=question,
        verdict=verdict,
        verdict_summary=summary,
        requirement_checks=checks,
        recommended_actions=actions,
        sources=response_sources,
    )

    # Step 5: optional Self-RAG reflection
    if use_self_rag:
        score = _issup_score(question, context_block, raw_answer)
        trace.issup_score = score
        passed = (score is not None) and score >= 3
        trace.issup_passed = passed
        if score is not None and not passed:
            # Retry once with broader retrieval — only single-hop top_k bump,
            # multi-hop already widened things.
            logger.info("Advisor IsSup score=%d, retrying with wider retrieval", score)
            trace.self_rag_retried = True
            try:
                wider = rag_manager.query(
                    question, similarity_top_k=top_k * 2, include_sources=True
                )
                wider_sources = wider.get("sources", []) or []
                if len(wider_sources) > len(raw_sources):
                    response_sources_2 = []
                    for src in wider_sources:
                        meta = src.get("metadata", {}) or {}
                        response_sources_2.append(
                            RAGSource(
                                title=meta.get("title", "Unknown"),
                                category=meta.get("category", "Uncategorized"),
                                institution=meta.get("institution"),
                                score=src.get("score"),
                            )
                        )
                    context_block_2 = _build_context_block(wider_sources)
                    prompt_2 = PROMPT_TEMPLATE.format(
                        question=question.strip(),
                        profile=profile_block,
                        context=context_block_2,
                    )
                    raw_answer_2 = str(llm.complete(prompt_2))
                    parsed_2 = _extract_json(raw_answer_2)
                    if parsed_2:
                        verdict_2 = _normalize_verdict(str(parsed_2.get("verdict", "")))
                        summary_2 = str(parsed_2.get("verdict_summary", "")).strip() or summary
                        checks_2_raw = parsed_2.get("requirement_checks") or []
                        checks_2: List[AdvisorRequirementCheck] = []
                        if isinstance(checks_2_raw, list):
                            for item in checks_2_raw[:10]:
                                if not isinstance(item, dict):
                                    continue
                                checks_2.append(
                                    AdvisorRequirementCheck(
                                        requirement=str(item.get("requirement", "")).strip()
                                        or "ไม่ระบุ",
                                        user_value=str(item.get("user_value", "")).strip()
                                        or "ไม่ระบุ",
                                        status=_normalize_status(str(item.get("status", ""))),
                                        explanation=str(item.get("explanation", "")).strip(),
                                    )
                                )
                        actions_2_raw = parsed_2.get("recommended_actions") or []
                        actions_2: List[str] = []
                        if isinstance(actions_2_raw, list):
                            for a in actions_2_raw[:6]:
                                t = str(a).strip()
                                if t:
                                    actions_2.append(t)
                        response.verdict = verdict_2
                        response.verdict_summary = summary_2
                        response.requirement_checks = checks_2 or response.requirement_checks
                        response.recommended_actions = actions_2 or response.recommended_actions
                        response.sources = response_sources_2
                        # Re-score
                        rescore = _issup_score(question, context_block_2, raw_answer_2)
                        if rescore is not None:
                            trace.issup_score = rescore
                            trace.issup_passed = rescore >= 3
            except Exception as exc:
                logger.warning("Self-RAG retry failed: %s", exc)

    trace.elapsed_seconds = round(time.monotonic() - t_start, 3)
    response.reasoning_trace = trace
    return response
