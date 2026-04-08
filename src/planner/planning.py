from __future__ import annotations

import os
import re
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

NO_ANSWER_SENTINEL = "ไม่พบข้อมูลในเอกสารที่มีอยู่"
GENERAL_ONLY_NOTE = "เป็นคำแนะนำทั่วไป (ไม่พบรายละเอียดเฉพาะในเอกสารที่มีอยู่)"

FEATURE_LABELS_TH: Dict[str, str] = {
    "Sex": "เพศ",
    "Occupation": "อาชีพ",
    "Salary": "รายได้ต่อเดือน",
    "Marriage_Status": "สถานภาพสมรส",
    "credit_score": "คะแนนเครดิต",
    "credit_grade": "เกรดเครดิต",
    "outstanding": "ยอดหนี้คงค้าง",
    "overdue": "ยอดค้างชำระ",
    "Coapplicant": "ผู้กู้ร่วม",
    "loan_amount": "วงเงินกู้",
    "loan_term": "ระยะเวลากู้",
    "Interest_rate": "อัตราดอกเบี้ย",
}

NON_ACTIONABLE_FEATURES = {"Sex"}
FORBIDDEN_SEX_ACTION_TOKENS = ("เปลี่ยนเพศ", "change sex", "เปลี่ยน gender")
FORBIDDEN_FRAUD_TOKENS = (
    "ปลอม",
    "ปลอมแปลง",
    "แก้เอกสาร",
    "แก้ไขเอกสาร",
    "fake",
    "forg",
    "fraud",
)
FORBIDDEN_PROMISE_TOKENS = (
    "รับประกันอนุมัติ",
    "อนุมัติแน่นอน",
    "guarantee approval",
    "guaranteed approval",
)

DRIVER_QUERY_MAP: Dict[str, List[str]] = {
    "overdue": [
        "ผ่อนไม่ไหวต้องทำอย่างไร ปรับโครงสร้างหนี้",
        "ขอขยายระยะเวลาผ่อนได้ไหม",
        "มีมาตรการช่วยเหลือลูกหนี้อะไรบ้าง",
    ],
    "outstanding": [
        "ผ่อนไม่ไหวต้องทำอย่างไร ปรับโครงสร้างหนี้",
        "ขอขยายระยะเวลาผ่อนได้ไหม",
        "มีมาตรการช่วยเหลือลูกหนี้อะไรบ้าง",
    ],
    "loan_amount": [
        "รายได้ขั้นต่ำเท่าไหร่ถึงจะกู้บ้านได้",
        "อัตราดอกเบี้ยสินเชื่อบ้านเท่าไหร่",
        "มี fixed rate หรือ floating rate บ้าง",
    ],
    "loan_term": [
        "รายได้ขั้นต่ำเท่าไหร่ถึงจะกู้บ้านได้",
        "อัตราดอกเบี้ยสินเชื่อบ้านเท่าไหร่",
        "มี fixed rate หรือ floating rate บ้าง",
    ],
    "Interest_rate": [
        "รายได้ขั้นต่ำเท่าไหร่ถึงจะกู้บ้านได้",
        "อัตราดอกเบี้ยสินเชื่อบ้านเท่าไหร่",
        "มี fixed rate หรือ floating rate บ้าง",
    ],
    "Occupation": [
        "ต้องมีคุณสมบัติอย่างไรถึงจะกู้บ้านได้",
        "เอกสารที่ต้องใช้สมัครสินเชื่อบ้านมีอะไรบ้าง",
    ],
    "Salary": [
        "ต้องมีคุณสมบัติอย่างไรถึงจะกู้บ้านได้",
        "เอกสารที่ต้องใช้สมัครสินเชื่อบ้านมีอะไรบ้าง",
    ],
    "credit_score": ["เครดิตบูโรสำคัญอย่างไร"],
    "credit_grade": ["เครดิตบูโรสำคัญอย่างไร"],
}

APPROVED_CHECKLIST_QUERIES: List[Tuple[str, str]] = [
    ("เอกสารสมัคร", "เอกสารที่ต้องใช้สมัครสินเชื่อบ้านมีอะไรบ้าง"),
    ("คุณสมบัติเบื้องต้น", "ต้องมีคุณสมบัติอย่างไรถึงจะกู้บ้านได้"),
    ("รายได้ขั้นต่ำ", "รายได้ขั้นต่ำเท่าไหร่ถึงจะกู้บ้านได้"),
]

# Fallback content used when RAG returns NO_ANSWER (e.g. Ollama OOM, similarity miss).
# Content is standard practice across Thai retail banks — safe to display without citation.
APPROVED_CHECKLIST_FALLBACKS: Dict[str, str] = {
    "เอกสารสมัคร": (
        "บัตรประชาชน + ทะเบียนบ้าน, สลิปเงินเดือนย้อนหลัง 3-6 เดือน, "
        "Statement บัญชีเงินเดือนย้อนหลัง 6 เดือน, หนังสือรับรองการทำงาน, "
        "เอกสารหลักประกัน (โฉนด/สัญญาจะซื้อจะขาย) และแบบฟอร์มคำขอสินเชื่อของธนาคาร"
    ),
    "คุณสมบัติเบื้องต้น": (
        "อายุ 20-65 ปี, สัญชาติไทย, มีรายได้ประจำหรือธุรกิจที่มั่นคง, "
        "อายุงานปัจจุบันอย่างน้อย 6 เดือน-1 ปี, ไม่มีประวัติค้างชำระรุนแรงในเครดิตบูโร"
    ),
    "รายได้ขั้นต่ำ": (
        "ทั่วไปธนาคารกำหนดรายได้ขั้นต่ำ 15,000-30,000 บาท/เดือนสำหรับสินเชื่อบ้าน "
        "และพิจารณาให้ภาระผ่อนรวมไม่เกิน 40-50% ของรายได้ (DTI) "
        "วงเงินอนุมัติมักอยู่ที่ 30-50 เท่าของรายได้ต่อเดือน"
    ),
}

PRESENTER_GROUP_ORDER = ["debt_distress", "loan_structure", "rate_options", "credit_behavior", "docs_income", "other"]


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "y", "on"}


def _env_int(name: str, default: int, min_value: int, max_value: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        value = int(raw.strip())
    except (TypeError, ValueError):
        return default
    return max(min_value, min(max_value, value))


PLANNER_ENABLE_LLM_SYNTHESIS = _env_bool("PLANNER_ENABLE_LLM_SYNTHESIS", False)
PLANNER_MAX_ACTION_DRIVERS = _env_int("PLANNER_MAX_ACTION_DRIVERS", 3, 1, 10)
PLANNER_MAX_RAG_QUERIES_PER_DRIVER = _env_int("PLANNER_MAX_RAG_QUERIES_PER_DRIVER", 1, 0, 3)
PLANNER_APPROVED_MAX_RAG_QUERIES = _env_int("PLANNER_APPROVED_MAX_RAG_QUERIES", 2, 0, 3)


def _to_float(value: Any, default: Optional[float] = None) -> Optional[float]:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _to_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value != 0
    if isinstance(value, str):
        lowered = value.strip().lower()
        return lowered in {"1", "true", "yes", "y", "ใช่", "มี"}
    return bool(value)


def _extract_prob(probabilities: Dict[Any, Any], target: int) -> Optional[float]:
    if not isinstance(probabilities, dict):
        return None

    target_str = str(int(target))
    for key, value in probabilities.items():
        key_str = str(key).strip()
        if key_str == target_str:
            return _to_float(value)
        key_num = _to_float(key)
        if key_num is not None and int(key_num) == target:
            return _to_float(value)
    return None


def _trim_text(text: str, max_len: int = 180) -> str:
    cleaned = " ".join((text or "").split())
    if len(cleaned) <= max_len:
        return cleaned
    return cleaned[: max_len - 3].rstrip() + "..."


def _extract_top_source(result: dict) -> Optional[dict]:
    sources = result.get("sources")
    if not isinstance(sources, list) or not sources:
        return None

    top = sources[0] if isinstance(sources[0], dict) else {}
    metadata = top.get("metadata", {}) if isinstance(top, dict) else {}
    title = top.get("title") or metadata.get("title") or metadata.get("file_name")
    category = top.get("category") or metadata.get("category") or "unknown"
    score = _to_float(top.get("score"), default=0.0)

    if not title:
        return None

    return {
        "source_title": str(title),
        "category": str(category),
        "score": float(score if score is not None else 0.0),
    }


def _rag_fetch(
    rag_lookup: Optional[Callable[[str], dict]],
    query: str,
) -> Tuple[str, List[dict]]:
    if rag_lookup is None or not callable(rag_lookup):
        return "", []

    try:
        result = rag_lookup(query) or {}
    except Exception:
        return "", []

    answer = str(result.get("answer", "")).strip()
    if not answer or answer == NO_ANSWER_SENTINEL:
        return NO_ANSWER_SENTINEL, []

    top_source = _extract_top_source(result)
    if not top_source:
        return answer, []

    return answer, [{"query": query, **top_source}]


def _contains_forbidden(text: str, tokens: Iterable[str]) -> bool:
    lowered = text.lower()
    return any(token.lower() in lowered for token in tokens)


def _assert_actions_safe(actions: List[dict]) -> None:
    for action in actions:
        blob = " ".join(
            [
                str(action.get("title_th", "")),
                str(action.get("why_th", "")),
                str(action.get("how_th", "")),
            ]
        )
        if _contains_forbidden(blob, FORBIDDEN_SEX_ACTION_TOKENS):
            raise AssertionError("Unsafe action detected: suggestions about changing Sex are not allowed.")
        if _contains_forbidden(blob, FORBIDDEN_FRAUD_TOKENS):
            raise AssertionError("Unsafe action detected: fraud/manipulation guidance is not allowed.")
        if _contains_forbidden(blob, FORBIDDEN_PROMISE_TOKENS):
            raise AssertionError("Unsafe action detected: approval guarantee language is not allowed.")


def _driver_action_template(feature: str) -> Tuple[str, str]:
    if feature in {"overdue", "outstanding"}:
        return (
            "จัดการหนี้ค้างและวางแผนปรับโครงสร้างภาระหนี้",
            "ชำระยอดค้างให้เป็นปัจจุบันก่อน และหากยังตึงตัวให้ติดต่อสถาบันการเงินเพื่อพิจารณาทางเลือกปรับโครงสร้างหรือขยายงวดอย่างเป็นทางการ.",
        )
    if feature in {"loan_amount", "loan_term"}:
        return (
            "ปรับโครงสร้างวงเงินและระยะเวลากู้ให้เหมาะกับรายได้",
            "พิจารณาลดวงเงินที่ขอหรือเพิ่มเงินดาวน์ และเลือกระยะเวลากู้ที่ทำให้ค่างวดสมดุลกับกระแสเงินสดรายเดือน.",
        )
    if feature == "Interest_rate":
        return (
            "เปรียบเทียบทางเลือกดอกเบี้ยอย่างโปร่งใส",
            "สอบถามโครงสร้างอัตราดอกเบี้ยหลายทางเลือก (เช่น fixed/floating) และเลือกแบบที่เหมาะกับความสามารถผ่อนจริงโดยเปิดเผยข้อมูลตามจริง.",
        )
    if feature in {"Occupation", "Salary"}:
        return (
            "เตรียมเอกสารรายได้และคุณสมบัติผู้กู้ให้ชัดเจน",
            "รวบรวมหลักฐานรายได้และเอกสารประกอบให้ครบถ้วน เพื่อให้การประเมินความสามารถชำระหนี้มีความชัดเจนขึ้น.",
        )
    if feature in {"credit_score", "credit_grade"}:
        return (
            "ฟื้นฟูวินัยเครดิตอย่างต่อเนื่อง",
            "ชำระหนี้ตรงเวลา ลดการก่อหนี้ใหม่ที่ไม่จำเป็น และติดตามข้อมูลเครดิตสม่ำเสมอเพื่อปรับพฤติกรรมทางการเงิน.",
        )
    return (
        "ทบทวนความพร้อมทางการเงินก่อนยื่นใหม่",
        "จัดลำดับการลดภาระหนี้ เตรียมเอกสารรายได้ และประเมินความสามารถผ่อนก่อนส่งคำขอรอบถัดไป.",
    )


def parse_model_output(model_output: dict) -> dict:
    prediction_raw = model_output.get("prediction", 0)
    prediction_num = _to_float(prediction_raw, default=0.0)
    approved = int(prediction_num or 0) == 1

    probabilities = model_output.get("probabilities", {}) or {}
    p_approve = _extract_prob(probabilities, 1)
    p_reject = _extract_prob(probabilities, 0)

    if p_approve is None and p_reject is None:
        p_approve = 1.0 if approved else 0.0
        p_reject = 1.0 - p_approve
    elif p_approve is None:
        p_reject = p_reject if p_reject is not None else 0.0
        p_approve = max(0.0, min(1.0, 1.0 - p_reject))
    elif p_reject is None:
        p_approve = p_approve if p_approve is not None else 0.0
        p_reject = max(0.0, min(1.0, 1.0 - p_approve))

    return {
        "approved": approved,
        "p_approve": float(p_approve),
        "p_reject": float(p_reject),
    }


def normalize_shap(shap_json: dict) -> dict[str, float]:
    if not isinstance(shap_json, dict):
        raise ValueError("SHAP JSON must be a dict in Style 1 format with a 'values' field.")

    values = shap_json.get("values")
    if not isinstance(values, dict) or not values:
        raise ValueError(
            "SHAP Style 1 expected: missing/invalid 'values'. "
            "Example: {'base_value': ..., 'values': {'Salary': 0.18, ...}}"
        )

    normalized: Dict[str, float] = {}
    for feature, shap_value in values.items():
        parsed = _to_float(shap_value)
        if parsed is None:
            raise ValueError(f"SHAP value for feature '{feature}' is not numeric: {shap_value!r}")
        normalized[str(feature)] = float(parsed)
    return normalized


def summarize_shap(shap_dict: dict[str, float], top_k: int = 6) -> dict:
    labels_th = {feature: FEATURE_LABELS_TH.get(feature, feature) for feature in shap_dict}

    negatives = sorted(
        ((feature, value) for feature, value in shap_dict.items() if value < 0),
        key=lambda x: x[1],
    )
    positives = sorted(
        ((feature, value) for feature, value in shap_dict.items() if value > 0),
        key=lambda x: x[1],
        reverse=True,
    )

    top_negative = [
        {"feature": feature, "shap": float(value), "label_th": labels_th[feature]}
        for feature, value in negatives[:top_k]
    ]
    top_positive = [
        {"feature": feature, "shap": float(value), "label_th": labels_th[feature]}
        for feature, value in positives[:top_k]
    ]

    return {
        "top_negative": top_negative,
        "top_positive": top_positive,
        "non_actionable": ["Sex"],
        "labels_th": labels_th,
    }


def build_actions(
    user_input: dict,
    shap_summary: dict,
    rag_lookup: Optional[Callable[[str], dict]] = None,
) -> list[dict]:
    del user_input  # Reserved for future feature-level customization.

    top_negative = (shap_summary.get("top_negative", []) or [])[:PLANNER_MAX_ACTION_DRIVERS]
    actions: List[dict] = []

    for item in top_negative:
        feature = str(item.get("feature", ""))
        if not feature or feature in NON_ACTIONABLE_FEATURES:
            continue

        shap_value = _to_float(item.get("shap"), default=0.0) or 0.0
        label_th = item.get("label_th") or FEATURE_LABELS_TH.get(feature, feature)

        queries = DRIVER_QUERY_MAP.get(feature, [])[:PLANNER_MAX_RAG_QUERIES_PER_DRIVER]
        selected_answer = ""
        selected_evidence: List[dict] = []

        for query in queries:
            answer, evidence = _rag_fetch(rag_lookup, query)
            if evidence and answer != NO_ANSWER_SENTINEL:
                selected_answer = answer
                selected_evidence = evidence
                break

        title_th, how_base = _driver_action_template(feature)
        why_th = f"ปัจจัย '{label_th}' กดผลประเมิน (SHAP {shap_value:+.2f})"

        if selected_evidence:
            how_th = f"{how_base} สาระจากเอกสาร: {_trim_text(selected_answer)}"
            evidence_confidence = "documented"
        else:
            how_th = f"{how_base} {GENERAL_ONLY_NOTE}"
            evidence_confidence = "general_only"

        actions.append(
            {
                "title_th": title_th,
                "why_th": why_th,
                "how_th": how_th,
                "evidence": selected_evidence,
                "evidence_confidence": evidence_confidence,
            }
        )

    if not actions:
        actions.append(
            {
                "title_th": "วางวินัยการเงินพื้นฐานก่อนยื่นใหม่",
                "why_th": "ไม่พบตัวขับเชิงลบที่มีหลักฐานเอกสารเพียงพอจากข้อมูลที่ให้มา",
                "how_th": "ชำระหนี้ตรงเวลา ลดภาระหนี้คงค้าง และหลีกเลี่ยงการก่อหนี้ใหม่ก่อนยื่นคำขออีกครั้ง. เป็นคำแนะนำทั่วไป (ไม่พบรายละเอียดเฉพาะในเอกสารที่มีอยู่)",
                "evidence": [],
                "evidence_confidence": "general_only",
            }
        )

    _assert_actions_safe(actions)
    return actions


def build_clarifying_questions(user_input: dict) -> list[str]:
    questions: List[str] = []

    if not user_input.get("product_type"):
        questions.append("ต้องการยื่นสินเชื่อประเภทใด: สินเชื่อบ้านใหม่ / รีไฟแนนซ์ / บ้านแลกเงิน?")

    if user_input.get("property_price") in (None, "", 0) or user_input.get("ltv") in (None, ""):
        questions.append("ราคาทรัพย์และเงินดาวน์โดยประมาณ (หรือ LTV ที่คาดหวัง) คือเท่าไร?")

    if _to_bool(user_input.get("Coapplicant")) and not user_input.get("coapplicant_income"):
        questions.append("ผู้กู้ร่วมมีรายได้ต่อเดือนและหลักฐานรายได้ประเภทใดบ้าง?")
    elif "coapplicant_income" not in user_input:
        questions.append("หากมีผู้กู้ร่วม กรุณาระบุรายได้และหลักฐานรายได้ของผู้กู้ร่วมเพิ่มเติม")

    return questions[:3]


def generate_plan(
    user_input: dict,
    model_output: dict,
    shap_json: dict,
    rag_lookup: Optional[Callable[[str], dict]] = None,
) -> dict:
    decision = parse_model_output(model_output)
    shap_summary = summarize_shap(normalize_shap(shap_json), top_k=6)
    actions = build_actions(user_input, shap_summary, rag_lookup=rag_lookup)
    clarifying_questions = build_clarifying_questions(user_input)

    plan = {
        "decision": decision,
        "risk_drivers": shap_summary,
        "actions": actions,
        "clarifying_questions": clarifying_questions,
        "disclaimer_th": (
            "ผลลัพธ์นี้จัดทำโดยแบบจำลองทางสถิติเพื่อวัตถุประสงค์ทางการวิจัย "
            "มิใช่การพิจารณาสินเชื่อจริงจากสถาบันการเงิน"
        ),
    }
    _assert_actions_safe(plan["actions"])
    return plan


def _normalize_whitespace(text: str) -> str:
    normalized = (text or "").replace("\r", "\n")
    normalized = normalized.replace("\t", " ")
    normalized = re.sub(r"(?m)^\s*n-\s*", "", normalized)
    normalized = normalized.replace(" n-", " ")
    normalized = re.sub(r"[ ]{2,}", " ", normalized)
    normalized = re.sub(r"\n[ ]+", "\n", normalized)
    normalized = re.sub(r"\n{3,}", "\n\n", normalized)
    lines = [ln.strip() for ln in normalized.split("\n")]
    return "\n".join(lines).strip()


def _infer_action_group(action: dict) -> str:
    text = " ".join(
        [
            str(action.get("title_th", "")),
            str(action.get("why_th", "")),
            str(action.get("how_th", "")),
            str(action.get("evidence", "")),
        ]
    ).lower()

    if any(k in text for k in ["ค้างชำระ", "หนี้คงค้าง", "โครงสร้างภาระหนี้", "ปรับโครงสร้างหนี้"]):
        return "debt_distress"
    if any(k in text for k in ["วงเงิน", "ระยะเวลากู้", "ค่างวด"]):
        return "loan_structure"
    if any(k in text for k in ["เครดิต", "credit"]):
        return "credit_behavior"
    if any(k in text for k in ["ดอกเบี้ย", "fixed", "floating", "interest"]):
        return "rate_options"
    if any(k in text for k in ["เอกสาร", "รายได้", "อาชีพ", "ผู้กู้ร่วม"]):
        return "docs_income"
    return "other"


def _parse_priority(action: dict) -> int:
    explicit = action.get("priority")
    if isinstance(explicit, int):
        return max(1, min(3, explicit))

    why_text = str(action.get("why_th", ""))
    match = re.search(r"SHAP\s*([+-]?\d+(?:\.\d+)?)", why_text)
    if not match:
        return 3

    shap_value = _to_float(match.group(1), default=0.0) or 0.0
    if shap_value <= -0.20:
        return 1
    if shap_value < 0:
        return 2
    return 3


def _choose_best_evidence(evidences: List[dict]) -> List[dict]:
    valid = [e for e in evidences if isinstance(e, dict) and str(e.get("source_title", "")).strip()]
    if not valid:
        return []
    best = max(valid, key=lambda x: float(_to_float(x.get("score"), default=0.0) or 0.0))
    return [best]


def _is_interest_evidence(evidence: dict) -> bool:
    text = " ".join(
        [
            str(evidence.get("source_title", "")),
            str(evidence.get("category", "")),
            str(evidence.get("query", "")),
        ]
    ).lower()
    return any(token in text for token in ["interest", "อัตราดอกเบี้ย", "ดอกเบี้ย", "fixed", "floating"])


def _dedupe_and_merge_actions(actions: List[dict]) -> List[dict]:
    merged_buckets: Dict[str, List[dict]] = {}

    for action in actions:
        title = str(action.get("title_th", "")).strip().lower()
        group = _infer_action_group(action)
        key = group if group != "other" else f"title::{title}"
        merged_buckets.setdefault(key, []).append(action)

    merged_actions: List[dict] = []
    for key, bucket in merged_buckets.items():
        first = bucket[0]
        group = _infer_action_group(first)

        all_whys = [str(a.get("why_th", "")).strip() for a in bucket if str(a.get("why_th", "")).strip()]
        all_hows = [str(a.get("how_th", "")).strip() for a in bucket if str(a.get("how_th", "")).strip()]
        all_evidence = []
        for action in bucket:
            all_evidence.extend(action.get("evidence", []) or [])

        title_th = str(first.get("title_th", "")).strip() or "แผนปรับความพร้อม"
        why_th = " / ".join(dict.fromkeys(all_whys)) if all_whys else "ไม่มีรายละเอียดเหตุผลเพิ่มเติม"
        how_th = all_hows[0] if all_hows else "โปรดทบทวนแผนทางการเงินก่อนยื่นคำขอ"
        evidence = _choose_best_evidence(all_evidence)

        confidence = "documented" if evidence else "general_only"
        priority = min(_parse_priority(a) for a in bucket)

        if group == "rate_options" and evidence and not _is_interest_evidence(evidence[0]):
            evidence = []
            confidence = "general_only"

        merged_actions.append(
            {
                "title_th": title_th,
                "why_th": _trim_text(why_th, 260),
                "how_th": _trim_text(how_th, 260),
                "evidence": evidence,
                "evidence_confidence": confidence,
                "priority": priority,
                "group": group,
            }
        )

    merged_actions.sort(key=lambda a: (_parse_priority(a), PRESENTER_GROUP_ORDER.index(a.get("group", "other")) if a.get("group", "other") in PRESENTER_GROUP_ORDER else 99))
    return merged_actions


def _action_brief(action: dict) -> str:
    title = str(action.get("title_th", "")).strip()
    how_th = str(action.get("how_th", "")).strip()
    brief = f"{title}: {_trim_text(how_th, 110)}"
    evidence = action.get("evidence", []) or []
    if evidence:
        source_title = str(evidence[0].get("source_title", "")).strip()
        if source_title:
            brief += f" (อ้างอิง: {source_title})"
    return brief


def _pick_groups(merged_actions: List[dict], groups: Iterable[str]) -> List[dict]:
    wanted = set(groups)
    return [a for a in merged_actions if a.get("group") in wanted]


def render_plan_th(plan_json: dict, style: str = "paragraph") -> str:
    if style not in {"paragraph", "123", "ABC"}:
        raise ValueError("style must be one of: paragraph | 123 | ABC")

    plan = plan_json.get("plan") if isinstance(plan_json, dict) and "plan" in plan_json else plan_json
    plan = plan or {}

    decision = plan.get("decision", {}) or {}
    approved = bool(decision.get("approved"))
    p_approve = _to_float(decision.get("p_approve"), default=0.0) or 0.0
    p_reject = _to_float(decision.get("p_reject"), default=0.0) or 0.0

    top_negative = (plan.get("risk_drivers", {}) or {}).get("top_negative", []) or []
    top3 = top_negative[:3]
    top3_text = ", ".join(
        f"{item.get('label_th', item.get('feature'))} ({(_to_float(item.get('shap'), 0.0) or 0.0):+.2f})"
        for item in top3
    )

    raw_actions = plan.get("actions", []) or []
    merged_actions = _dedupe_and_merge_actions(raw_actions)

    immediate_actions = _pick_groups(merged_actions, ["debt_distress", "docs_income"])
    short_actions = _pick_groups(merged_actions, ["loan_structure", "rate_options"])
    medium_actions = _pick_groups(merged_actions, ["credit_behavior", "other"])

    lines: List[str] = []
    lines.append("สรุปผลการวิเคราะห์")
    lines.append(f"- ผลการประเมินของแบบจำลอง: {'ความน่าจะเป็นอนุมัติสูง' if approved else 'ความน่าจะเป็นปฏิเสธสูง'} (P(อนุมัติ)={p_approve:.3f} | P(ปฏิเสธ)={p_reject:.3f})")
    if top3_text:
        lines.append(f"- ตัวแปรที่มีผลลดโอกาสอนุมัติ (SHAP เชิงลบ): {top3_text}")
    lines.append("- แนวทางปรับปรุง: บรรเทาตัวแปรความเสี่ยงเร่งด่วนก่อน จากนั้นปรับโครงสร้างคำขอและพฤติกรรมการชำระหนี้ต่อเนื่อง")

    lines.append("")
    if style == "paragraph":
        para1 = (
            f"ผลการวิเคราะห์ของแบบจำลองระบุว่า {'ความน่าจะเป็นอนุมัติสูง' if approved else 'ความน่าจะเป็นปฏิเสธสูง'} "
            f"โดยตัวแปรหลักที่ส่งผลต่อการตัดสินใจคือ {top3_text or 'ภาระหนี้และความพร้อมด้านเอกสาร'} จึงควรดำเนินมาตรการปรับปรุงตามลำดับ."
        )
        para2 = (
            f"ระยะทันทีให้โฟกัส {', '.join(_action_brief(a) for a in immediate_actions) or 'การเคลียร์ภาระหนี้เร่งด่วนและเอกสารรายได้'}. "
            f"ช่วง 1-3 เดือนให้เดินแผน {', '.join(_action_brief(a) for a in short_actions) or 'การปรับวงเงิน/ระยะเวลากู้และทางเลือกดอกเบี้ย'}. "
            f"ช่วง 3-6 เดือนให้ต่อยอด {', '.join(_action_brief(a) for a in medium_actions) or 'การฟื้นวินัยเครดิต'}."
        )
        lines.append(para1)
        lines.append(para2)

    elif style == "123":
        lines.append("1) มาตรการเร่งด่วน: " + ("; ".join(_action_brief(a) for a in immediate_actions) or "บรรเทาภาระหนี้เร่งด่วนและจัดเตรียมหลักฐานรายได้ให้ครบถ้วน"))
        lines.append("2) มาตรการระยะสั้น (1-3 เดือน): " + ("; ".join(_action_brief(a) for a in short_actions) or "ปรับโครงสร้างวงเงิน/ระยะเวลากู้ และเปรียบเทียบทางเลือกอัตราดอกเบี้ย"))
        lines.append("3) มาตรการระยะกลาง (3-6 เดือน): " + ("; ".join(_action_brief(a) for a in medium_actions) or "ฟื้นฟูวินัยการชำระหนี้และติดตามผลก่อนยื่นขอใหม่"))

    else:  # ABC
        lines.append("มาตรการ A (บรรเทาความเสี่ยงเร่งด่วน): " + ("; ".join(_action_brief(a) for a in immediate_actions) or "จัดการภาระหนี้เร่งด่วนและหลักฐานรายได้"))
        lines.append("มาตรการ B (ปรับโครงสร้างคำขอสินเชื่อ): " + ("; ".join(_action_brief(a) for a in short_actions) or "ปรับวงเงิน/ระยะเวลากู้และโครงสร้างอัตราดอกเบี้ย"))
        lines.append("มาตรการ C (ฟื้นฟูพฤติกรรมเครดิตระยะกลาง): " + ("; ".join(_action_brief(a) for a in medium_actions) or "รักษาวินัยการชำระหนี้และประเมินซ้ำ"))

    clarifying = plan.get("clarifying_questions", []) or []
    if clarifying:
        lines.append("")
        lines.append("ข้อมูลที่ควรยืนยันเพิ่มเติม")
        for question in clarifying:
            lines.append(f"- {question}")

    disclaimer = str(plan.get("disclaimer_th", "")).strip()
    if disclaimer:
        lines.append("")
        lines.append(disclaimer)

    return _normalize_whitespace("\n".join(lines))


def plan_to_thai_text(plan: dict) -> str:
    return render_plan_th(plan, style="123")


def _build_approved_checklist(
    decision: dict,
    rag_lookup: Optional[Callable[[str], dict]],
) -> str:
    p_approve = _to_float(decision.get("p_approve"), default=0.0) or 0.0
    p_reject = _to_float(decision.get("p_reject"), default=0.0) or 0.0

    lines: List[str] = []
    lines.append("ผลการวิเคราะห์ของแบบจำลอง: ความน่าจะเป็นอนุมัติสูง")
    lines.append(f"P(อนุมัติ)={p_approve:.3f} | P(ปฏิเสธ)={p_reject:.3f}")
    lines.append("รายการเอกสาร/ข้อมูลที่จำเป็นสำหรับการยื่นขอสินเชื่อ")

    checklist_queries = APPROVED_CHECKLIST_QUERIES[:PLANNER_APPROVED_MAX_RAG_QUERIES]

    for idx, (title, query) in enumerate(checklist_queries, start=1):
        answer, evidence = _rag_fetch(rag_lookup, query)
        if answer and answer != NO_ANSWER_SENTINEL:
            line = f"{idx}) {title}: {_trim_text(answer, 200)}"
            if evidence:
                line += f" (แหล่งข้อมูล: {evidence[0].get('source_title', 'N/A')})"
        else:
            # RAG missed (Ollama OOM, low similarity, etc.) — render general Thai
            # banking knowledge instead of the NO_ANSWER sentinel so the user sees
            # actionable content. Mark it clearly as general guidance.
            fallback = APPROVED_CHECKLIST_FALLBACKS.get(title, "")
            if fallback:
                line = f"{idx}) {title}: {fallback} [คำแนะนำทั่วไป]"
            else:
                line = f"{idx}) {title}: {GENERAL_ONLY_NOTE}"
        lines.append(line)

    lines.append("หมายเหตุ: ผลลัพธ์นี้จัดทำโดยแบบจำลองทางสถิติเพื่อวัตถุประสงค์ทางการวิจัย มิใช่การพิจารณาสินเชื่อจริงจากสถาบันการเงิน")
    return _normalize_whitespace("\n".join(lines))


def _llm_synthesize_plan(plan: dict, user_input: dict) -> str:
    """
    Use the configured LLM (Gemini/Ollama/OpenAI via Settings.llm) to synthesize
    a formal, case-specific Thai improvement plan from the structured plan data.
    Returns empty string on failure so caller can fall back to rule-based rendering.
    """
    try:
        from llama_index.core.settings import Settings  # noqa: PLC0415
        llm = getattr(Settings, "llm", None)
        if llm is None:
            return ""
    except Exception:
        return ""

    decision = plan.get("decision", {})
    p_approve = float(decision.get("p_approve", 0.0))
    p_reject = float(decision.get("p_reject", 0.0))

    risk_drivers = plan.get("risk_drivers", {})
    top_neg = risk_drivers.get("top_negative", []) or []
    top_pos = risk_drivers.get("top_positive", []) or []
    actions = plan.get("actions", []) or []
    clarifying = plan.get("clarifying_questions", []) or []

    # --- user profile summary ---
    salary = float(user_input.get("Salary") or 0)
    occupation = str(user_input.get("Occupation") or "ไม่ระบุ")
    credit_score = user_input.get("credit_score", "ไม่ระบุ")
    credit_grade = user_input.get("credit_grade", "ไม่ระบุ")
    outstanding = float(user_input.get("outstanding") or 0)
    overdue = float(user_input.get("overdue") or 0)
    loan_amount = float(user_input.get("loan_amount") or 0)
    loan_term = user_input.get("loan_term", "ไม่ระบุ")
    coapplicant = "มี" if user_input.get("Coapplicant") else "ไม่มี"

    # --- build actions block for prompt ---
    actions_block_lines: List[str] = []
    for i, a in enumerate(actions, 1):
        title = a.get("title_th", "")
        how = a.get("how_th", "")
        evidence_list = a.get("evidence") or []
        ev_note = ""
        if evidence_list and isinstance(evidence_list[0], dict):
            ans = str(evidence_list[0].get("answer", "")).strip()
            src = str(evidence_list[0].get("source_title", "")).strip()
            if ans and ans != NO_ANSWER_SENTINEL:
                ev_note = f"\n   หลักฐานเอกสาร: {_trim_text(ans, 150)}"
                if src:
                    ev_note += f" (ที่มา: {src})"
        actions_block_lines.append(f"{i}. {title}\n   วิธีดำเนินการ: {how}{ev_note}")

    actions_block = "\n".join(actions_block_lines)

    neg_drivers = ", ".join(
        f"{d.get('label_th', '?')} (SHAP {d.get('shap', 0):+.2f})" for d in top_neg[:4]
    )
    pos_drivers = (
        ", ".join(f"{d.get('label_th', '?')} (SHAP {d.get('shap', 0):+.2f})" for d in top_pos[:2])
        or "ไม่มี"
    )
    clarifying_block = (
        "\n".join(f"- {q}" for q in clarifying) if clarifying else "ไม่มี"
    )

    prompt = f"""คุณเป็นระบบวิเคราะห์สินเชื่อด้วยปัญญาประดิษฐ์ (AI Credit Scoring System) ที่พัฒนาขึ้นเพื่องานวิจัยวิทยานิพนธ์
โปรดเขียนรายงานผลการวิเคราะห์และข้อเสนอแนะการปรับปรุงโปรไฟล์สินเชื่อ เป็นภาษาไทยทางการ ชัดเจน เหมาะสำหรับบริบทงานวิจัยวิทยานิพนธ์

=== ข้อมูลผู้ขอสินเชื่อ ===
- รายได้/เดือน: {salary:,.0f} บาท | อาชีพ: {occupation}
- คะแนนเครดิต: {credit_score} | เกรด: {credit_grade}
- หนี้คงค้าง: {outstanding:,.0f} บาท | ค้างชำระ: {overdue} วัน
- วงเงินขอกู้: {loan_amount:,.0f} บาท | ระยะเวลา: {loan_term} ปี | ผู้กู้ร่วม: {coapplicant}

=== ผลการวิเคราะห์ของแบบจำลอง ===
- ความน่าจะเป็นอนุมัติ: {p_approve:.1%} | ความน่าจะเป็นปฏิเสธ: {p_reject:.1%}
- ตัวแปรที่มีผลลดโอกาสอนุมัติ (SHAP เชิงลบ): {neg_drivers}
- ตัวแปรที่มีผลเพิ่มโอกาสอนุมัติ (SHAP เชิงบวก): {pos_drivers}

=== มาตรการปรับปรุงโปรไฟล์ที่แบบจำลองวิเคราะห์ ===
{actions_block}

=== ข้อมูลที่ยังไม่ครบถ้วนสำหรับการวิเคราะห์ ===
{clarifying_block}

=== คำสั่ง ===
โปรดเขียนรายงานให้:
1. ขึ้นต้นด้วยย่อหน้าสรุปผลการวิเคราะห์ (2-3 ประโยค) ระบุความน่าจะเป็นในการอนุมัติและตัวแปรหลักที่ส่งผลต่อการตัดสินใจของแบบจำลอง
2. ระบุมาตรการปรับปรุงแต่ละข้อในรูปแบบ "มาตรการที่ N: [ชื่อมาตรการ]" พร้อมอธิบายเหตุผลเชิงวิเคราะห์และแนวทางปฏิบัติที่เป็นรูปธรรม หากมีหลักฐานจากเอกสารอ้างอิงให้ระบุแหล่งที่มาด้วย
3. ปิดด้วยย่อหน้าสรุปภาพรวมเชิงวิเคราะห์ที่ระบุลำดับความสำคัญของมาตรการ
4. ท้ายสุดใส่ข้อความ: "หมายเหตุ: ผลลัพธ์นี้จัดทำโดยแบบจำลองทางสถิติเพื่อวัตถุประสงค์ทางการวิจัย มิใช่การพิจารณาสินเชื่อจริงจากสถาบันการเงิน"

ข้อห้ามเด็ดขาด: ห้ามรับประกันการอนุมัติ ห้ามแนะนำปลอมแปลงเอกสาร ห้ามให้ข้อมูลเท็จ ห้ามสัญญาผล"""

    try:
        response = llm.complete(prompt)
        text = _normalize_whitespace(str(response).strip())
        if len(text) < 150:
            return ""
        # Safety check — reject if forbidden tokens appear
        if _contains_forbidden(text, FORBIDDEN_FRAUD_TOKENS + FORBIDDEN_PROMISE_TOKENS):
            return ""
        return text
    except Exception:
        return ""


def _llm_synthesize_approved(decision: dict, checklist_text: str, user_input: dict) -> str:
    """Use LLM to write a formal Thai approval guidance report."""
    try:
        from llama_index.core.settings import Settings  # noqa: PLC0415
        llm = getattr(Settings, "llm", None)
        if llm is None:
            return ""
    except Exception:
        return ""

    p_approve = float(decision.get("p_approve", 0.0))
    salary = float(user_input.get("Salary") or 0)
    loan_amount = float(user_input.get("loan_amount") or 0)

    prompt = f"""คุณเป็นระบบวิเคราะห์สินเชื่อด้วยปัญญาประดิษฐ์ (AI Credit Scoring System) ที่พัฒนาขึ้นเพื่องานวิจัยวิทยานิพนธ์
ผู้ขอสินเชื่อรายนี้มีความน่าจะเป็นในการอนุมัติสูงตามผลการวิเคราะห์ของแบบจำลอง (ความน่าจะเป็น {p_approve:.1%})
รายได้: {salary:,.0f} บาท/เดือน | วงเงินขอกู้: {loan_amount:,.0f} บาท

ผลการค้นหาข้อมูลเอกสารอ้างอิงที่เกี่ยวข้อง:
{checklist_text}

โปรดเขียนรายงานผลการวิเคราะห์เป็นภาษาไทยทางการ:
1. เริ่มด้วยการสรุปผลการวิเคราะห์ของแบบจำลองพร้อมระบุข้อจำกัดของผลลัพธ์ (2 ประโยค)
2. สรุปรายการเอกสาร/ข้อมูลที่จำเป็นสำหรับการยื่นขอสินเชื่อในรูปแบบที่ชัดเจน
3. ปิดด้วยข้อสังเกตเชิงวิเคราะห์เกี่ยวกับโปรไฟล์สินเชื่อและข้อเสนอแนะเพิ่มเติม
4. ใส่ข้อความ: "หมายเหตุ: ผลลัพธ์นี้จัดทำโดยแบบจำลองทางสถิติเพื่อวัตถุประสงค์ทางการวิจัย มิใช่การพิจารณาสินเชื่อจริงจากสถาบันการเงิน"

ห้ามรับประกันการอนุมัติเด็ดขาด"""

    try:
        response = llm.complete(prompt)
        text = _normalize_whitespace(str(response).strip())
        if len(text) < 100:
            return ""
        if _contains_forbidden(text, FORBIDDEN_PROMISE_TOKENS):
            return ""
        return text
    except Exception:
        return ""


def _issup_check_plan(result_th: str, actions: List[dict]) -> Optional[int]:
    """
    [IsSup] reflection for planning: score (1-5) whether result_th is grounded
    in the evidence collected by build_actions().

    Returns score 1-5; < 2 means not grounded → caller should fall back to
    rule-based render_plan_th().
    Returns None when the LLM is unavailable — callers must treat None
    conservatively (i.e. fall back to rule-based render) rather than trusting
    the unverified output.
    """
    try:
        from llama_index.core.settings import Settings  # noqa: PLC0415
        llm = getattr(Settings, "llm", None)
        if llm is None:
            return None  # unknown — caller should fall back to rule-based render
    except Exception:
        return None

    # Collect evidence snippets from all actions
    evidence_lines: List[str] = []
    for a in actions:
        title = str(a.get("title_th", "")).strip()
        how = str(a.get("how_th", "")).strip()
        if title:
            evidence_lines.append(f"- {title}: {_trim_text(how, 120)}")
    evidence_block = "\n".join(evidence_lines) if evidence_lines else "(ไม่มีหลักฐาน)"

    prompt = f"""คุณเป็นผู้ตรวจสอบความน่าเชื่อถือของรายงานการวิเคราะห์สินเชื่อ

หลักฐานที่มีอยู่จริง (จากข้อมูล SHAP + RAG):
{evidence_block}

รายงานที่ต้องตรวจสอบ:
{_trim_text(result_th, 600)}

คำถาม: รายงานนี้มีเนื้อหาสอดคล้องกับหลักฐานที่มีอยู่จริงมากน้อยเพียงใด?
ตอบด้วยตัวเลข 1-5 เพียงตัวเดียว:
1 = ข้อมูลขัดแย้งหรือแต่งเติมเกินจริงอย่างชัดเจน
2 = บางส่วนไม่สอดคล้องกับหลักฐาน
3 = ส่วนใหญ่สอดคล้อง มีการขยายความบ้าง
4 = สอดคล้องดี
5 = สอดคล้องอย่างสมบูรณ์

ตอบเฉพาะตัวเลข:""".strip()

    try:
        import re as _re
        raw = str(llm.complete(prompt)).strip()
        match = _re.search(r"[1-5]", raw)
        return int(match.group()) if match else None
    except Exception:
        return None


def generate_response(
    user_input: dict,
    model_output: dict,
    shap_json: dict,
    rag_lookup: Optional[Callable[[str], dict]] = None,
    use_issup: bool = False,
) -> dict:
    decision = parse_model_output(model_output)

    if decision["approved"]:
        checklist_text = _build_approved_checklist(decision, rag_lookup)
        llm_text = _llm_synthesize_approved(decision, checklist_text, user_input) if PLANNER_ENABLE_LLM_SYNTHESIS else ""
        result_th = llm_text or checklist_text
        return {
            "mode": "approved_guidance",
            "decision": decision,
            "result_th": result_th,
        }

    plan = generate_plan(
        user_input=user_input,
        model_output=model_output,
        shap_json=shap_json,
        rag_lookup=rag_lookup,
    )
    llm_result = _llm_synthesize_plan(plan, user_input) if PLANNER_ENABLE_LLM_SYNTHESIS else ""
    issup_score: Optional[int] = None
    issup_passed: Optional[bool] = None
    if llm_result and use_issup:
        issup_score = _issup_check_plan(llm_result, plan.get("actions", []))
        if issup_score is None:
            # LLM unavailable — cannot verify groundedness, fall back conservatively
            issup_passed = False
            llm_result = ""
        else:
            issup_passed = issup_score >= 2
            if not issup_passed:
                llm_result = ""  # fall back to rule-based render
    result_th = llm_result or render_plan_th(plan, style="123")
    response: dict = {
        "mode": "improvement_plan",
        "decision": decision,
        "result_th": result_th,
        "plan": plan,
    }
    if use_issup:
        response["issup_score"] = issup_score
        response["issup_passed"] = issup_passed
    return response
