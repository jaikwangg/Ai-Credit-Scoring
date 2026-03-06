from __future__ import annotations

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

PRESENTER_GROUP_ORDER = ["debt_distress", "loan_structure", "rate_options", "credit_behavior", "docs_income", "other"]


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
            "ชำระยอดค้างให้เป็นปัจจุบันก่อน และหากยังตึงตัวให้ติดต่อธนาคารเพื่อพิจารณาทางเลือกปรับโครงสร้างหรือขยายงวดอย่างเป็นทางการ.",
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

    top_negative = shap_summary.get("top_negative", []) or []
    actions: List[dict] = []

    for item in top_negative:
        feature = str(item.get("feature", ""))
        if not feature or feature in NON_ACTIONABLE_FEATURES:
            continue

        shap_value = _to_float(item.get("shap"), default=0.0) or 0.0
        label_th = item.get("label_th") or FEATURE_LABELS_TH.get(feature, feature)

        queries = DRIVER_QUERY_MAP.get(feature, [])
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
            "คำแนะนำนี้เป็นข้อเสนอเชิงข้อมูลจากแบบจำลองและเอกสารที่ค้นพบ "
            "ไม่สามารถรับประกันผลอนุมัติ และควรตรวจสอบเงื่อนไขล่าสุดกับธนาคาร"
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
    lines.append("สรุปสั้น")
    lines.append(f"- สถานะปัจจุบัน: {'มีแนวโน้มอนุมัติ' if approved else 'ยังมีความเสี่ยงไม่อนุมัติ'} (อนุมัติ {p_approve:.3f} | ไม่อนุมัติ {p_reject:.3f})")
    if top3_text:
        lines.append(f"- ปัจจัยกดผลลัพธ์หลัก: {top3_text}")
    lines.append("- แนวทางรวม: ลดความเสี่ยงเร่งด่วนก่อน แล้วค่อยปรับโครงสร้างคำขอและวินัยเครดิตต่อเนื่อง")

    lines.append("")
    if style == "paragraph":
        para1 = (
            f"ภาพรวมการประเมินตอนนี้ {'มีแนวโน้มอนุมัติ' if approved else 'ยังมีความเสี่ยงไม่อนุมัติ'} "
            f"โดยตัวขับหลักคือ {top3_text or 'ภาระหนี้และความพร้อมเอกสาร'} จึงควรปรับแผนแบบเป็นลำดับ."
        )
        para2 = (
            f"ระยะทันทีให้โฟกัส {', '.join(_action_brief(a) for a in immediate_actions) or 'การเคลียร์ภาระหนี้เร่งด่วนและเอกสารรายได้'}. "
            f"ช่วง 1-3 เดือนให้เดินแผน {', '.join(_action_brief(a) for a in short_actions) or 'การปรับวงเงิน/ระยะเวลากู้และทางเลือกดอกเบี้ย'}. "
            f"ช่วง 3-6 เดือนให้ต่อยอด {', '.join(_action_brief(a) for a in medium_actions) or 'การฟื้นวินัยเครดิต'}."
        )
        lines.append(para1)
        lines.append(para2)

    elif style == "123":
        lines.append("1) ทำทันที: " + ("; ".join(_action_brief(a) for a in immediate_actions) or "เคลียร์ภาระหนี้เร่งด่วนและจัดเอกสารรายได้ให้ครบ"))
        lines.append("2) ภายใน 1-3 เดือน: " + ("; ".join(_action_brief(a) for a in short_actions) or "ปรับโครงสร้างวงเงิน/ระยะเวลากู้ และเทียบทางเลือกดอกเบี้ย"))
        lines.append("3) ภายใน 3-6 เดือน: " + ("; ".join(_action_brief(a) for a in medium_actions) or "รักษาวินัยเครดิตและติดตามผลก่อนยื่นใหม่"))

    else:  # ABC
        lines.append("แผน A (ลดความเสี่ยงเร่งด่วน): " + ("; ".join(_action_brief(a) for a in immediate_actions) or "จัดการภาระหนี้เร่งด่วนและเอกสารรายได้"))
        lines.append("แผน B (ปรับโครงสร้างคำขอ): " + ("; ".join(_action_brief(a) for a in short_actions) or "ปรับวงเงิน/ระยะเวลากู้และโครงสร้างดอกเบี้ย"))
        lines.append("แผน C (ฟื้นเครดิตระยะกลาง): " + ("; ".join(_action_brief(a) for a in medium_actions) or "รักษาวินัยเครดิตและประเมินซ้ำ"))

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
    lines.append("ผลประเมินเบื้องต้น: มีแนวโน้มอนุมัติ")
    lines.append(f"ความน่าจะเป็นอนุมัติ {p_approve:.3f} | ไม่อนุมัติ {p_reject:.3f}")
    lines.append("เช็กลิสต์เตรียมยื่นสมัคร")

    for idx, (title, query) in enumerate(APPROVED_CHECKLIST_QUERIES, start=1):
        answer, evidence = _rag_fetch(rag_lookup, query)
        if answer and answer != NO_ANSWER_SENTINEL:
            line = f"{idx}) {title}: {_trim_text(answer, 150)}"
        else:
            line = f"{idx}) {title}: {NO_ANSWER_SENTINEL}"

        if evidence:
            line += f" (แหล่งข้อมูล: {evidence[0].get('source_title', 'N/A')})"
        lines.append(line)

    lines.append("หมายเหตุ: ควรตรวจสอบเงื่อนไขล่าสุดกับธนาคารอีกครั้งก่อนยื่นจริง")
    return _normalize_whitespace("\n".join(lines))


def generate_response(
    user_input: dict,
    model_output: dict,
    shap_json: dict,
    rag_lookup: Optional[Callable[[str], dict]] = None,
) -> dict:
    decision = parse_model_output(model_output)

    if decision["approved"]:
        result_th = _build_approved_checklist(decision, rag_lookup)
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
    return {
        "mode": "improvement_plan",
        "decision": decision,
        "result_th": render_plan_th(plan, style="123"),
        "plan": plan,
    }
