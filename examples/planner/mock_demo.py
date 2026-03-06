from __future__ import annotations

import json
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

from src.planner.planning import NO_ANSWER_SENTINEL, generate_response, render_plan_th


def _load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def mock_rag_lookup(query: str) -> dict:
    kb = {
        "เอกสารที่ต้องใช้สมัครสินเชื่อบ้านมีอะไรบ้าง": (
            "ใช้บัตรประชาชน ทะเบียนบ้าน และเอกสารแสดงรายได้ตามประเภทอาชีพ",
            "โฮมโลนฟอร์ยู | สินเชื่อบ้าน | CIMB TH",
            "policy_requirement",
            0.91,
        ),
        "ต้องมีคุณสมบัติอย่างไรถึงจะกู้บ้านได้": (
            "ต้องมีสัญชาติไทยและมีรายได้สม่ำเสมอตามเกณฑ์ธนาคาร",
            "โฮมโลนฟอร์ยู | สินเชื่อบ้าน | CIMB TH",
            "policy_requirement",
            0.89,
        ),
        "รายได้ขั้นต่ำเท่าไหร่ถึงจะกู้บ้านได้": (
            "รายได้ขั้นต่ำเป็นไปตามเงื่อนไขผลิตภัณฑ์และประเภทผู้กู้",
            "โฮมโลนฟอร์ยู | สินเชื่อบ้าน | CIMB TH",
            "policy_requirement",
            0.84,
        ),
        "ผ่อนไม่ไหวต้องทำอย่างไร ปรับโครงสร้างหนี้": (
            "สามารถยื่นคำขอปรับโครงสร้างหนี้และขยายงวดผ่อนภายใต้เงื่อนไขธนาคาร",
            "ใบคำขอปรับปรุงโครงสร้างหนี้ (ขยายระยะเวลาผ่อน)",
            "hardship_support",
            0.87,
        ),
        "ขอขยายระยะเวลาผ่อนได้ไหม": (
            "สามารถขอขยายระยะเวลาผ่อนได้ โดยธนาคารจะพิจารณาตามความสามารถชำระ",
            "ใบคำขอปรับปรุงโครงสร้างหนี้ (ขยายระยะเวลาผ่อน)",
            "hardship_support",
            0.86,
        ),
        "มีมาตรการช่วยเหลือลูกหนี้อะไรบ้าง": (
            "มีมาตรการช่วยเหลือลูกหนี้เป็นระยะตามประกาศของธนาคาร",
            "มาตรการช่วยเหลือลูกหนี้ระยะที่ 2",
            "hardship_support",
            0.83,
        ),
        "อัตราดอกเบี้ยสินเชื่อบ้านเท่าไหร่": (
            "อัตราดอกเบี้ยขึ้นกับแผนสินเชื่อและช่วงเวลาโปรโมชัน",
            "อัตราดอกเบี้ยสินเชื่อบ้านใหม่ (Generic) ปี 2568/2569",
            "interest_structure",
            0.88,
        ),
        "มี fixed rate หรือ floating rate บ้าง": (
            "มีทั้ง fixed rate และ floating rate ตามแผนสินเชื่อ",
            "loan-interest-rates-th.txt",
            "interest_structure",
            0.81,
        ),
        "เครดิตบูโรสำคัญอย่างไร": (
            NO_ANSWER_SENTINEL,
            "",
            "",
            0.0,
        ),
    }

    item = kb.get(query)
    if not item:
        return {"answer": NO_ANSWER_SENTINEL, "sources": []}

    answer, title, category, score = item
    if answer == NO_ANSWER_SENTINEL:
        return {"answer": answer, "sources": []}

    return {
        "answer": answer,
        "sources": [{"title": title, "category": category, "score": score}],
    }


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")

    base = Path(__file__).resolve().parent
    user = _load_json(base / "user.json")
    shap = _load_json(base / "shap.json")
    model_approved = _load_json(base / "model_approved.json")
    model_rejected = _load_json(base / "model_rejected.json")

    approved = generate_response(user, model_approved, shap, rag_lookup=mock_rag_lookup)
    rejected = generate_response(user, model_rejected, shap, rag_lookup=mock_rag_lookup)

    print("=== HUMAN READABLE (Paragraph) ===")
    print(render_plan_th(rejected, style="paragraph"))
    print("\n=== HUMAN READABLE (123) ===")
    print(render_plan_th(rejected, style="123"))

    print("\n=== RAW JSON (APPROVED) ===")
    print(json.dumps(approved, ensure_ascii=False, indent=2))
    print("\n=== RAW JSON (REJECTED) ===")
    print(json.dumps(rejected, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
