"""Multi-hop RAG: query decomposition + per-hop retrieval + final synthesis.

Vanilla RAG retrieves once. Multi-hop RAG:
  1. Decomposes the user question into 2-4 atomic sub-questions
  2. Runs RAG retrieval for each sub-question independently
  3. Merges + deduplicates the retrieved chunks
  4. Synthesises a final answer grounded in the union of contexts

This addresses two failure modes of single-hop RAG:
  - "Wide" questions where one query embedding can't cover all aspects
    (e.g. "ฉันจะกู้ได้ไหม" needs facts about income, age, tenure, DSR)
  - "Sequential" questions where one fact depends on another
    (e.g. "ฉันมีรายได้ 18k DSR ฉันได้เท่าไหร่" needs the DSR rule first)

For the thesis this is a measurable contribution: report retrieval recall
delta vs single-hop on a held-out question set.
"""
from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, List, Optional

from llama_index.core.settings import Settings

logger = logging.getLogger(__name__)


DECOMPOSITION_PROMPT = """\
คุณเป็นผู้ช่วยที่แตกคำถามซับซ้อนเป็นคำถามย่อยที่ค้นข้อมูลแยกกันได้

คำถามต้นฉบับ:
{question}

{profile_block}

คำสั่ง:
1. แตกคำถามต้นฉบับเป็น 2-4 sub-questions ที่:
   - แต่ละ sub-question ค้นข้อมูลจากเอกสารได้ในตัวของมันเอง (atomic)
   - sub-question ครอบคลุมแง่มุมต่างกัน (ห้ามซ้ำซ้อน)
   - ใช้ภาษาไทยที่ใกล้เคียงกับคำที่อยู่ในเอกสารนโยบายธนาคาร
2. ถ้าคำถามต้นฉบับเรียบง่ายอยู่แล้ว (atomic) ให้ส่ง 1 sub-question เท่านั้น
3. ตอบกลับเป็น JSON เท่านั้น รูปแบบ:
{{"sub_questions": ["sub 1", "sub 2", "sub 3"]}}
ห้ามมีข้อความอื่นใด ห้ามใส่ markdown code fence
"""


def _extract_json(text: str) -> Optional[dict]:
    if not text:
        return None
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    snippet = text[start : end + 1]
    try:
        return json.loads(snippet)
    except json.JSONDecodeError:
        snippet2 = re.sub(r",(\s*[}\]])", r"\1", snippet)
        try:
            return json.loads(snippet2)
        except json.JSONDecodeError:
            return None


def decompose_question(question: str, profile_text: str = "") -> List[str]:
    """Use the LLM to break a question into atomic sub-questions.

    Falls back to [question] if LLM is unavailable or output is malformed.
    """
    llm = getattr(Settings, "llm", None)
    if llm is None:
        return [question]

    profile_block = (
        f"บริบทผู้ใช้ (ถ้ามี):\n{profile_text}\n"
        if profile_text.strip()
        else ""
    )
    prompt = DECOMPOSITION_PROMPT.format(
        question=question.strip(),
        profile_block=profile_block,
    )

    try:
        raw = str(llm.complete(prompt))
    except Exception as exc:
        logger.warning("Decomposition LLM call failed: %s", exc)
        return [question]

    parsed = _extract_json(raw)
    if not parsed:
        return [question]

    subs = parsed.get("sub_questions") or []
    if not isinstance(subs, list):
        return [question]

    cleaned: List[str] = []
    seen = set()
    for s in subs:
        text = str(s).strip()
        if not text:
            continue
        # Dedup by lowercased prefix
        key = text.lower()[:60]
        if key in seen:
            continue
        seen.add(key)
        cleaned.append(text)
        if len(cleaned) >= 4:
            break

    return cleaned or [question]


def merge_sources(
    per_hop_sources: List[List[Dict[str, Any]]],
    max_total: int = 12,
) -> List[Dict[str, Any]]:
    """Deduplicate retrieved chunks across hops by content prefix.

    Score is the max across hops (best similarity wins). Returns at most
    `max_total` unique chunks, sorted by score descending.
    """
    by_key: Dict[str, Dict[str, Any]] = {}
    for hop in per_hop_sources:
        for src in hop or []:
            content = (src.get("content") or "").strip()
            if not content:
                continue
            key = content[:200]
            existing = by_key.get(key)
            if existing is None:
                by_key[key] = dict(src)
            else:
                # Keep the higher score
                old_score = existing.get("score") or 0
                new_score = src.get("score") or 0
                if new_score > old_score:
                    existing["score"] = new_score

    merged = list(by_key.values())
    merged.sort(key=lambda s: (s.get("score") or 0), reverse=True)
    return merged[:max_total]


def multihop_retrieve(
    question: str,
    rag_manager: Any,
    profile_text: str = "",
    top_k_per_hop: int = 4,
    max_total_sources: int = 12,
) -> Dict[str, Any]:
    """Run multi-hop retrieval and return merged sources + decomposition trace.

    The output dict shape:
        {
          "sub_questions": [...],
          "sources": [merged + dedup'd chunk dicts],
          "per_hop_counts": [n1, n2, n3],
        }
    """
    sub_questions = decompose_question(question, profile_text=profile_text)
    logger.info("Multi-hop decomposition: %d sub-questions", len(sub_questions))

    per_hop: List[List[Dict[str, Any]]] = []
    per_hop_counts: List[int] = []
    for sq in sub_questions:
        try:
            result = rag_manager.query(sq, similarity_top_k=top_k_per_hop, include_sources=True)
            srcs = result.get("sources", []) or []
        except Exception as exc:
            logger.warning("Multi-hop retrieval failed for %r: %s", sq[:60], exc)
            srcs = []
        per_hop.append(srcs)
        per_hop_counts.append(len(srcs))

    merged = merge_sources(per_hop, max_total=max_total_sources)
    return {
        "sub_questions": sub_questions,
        "sources": merged,
        "per_hop_counts": per_hop_counts,
    }
