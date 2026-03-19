"""Self-RAG orchestrator for the Thai home-loan RAG pipeline.

Adds three reflection steps on top of QueryEngineManager:
  1. [Retrieve]  — Is retrieval needed for this query?  (binary yes/no)
  2. [IsRel]     — Score each retrieved node for relevance (1-5), drop < threshold
  3. [IsSup]     — Is the generated answer supported by context?  If not, retry once.

LLM call budget per query:
  Happy path (IsSup passes):          [Retrieve](1) + [IsRel](1) + synthesis(1) + [IsSup](1) = 4
  IsSup retry (retry also passes):    4 + retry_synthesis(1) + retry_IsSup(1) = 6
  Retrieve=No (off-domain / greeting): 1
"""
from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

NO_ANSWER_MESSAGE = "ไม่พบข้อมูลในเอกสารที่มีอยู่"

# ── Prompt templates ──────────────────────────────────────────────────────────

_RETRIEVE_PROMPT = """\
คุณเป็นผู้ช่วยระบบ RAG สำหรับสินเชื่อบ้านไทย
คำถาม: {question}

จำเป็นต้องค้นหาข้อมูลจากเอกสารธนาคารเพื่อตอบคำถามนี้หรือไม่?
ตอบ "yes" หรือ "no" บนบรรทัดแรกเท่านั้น""".strip()

_ISREL_PROMPT = """\
คุณเป็นผู้ประเมินความเกี่ยวข้องของเอกสารสำหรับระบบ RAG สินเชื่อบ้านไทย

คำถาม: {question}

เอกสารที่ดึงมา:
{node_blocks}

ให้คะแนนความเกี่ยวข้องของแต่ละเอกสารกับคำถาม (1-5):
1 = ไม่เกี่ยวข้องเลย
3 = เกี่ยวข้องบางส่วน
5 = เกี่ยวข้องมากที่สุด

ตอบเป็น JSON array เท่านั้น เช่น: [{{"id": 0, "score": 4}}, {{"id": 1, "score": 2}}]""".strip()

_RESYNTH_PROMPT = """\
คุณเป็นผู้ช่วยตอบคำถามเกี่ยวกับสินเชื่อบ้านจากธนาคาร
ใช้เฉพาะข้อมูลจากบริบทที่ให้มาเท่านั้น ห้ามใช้ความรู้ภายนอก

กฎ:
1) ตอบเป็นภาษาเดียวกับคำถาม
2) ถ้าบริบทไม่เพียงพอ ตอบว่า "ไม่พบข้อมูลในเอกสารที่มีอยู่" เท่านั้น
3) ห้ามเดาตัวเลข อัตรา เงื่อนไข ที่ไม่มีในบริบท

บริบท:
{context}

คำถาม: {question}
คำตอบ:""".strip()

_ISSUP_PROMPT = """\
คุณเป็นผู้ประเมินคุณภาพคำตอบ RAG ระบบสินเชื่อบ้านไทย

คำถาม: {question}
บริบทที่ดึงมา: {context}
คำตอบที่สร้าง: {answer}

คำตอบนี้ได้รับการสนับสนุนโดยตรงจากบริบทที่ให้มาแค่ไหน?
1 = ไม่มีการสนับสนุนเลย / hallucinate ชัดเจน
2 = มีบางส่วน แต่มีข้อมูลที่ไม่มีในบริบท
3 = ส่วนใหญ่มาจากบริบท มีการอนุมานบ้าง
4 = มาจากบริบทเกือบทั้งหมด
5 = ทุกประโยคมาจากบริบทโดยตรง

ตอบด้วยตัวเลข 1-5 บนบรรทัดแรก""".strip()

_ISGEN_PROMPT = """\
คุณเป็นผู้ประเมินคุณภาพคำตอบ RAG ระบบสินเชื่อบ้านไทย

คำถาม: {question}
คำตอบ: {answer}

คำตอบนี้ตอบคำถามที่ถามได้ตรงประเด็นแค่ไหน? (ไม่สนใจว่ามาจากเอกสารหรือไม่)
1 = ไม่ตอบคำถามเลย / ตอบผิดประเด็น
2 = ตอบคำถามบางส่วน
3 = ตอบคำถามได้ส่วนใหญ่
4 = ตอบคำถามได้ดี
5 = ตอบคำถามได้ตรงประเด็นและครบถ้วน

ตอบด้วยตัวเลข 1-5 บนบรรทัดแรก""".strip()


# ── Dataclass ────────────────────────────────────────────────────────────────

@dataclass
class SelfRAGTrace:
    """Diagnostic trace emitted per query — serialisable to JSON."""
    retrieve_needed: bool = True
    nodes_before_isrel: int = 0
    nodes_after_isrel: int = 0
    isrel_scores: List[Dict[str, Any]] = field(default_factory=list)  # [{title, score}]
    issup_score: Optional[int] = None
    issup_passed: bool = False
    isgen_score: Optional[int] = None
    isgen_passed: bool = False
    retry_attempted: bool = False
    resynth_used: bool = False          # True when answer was re-synthesised after IsRel
    total_reflection_calls: int = 0
    elapsed_s: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "retrieve_needed": self.retrieve_needed,
            "nodes_before_isrel": self.nodes_before_isrel,
            "nodes_after_isrel": self.nodes_after_isrel,
            "isrel_scores": self.isrel_scores,
            "issup_score": self.issup_score,
            "issup_passed": self.issup_passed,
            "isgen_score": self.isgen_score,
            "isgen_passed": self.isgen_passed,
            "retry_attempted": self.retry_attempted,
            "resynth_used": self.resynth_used,
            "total_reflection_calls": self.total_reflection_calls,
            "elapsed_s": round(self.elapsed_s, 2),
        }


# ── Orchestrator ─────────────────────────────────────────────────────────────

class SelfRAGOrchestrator:
    """
    Wraps QueryEngineManager with Self-RAG reflection steps.

    Usage:
        manager = QueryEngineManager(index)
        orch = SelfRAGOrchestrator(manager)
        result = orch.query("อัตราดอกเบี้ยสินเชื่อบ้านเท่าไหร่")
        # result has same shape as manager.query() + "self_rag_trace" key
    """

    ISREL_THRESHOLD: int = 3          # nodes scoring < this are dropped (scale 1-5)
    ISSUP_THRESHOLD: int = 2          # answer must score >= this to be returned as-is
    ISGEN_THRESHOLD: int = 2          # answer must score >= this to be considered on-topic
    ISSUP_RETRY_TOP_K: int = 15       # broader retrieval top-K on IsSup failure
    MAX_ISREL_NODES: int = 15         # max nodes sent to IsRel prompt (top by score)
    CONTEXT_CHAR_LIMIT: int = 200     # chars per node for IsRel prompts

    def __init__(
        self,
        manager: Any,
        isrel_threshold: int = ISREL_THRESHOLD,
        issup_threshold: int = ISSUP_THRESHOLD,
        isgen_threshold: int = ISGEN_THRESHOLD,
        retry_top_k: int = ISSUP_RETRY_TOP_K,
    ) -> None:
        self.manager = manager
        self.isrel_threshold = isrel_threshold
        self.issup_threshold = issup_threshold
        self.isgen_threshold = isgen_threshold
        self.retry_top_k = retry_top_k
        self._llm = manager.llm

    # ── Public interface ──────────────────────────────────────────────────────

    def query(
        self,
        question: str,
        similarity_top_k: Optional[int] = None,
        response_mode: Optional[str] = None,
        include_sources: bool = True,
    ) -> Dict[str, Any]:
        """
        Self-RAG query.  Drop-in replacement for manager.query() with the same
        return dict shape, plus an additional key: "self_rag_trace".
        """
        t0 = time.monotonic()
        trace = SelfRAGTrace()

        # ── [Retrieve] ───────────────────────────────────────────────────────
        needs_retrieval = self._reflect_retrieve(question)
        trace.retrieve_needed = needs_retrieval
        trace.total_reflection_calls += 1

        if not needs_retrieval:
            trace.elapsed_s = time.monotonic() - t0
            return self._no_retrieval_result(question, trace)

        # ── Base pipeline (retrieval + synthesis) ────────────────────────────
        base_result = self.manager.query(
            question=question,
            similarity_top_k=similarity_top_k,
            response_mode=response_mode,
            include_sources=include_sources,
        )

        sources: List[Dict[str, Any]] = base_result.get("sources") or []
        answer: str = base_result.get("answer", "")

        # ── [IsRel] ───────────────────────────────────────────────────────────
        filtered_sources, isrel_scores = self._reflect_isrel(question, sources)
        trace.nodes_before_isrel = len(sources)
        trace.nodes_after_isrel = len(filtered_sources)
        trace.isrel_scores = isrel_scores
        trace.total_reflection_calls += 1

        if len(filtered_sources) == 0:
            # All nodes dropped — cannot answer
            trace.issup_passed = False
            trace.elapsed_s = time.monotonic() - t0
            result = dict(base_result)
            result["answer"] = NO_ANSWER_MESSAGE
            result["sources"] = []
            result["self_rag_trace"] = trace.to_dict()
            return result

        # Re-synthesise if IsRel dropped any nodes
        if len(filtered_sources) < len(sources):
            answer = self._resynthesize(question, filtered_sources)
            trace.resynth_used = True
            trace.total_reflection_calls += 1

        # ── [IsSup] ──────────────────────────────────────────────────────────
        context_text = self._sources_to_context(filtered_sources)
        issup_score = self._reflect_issup(question, answer, context_text)
        trace.issup_score = issup_score
        trace.total_reflection_calls += 1

        if issup_score is not None and issup_score >= self.issup_threshold:
            trace.issup_passed = True
        else:
            # IsSup failed — retry with broader top-K
            trace.retry_attempted = True
            logger.debug("[IsSup] score=%s < threshold=%s, retrying with top_k=%s",
                         issup_score, self.issup_threshold, self.retry_top_k)

            retry_result = self.manager.query(
                question=question,
                similarity_top_k=self.retry_top_k,
                response_mode=response_mode,
                include_sources=include_sources,
            )
            retry_answer = retry_result.get("answer", "")
            retry_sources = retry_result.get("sources") or []
            retry_context = self._sources_to_context(retry_sources)
            retry_score = self._reflect_issup(question, retry_answer, retry_context)
            trace.issup_score = retry_score
            trace.total_reflection_calls += 2  # retry synthesis counted separately

            if retry_score is not None and retry_score >= self.issup_threshold:
                trace.issup_passed = True
                answer = retry_answer
                filtered_sources = retry_sources
                base_result = retry_result
            else:
                # Both IsSup attempts failed — return NO_ANSWER
                trace.issup_passed = False
                trace.elapsed_s = time.monotonic() - t0
                result = dict(base_result)
                result["answer"] = NO_ANSWER_MESSAGE
                result["self_rag_trace"] = trace.to_dict()
                return result

        # ── [IsGen] ───────────────────────────────────────────────────────────
        isgen_score = self._reflect_isgen(question, answer)
        trace.isgen_score = isgen_score
        trace.total_reflection_calls += 1

        if isgen_score is not None and isgen_score >= self.isgen_threshold:
            trace.isgen_passed = True
        else:
            # Answer is grounded (IsSup passed) but does not address the question
            logger.debug("[IsGen] score=%s < threshold=%s — answer off-topic",
                         isgen_score, self.isgen_threshold)
            trace.isgen_passed = False
            trace.elapsed_s = time.monotonic() - t0
            result = dict(base_result)
            result["answer"] = NO_ANSWER_MESSAGE
            result["sources"] = []
            result["self_rag_trace"] = trace.to_dict()
            return result

        trace.elapsed_s = time.monotonic() - t0
        result = dict(base_result)
        result["answer"] = answer
        result["sources"] = filtered_sources
        result["self_rag_trace"] = trace.to_dict()
        return result

    # ── Reflection helpers (one LLM call each) ───────────────────────────────

    def _reflect_retrieve(self, question: str) -> bool:
        """
        [Retrieve]: does this query need document retrieval?

        Fast-path: if the query contains any domain keyword, always retrieve
        without asking the LLM.  Only queries that don't match any keyword go
        to the LLM, where truly off-domain questions (greetings, foreign topics)
        are filtered out.
        """
        from src.rag.router import ROUTE_KEYWORDS
        text = (question or "").lower()
        # Any domain keyword → definitely needs retrieval (skip LLM call)
        all_domain_terms = [kw for kws in ROUTE_KEYWORDS.values() for kw in kws]
        if any(term in text for term in all_domain_terms):
            return True

        prompt = _RETRIEVE_PROMPT.format(question=question)
        try:
            raw = str(self._llm.complete(prompt)).strip().lower()
            first_line = raw.splitlines()[0] if raw else ""
            # Explicit "no" → skip retrieval
            if re.search(r"\bno\b|ไม่จำเป็น|ไม่ต้อง", first_line):
                return False
        except Exception as exc:
            logger.debug("[Retrieve] LLM call failed, defaulting to True: %s", exc)
        return True

    def _reflect_isrel(
        self,
        question: str,
        sources: List[Dict[str, Any]],
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        [IsRel]: score each source node (1-5) and drop those below threshold.

        Batches all nodes into ONE LLM call.
        Returns (filtered_sources, isrel_scores_list).
        """
        if not sources:
            return sources, []

        # Take at most MAX_ISREL_NODES (top by score, already sorted by manager)
        candidates = sources[:self.MAX_ISREL_NODES]

        node_blocks = "\n\n".join(
            f"[{i}] {self._source_title(s)}: {self._source_snippet(s)}"
            for i, s in enumerate(candidates)
        )
        prompt = _ISREL_PROMPT.format(question=question, node_blocks=node_blocks)

        scores_by_id: Dict[int, int] = {}
        try:
            raw = str(self._llm.complete(prompt)).strip()
            # Extract JSON array from response
            json_match = re.search(r"\[.*?\]", raw, re.DOTALL)
            if json_match:
                parsed = json.loads(json_match.group(0))
                for entry in parsed:
                    scores_by_id[int(entry["id"])] = int(entry["score"])
        except Exception as exc:
            logger.debug("[IsRel] parse failed, keeping all nodes: %s", exc)
            # Defensive: if LLM/parse fails, keep all nodes
            return sources, []

        # Default score for unscored nodes is threshold-1 (treated as not relevant)
        _default = self.isrel_threshold - 1
        isrel_scores = [
            {"title": self._source_title(s), "score": scores_by_id.get(i, _default)}
            for i, s in enumerate(candidates)
        ]

        filtered = [
            s for i, s in enumerate(candidates)
            if scores_by_id.get(i, _default) >= self.isrel_threshold
        ]

        # Nodes beyond MAX_ISREL_NODES are treated as not scored → drop them
        # (They were not evaluated, so we cannot confirm relevance)
        # Empty filtered list propagates to query() → returns NO_ANSWER_MESSAGE
        return filtered, isrel_scores

    def _reflect_issup(
        self,
        question: str,
        answer: str,
        context: str,
    ) -> Optional[int]:
        """[IsSup]: score answer groundedness (1-5). Falls back to threshold on failure."""
        if not answer or answer.strip() == NO_ANSWER_MESSAGE:
            return None
        prompt = _ISSUP_PROMPT.format(
            question=question,
            context=context[:1500],
            answer=answer[:500],
        )
        try:
            raw = str(self._llm.complete(prompt)).strip()
            match = re.search(r"\b([1-5])\b", raw)
            if match:
                return int(match.group(1))
        except Exception as exc:
            logger.debug("[IsSup] LLM call failed, using threshold as default: %s", exc)
        return self.issup_threshold  # fail-safe: don't reject on LLM error

    def _reflect_isgen(self, question: str, answer: str) -> Optional[int]:
        """[IsGen]: score whether the answer actually addresses the question (1-5).

        Unlike [IsSup] which checks groundedness in retrieved context, [IsGen]
        checks relevance of the answer TO THE QUESTION — catching cases where
        the answer is grounded but talks about a different topic.
        Falls back to threshold on LLM error (fail-safe).
        """
        if not answer or answer.strip() == NO_ANSWER_MESSAGE:
            return None
        prompt = _ISGEN_PROMPT.format(question=question, answer=answer[:400])
        try:
            raw = str(self._llm.complete(prompt)).strip()
            match = re.search(r"\b([1-5])\b", raw)
            if match:
                return int(match.group(1))
        except Exception as exc:
            logger.debug("[IsGen] LLM call failed, using threshold as default: %s", exc)
        return self.isgen_threshold  # fail-safe: don't reject on LLM error

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _resynthesize(self, question: str, sources: List[Dict[str, Any]]) -> str:
        """Re-synthesise answer using only IsRel-filtered sources."""
        context = "\n\n".join(
            f"{self._source_title(s)}: {s.get('content', '')}"
            for s in sources
        )
        prompt = _RESYNTH_PROMPT.format(context=context[:2000], question=question)
        try:
            raw = str(self._llm.complete(prompt)).strip()
            return raw if raw else NO_ANSWER_MESSAGE
        except Exception as exc:
            logger.warning("[Resynth] LLM call failed: %s", exc)
            return NO_ANSWER_MESSAGE

    def _sources_to_context(self, sources: List[Dict[str, Any]]) -> str:
        return " | ".join(
            f"{self._source_title(s)}: {self._source_snippet(s)}"
            for s in sources[:8]
        )

    @staticmethod
    def _source_title(source: Dict[str, Any]) -> str:
        meta = source.get("metadata") or {}
        if isinstance(meta, dict):
            return str(meta.get("title", meta.get("file_name", "Unknown")))
        return "Unknown"

    @staticmethod
    def _source_snippet(source: Dict[str, Any], limit: int = 250) -> str:
        content = str(source.get("content", ""))
        return content[:limit] + "..." if len(content) > limit else content

    @staticmethod
    def _no_retrieval_result(question: str, trace: SelfRAGTrace) -> Dict[str, Any]:
        return {
            "question": question,
            "answer": NO_ANSWER_MESSAGE,
            "router_label": "general_info",
            "retrieved_node_count": 0,
            "validated_node_count": 0,
            "sources": [],
            "self_rag_trace": trace.to_dict(),
        }
