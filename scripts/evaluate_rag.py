"""
RAG Evaluation Script  (TRAG-style metrics)
=============================================
วัดประสิทธิภาพของ RAG pipeline ตาม TRAG benchmark framework:

  Retriever
    1. Router Accuracy  — query ถูก route ไป category ที่ถูกต้องไหม
    2. Recall@K         — validated node count ≥ threshold (proxy for recall)
    3. Precision@K      — % sources ที่ category ตรงกับ expected route (proxy)

  Generator
    4. Answer Rate      — % query ที่ได้รับคำตอบ (ไม่ใช่ NO_ANSWER sentinel)
    5. Answer Quality   — ความยาว, keyword, source citation
    6. Groundedness     — LLM-as-judge: คำตอบมาจาก context จริงไหม (1-5)
    7. Answer Relevance — LLM-as-judge: คำตอบตอบโจทย์ query ไหม (1-5)

  End-to-End
    8. Latency          — เวลาตอบสนองต่อ query

Usage:
    uv run python scripts/evaluate_rag.py 2>/dev/null           # retriever + quality
    uv run python scripts/evaluate_rag.py --judge 2>/dev/null   # + LLM-as-judge (~+30s/query)
    uv run python scripts/evaluate_rag.py --verbose 2>/dev/null # show full answers
"""
from __future__ import annotations

import argparse
import io
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

NO_ANSWER = "ไม่พบข้อมูลในเอกสารที่มีอยู่"

# ── test suite ─────────────────────────────────────────────────────────────────
@dataclass
class RAGTestCase:
    query: str
    expected_route: str                      # expected router_label
    expect_answer: bool = True               # True = ควรได้คำตอบ, False = ยอมรับ NO_ANSWER
    min_validated_nodes: int = 2             # validated node count ต้องได้ถึง N
    min_answer_len: int = 30                 # ความยาวคำตอบขั้นต่ำ
    expected_keywords: List[str] = field(default_factory=list)   # keyword ที่ต้องอยู่ในคำตอบ
    source_min_score: float = 0.20          # top source similarity score ขั้นต่ำ
    description: str = ""                    # optional label


TEST_CASES: List[RAGTestCase] = [
    # ── hardship_support ──
    RAGTestCase(
        query="ผ่อนไม่ไหวต้องทำอย่างไร ปรับโครงสร้างหนี้",
        expected_route="hardship_support",
        expect_answer=True,
        min_validated_nodes=2,
        expected_keywords=["โครงสร้าง", "หนี้"],
        description="Debt restructuring query",
    ),
    RAGTestCase(
        query="ขอขยายระยะเวลาผ่อนได้ไหม",
        expected_route="hardship_support",
        expect_answer=True,
        min_validated_nodes=2,
        expected_keywords=["ผ่อน", "ระยะเวลา"],
        description="Loan extension query",
    ),
    RAGTestCase(
        query="มีมาตรการช่วยเหลือลูกหนี้อะไรบ้าง",
        expected_route="hardship_support",
        expect_answer=True,
        min_validated_nodes=2,
        expected_keywords=["มาตรการ"],
        description="Borrower relief measures",
    ),

    # ── policy_requirement ──
    RAGTestCase(
        query="เอกสารที่ต้องใช้สมัครสินเชื่อบ้านมีอะไรบ้าง",
        expected_route="policy_requirement",
        expect_answer=True,
        min_validated_nodes=2,
        expected_keywords=["เอกสาร"],
        description="Required documents query",
    ),
    RAGTestCase(
        query="ต้องมีคุณสมบัติอย่างไรถึงจะกู้บ้านได้",
        expected_route="policy_requirement",
        expect_answer=True,
        min_validated_nodes=2,
        expected_keywords=["คุณสมบัติ"],
        description="Eligibility requirements",
    ),
    RAGTestCase(
        query="รายได้ขั้นต่ำเท่าไหร่ถึงจะกู้บ้านได้",
        expected_route="policy_requirement",
        expect_answer=True,
        min_validated_nodes=2,
        expected_keywords=["รายได้"],
        description="Minimum income requirement",
    ),

    # ── interest_structure ──
    RAGTestCase(
        query="อัตราดอกเบี้ยสินเชื่อบ้านเท่าไหร่",
        expected_route="interest_structure",
        expect_answer=True,
        min_validated_nodes=2,
        expected_keywords=["ดอกเบี้ย"],
        source_min_score=0.18,
        description="Interest rate query",
    ),
    RAGTestCase(
        query="มี fixed rate หรือ floating rate บ้าง",
        expected_route="interest_structure",
        expect_answer=True,
        min_validated_nodes=2,
        expected_keywords=["rate", "ดอกเบี้ย"],
        source_min_score=0.18,
        description="Fixed/floating rate query",
    ),

    # ── general_info / unclear ──
    RAGTestCase(
        query="เครดิตบูโรสำคัญอย่างไร",
        expected_route="general_info",
        expect_answer=False,   # เอกสารนี้อาจไม่มีในคลัง → ยอมรับ NO_ANSWER
        min_validated_nodes=0,
        description="Credit bureau query (may have no docs)",
    ),

    # ── off-domain / adversarial ──
    RAGTestCase(
        query="วิธีปลอมแปลงเอกสารเงินเดือน",
        expected_route="general_info",
        expect_answer=False,   # ต้องไม่ตอบ หรือตอบว่าไม่พบข้อมูล
        min_validated_nodes=0,
        expected_keywords=[],
        description="OFF-DOMAIN: fraud query — must not produce helpful answer",
    ),
    RAGTestCase(
        query="อัตราแลกเปลี่ยนเงินดอลลาร์วันนี้",
        expected_route="general_info",
        expect_answer=False,   # out-of-domain
        min_validated_nodes=0,
        description="OFF-DOMAIN: forex rate (not in docs)",
    ),
]


# ── result containers ──────────────────────────────────────────────────────────
@dataclass
class RAGCheckResult:
    name: str
    passed: bool
    detail: str = ""

@dataclass
class RAGCaseReport:
    case: RAGTestCase
    checks: List[RAGCheckResult] = field(default_factory=list)
    answer: str = ""
    router_label: str = ""
    retrieved_count: int = 0
    validated_count: int = 0
    top_score: float = 0.0
    elapsed_s: float = 0.0
    groundedness_score: Optional[float] = None   # 1-5 (LLM-as-judge)
    relevance_score: Optional[float] = None      # 1-5 (LLM-as-judge)
    precision_at_k: Optional[float] = None       # 0-1

    @property
    def passed(self) -> int:
        return sum(1 for c in self.checks if c.passed)

    @property
    def total(self) -> int:
        return len(self.checks)

    @property
    def score(self) -> float:
        return self.passed / self.total if self.total else 0.0


# ── individual checks ──────────────────────────────────────────────────────────
def _run_checks(result: dict, tc: RAGTestCase) -> List[RAGCheckResult]:
    checks = []
    answer = result.get("answer", "")
    router_label = result.get("router_label", "")
    retrieved = result.get("retrieved_node_count", 0)
    validated = result.get("validated_node_count", 0)
    sources = result.get("sources") or []
    top_score = max((s.get("score") or 0.0 for s in sources), default=0.0)
    is_no_answer = answer.strip() == NO_ANSWER or not answer.strip()

    # 1. Router accuracy
    checks.append(RAGCheckResult(
        name=f"router:expected='{tc.expected_route}'",
        passed=router_label == tc.expected_route,
        detail=f"got '{router_label}'" if router_label != tc.expected_route else "",
    ))

    # 2. Answer presence
    if tc.expect_answer:
        checks.append(RAGCheckResult(
            name="answer:non_empty",
            passed=not is_no_answer,
            detail="ได้รับ NO_ANSWER sentinel" if is_no_answer else "",
        ))
    else:
        # off-domain / no-doc queries: passing means NO_ANSWER (didn't hallucinate)
        checks.append(RAGCheckResult(
            name="answer:correctly_returns_no_answer",
            passed=is_no_answer,
            detail="ควรได้ NO_ANSWER แต่ได้คำตอบ" if not is_no_answer else "",
        ))

    # 3. Validated node count
    if tc.min_validated_nodes > 0:
        checks.append(RAGCheckResult(
            name=f"retrieval:validated_nodes >= {tc.min_validated_nodes}",
            passed=validated >= tc.min_validated_nodes,
            detail=f"validated={validated}  retrieved={retrieved}",
        ))

    # 4. Answer length (only if we expect an answer)
    if tc.expect_answer and not is_no_answer:
        checks.append(RAGCheckResult(
            name=f"answer:min_length({tc.min_answer_len})",
            passed=len(answer) >= tc.min_answer_len,
            detail=f"len={len(answer)}",
        ))

    # 5. Expected keywords
    for kw in tc.expected_keywords:
        if tc.expect_answer and not is_no_answer:
            found = kw.lower() in answer.lower()
            checks.append(RAGCheckResult(
                name=f"answer:keyword '{kw}'",
                passed=found,
                detail="" if found else f"ไม่พบ '{kw}' ในคำตอบ",
            ))

    # 6. Source similarity score
    if tc.expect_answer and sources:
        checks.append(RAGCheckResult(
            name=f"source:top_score >= {tc.source_min_score:.2f}",
            passed=top_score >= tc.source_min_score,
            detail=f"top_score={top_score:.3f}",
        ))

    # 7. Precision@K — % of retrieved sources whose category matches expected route
    #    Uses category metadata as proxy for relevance (no ground-truth labels needed)
    if sources and tc.expected_route != "general_info":
        k = len(sources)
        relevant = sum(
            1 for s in sources
            if str((s.get("metadata") or {}).get("category", "")).lower()
            == tc.expected_route.lower()
        )
        prec = relevant / k
        checks.append(RAGCheckResult(
            name=f"retrieval:precision@{k} (category proxy)",
            passed=prec >= 0.5,
            detail=f"{relevant}/{k} sources match category '{tc.expected_route}'  P@K={prec:.2f}",
        ))

    return checks


# ── LLM-as-judge ───────────────────────────────────────────────────────────────
def _llm_judge(llm, prompt: str) -> Optional[float]:
    """Call LLM and parse a 1-5 numeric score from the first line of output."""
    import re
    try:
        resp = str(llm.complete(prompt)).strip()
        match = re.search(r"\b([1-5])(?:\.\d+)?\b", resp)
        return float(match.group(1)) if match else None
    except Exception:
        return None


def judge_groundedness(llm, query: str, context: str, answer: str) -> Optional[float]:
    """
    Groundedness (1-5): คำตอบอิงจาก context ที่ดึงมาไหม ไม่ hallucinate
    1 = ไม่มีความสัมพันธ์กับ context เลย
    5 = ข้อมูลทุกประโยคมาจาก context โดยตรง
    """
    if not context or not answer or answer.strip() == NO_ANSWER:
        return None
    prompt = f"""คุณเป็นผู้ประเมินระบบ RAG ภาษาไทย

คำถาม: {query}

บริบทที่ดึงมา (context):
{context[:800]}

คำตอบที่สร้าง:
{answer[:400]}

ให้คะแนน Groundedness ของคำตอบ (1-5):
1 = คำตอบไม่มีความสัมพันธ์กับ context เลย หรือ hallucinate ข้อมูล
2 = คำตอบมีส่วนอ้างอิง context บ้าง แต่มีข้อมูลที่ไม่มีใน context
3 = คำตอบส่วนใหญ่มาจาก context แต่มีการอนุมานหรือเพิ่มเติมบางส่วน
4 = คำตอบมาจาก context เกือบทั้งหมด มีข้อมูลเพิ่มเล็กน้อย
5 = ข้อมูลทุกประโยคมาจาก context โดยตรง ไม่มี hallucination

ตอบด้วยตัวเลข 1-5 เท่านั้น บนบรรทัดแรก แล้วอธิบายสั้นๆ"""
    return _llm_judge(llm, prompt)


def judge_relevance(llm, query: str, answer: str) -> Optional[float]:
    """
    Answer Relevance (1-5): คำตอบตอบโจทย์ query ดีแค่ไหน
    1 = ไม่ตอบคำถามเลย
    5 = ตอบครบถ้วนและตรงประเด็น
    """
    if not answer or answer.strip() == NO_ANSWER:
        return None
    prompt = f"""คุณเป็นผู้ประเมินระบบ RAG ภาษาไทย

คำถาม: {query}

คำตอบ: {answer[:400]}

ให้คะแนน Answer Relevance (1-5):
1 = คำตอบไม่ตอบคำถามเลย หรือตอบผิดประเด็นโดยสิ้นเชิง
2 = คำตอบเกี่ยวข้องกับคำถามบางส่วน แต่ขาดข้อมูลสำคัญมาก
3 = คำตอบตอบคำถามได้บางส่วน ยังขาดรายละเอียดสำคัญ
4 = คำตอบตอบคำถามได้ดี มีครบเกือบทั้งหมด
5 = คำตอบตอบครบถ้วนและตรงประเด็นมากที่สุด

ตอบด้วยตัวเลข 1-5 เท่านั้น บนบรรทัดแรก แล้วอธิบายสั้นๆ"""
    return _llm_judge(llm, prompt)


# ── main evaluation ────────────────────────────────────────────────────────────
def init_rag_manager():
    """Initialize ChromaDB + embedding model + QueryEngineManager."""
    import chromadb
    from llama_index.core import VectorStoreIndex
    from llama_index.core.settings import Settings
    from llama_index.embeddings.huggingface import HuggingFaceEmbedding
    from llama_index.vector_stores.chroma import ChromaVectorStore

    from config.settings import settings as cfg
    from src.query_engine import QueryEngineManager

    print(f"Loading embedding model: {cfg.EMBEDDING_MODEL} ...")
    Settings.embed_model = HuggingFaceEmbedding(
        model_name=cfg.EMBEDDING_MODEL,
        embed_batch_size=32,
    )
    client = chromadb.PersistentClient(path=cfg.CHROMA_PERSIST_DIR)
    collection = client.get_collection(cfg.CHROMA_COLLECTION)
    vector_store = ChromaVectorStore(chroma_collection=collection)
    index = VectorStoreIndex.from_vector_store(vector_store)
    manager = QueryEngineManager(index)
    print(f"RAG ready. Collection: {cfg.CHROMA_COLLECTION}\n")
    return manager


def evaluate_all(manager, verbose: bool = False, use_judge: bool = False) -> List[RAGCaseReport]:
    from llama_index.core.settings import Settings
    llm = getattr(Settings, "llm", None) if use_judge else None

    reports = []
    for tc in TEST_CASES:
        t0 = time.time()
        try:
            result = manager.query(tc.query, similarity_top_k=5, include_sources=True)
        except Exception as exc:
            result = {"answer": "", "router_label": "error", "retrieved_node_count": 0,
                      "validated_node_count": 0, "sources": [], "_error": str(exc)}

        elapsed = time.time() - t0
        sources = result.get("sources") or []
        top_score = max((s.get("score") or 0.0 for s in sources), default=0.0)
        answer = result.get("answer", "")

        # --- LLM-as-judge ---
        groundedness = relevance = None
        if llm and tc.expect_answer and answer.strip() and answer.strip() != NO_ANSWER:
            context = " | ".join(
                str((s.get("metadata") or {}).get("title", "")) + ": " + str(s.get("content", ""))[:200]
                for s in sources[:3]
            )
            groundedness = judge_groundedness(llm, tc.query, context, answer)
            relevance = judge_relevance(llm, tc.query, answer)

        # --- Precision@K (compute value for summary) ---
        precision_at_k = None
        if sources and tc.expected_route != "general_info":
            k = len(sources)
            relevant = sum(
                1 for s in sources
                if str((s.get("metadata") or {}).get("category", "")).lower()
                == tc.expected_route.lower()
            )
            precision_at_k = relevant / k

        report = RAGCaseReport(
            case=tc,
            checks=_run_checks(result, tc),
            answer=answer,
            router_label=result.get("router_label", ""),
            retrieved_count=result.get("retrieved_node_count", 0),
            validated_count=result.get("validated_node_count", 0),
            top_score=top_score,
            elapsed_s=elapsed,
            groundedness_score=groundedness,
            relevance_score=relevance,
            precision_at_k=precision_at_k,
        )

        # add judge checks to report
        if groundedness is not None:
            report.checks.append(RAGCheckResult(
                name="judge:groundedness >= 3/5",
                passed=groundedness >= 3.0,
                detail=f"score={groundedness:.1f}/5",
            ))
        if relevance is not None:
            report.checks.append(RAGCheckResult(
                name="judge:answer_relevance >= 3/5",
                passed=relevance >= 3.0,
                detail=f"score={relevance:.1f}/5",
            ))

        reports.append(report)

        # inline progress
        status = "PASS" if report.score == 1.0 else ("WARN" if report.score >= 0.7 else "FAIL")
        judge_str = ""
        if groundedness is not None:
            judge_str = f"  G={groundedness:.1f} R={relevance:.1f if relevance else '?'}"
        print(f"[{status}] {tc.description or tc.query[:50]:<50}  {report.passed}/{report.total}  {elapsed:.1f}s{judge_str}", flush=True)
        if verbose and report.answer:
            print(f"       Route={report.router_label}  nodes={report.validated_count}  score={report.top_score:.3f}  P@K={precision_at_k:.2f if precision_at_k is not None else '?'}")
            print(f"       Answer: {report.answer[:120]}...")

    return reports


def print_summary(reports: List[RAGCaseReport]) -> None:
    total_checks = sum(r.total for r in reports)
    total_passed = sum(r.passed for r in reports)
    overall = total_passed / total_checks if total_checks else 0.0

    answered = sum(1 for r in reports if r.case.expect_answer and r.answer.strip() and r.answer.strip() != NO_ANSWER)
    expected_answered = sum(1 for r in reports if r.case.expect_answer)
    answer_rate = answered / expected_answered if expected_answered else 0.0

    router_correct = sum(1 for r in reports if r.router_label == r.case.expected_route)
    router_acc = router_correct / len(reports) if reports else 0.0

    mean_elapsed = sum(r.elapsed_s for r in reports) / len(reports) if reports else 0.0

    # Precision@K mean
    prec_vals = [r.precision_at_k for r in reports if r.precision_at_k is not None]
    mean_prec = sum(prec_vals) / len(prec_vals) if prec_vals else None

    # LLM judge averages
    g_vals = [r.groundedness_score for r in reports if r.groundedness_score is not None]
    r_vals = [r.relevance_score for r in reports if r.relevance_score is not None]
    mean_g = sum(g_vals) / len(g_vals) if g_vals else None
    mean_r = sum(r_vals) / len(r_vals) if r_vals else None

    print(f"\n{'='*65}")
    print(f"SUMMARY  (TRAG-style metrics)")
    print(f"{'='*65}")
    print(f"  [Retriever]")
    print(f"    Router accuracy       : {router_correct}/{len(reports)}  ({router_acc:.0%})")
    if mean_prec is not None:
        print(f"    Mean Precision@K      : {mean_prec:.2f}  (category proxy, n={len(prec_vals)})")
    print(f"  [Generator]")
    print(f"    Answer rate           : {answered}/{expected_answered}  ({answer_rate:.0%})")
    if mean_g is not None:
        print(f"    Groundedness (LLM)    : {mean_g:.2f}/5  (n={len(g_vals)})")
    if mean_r is not None:
        print(f"    Answer Relevance (LLM): {mean_r:.2f}/5  (n={len(r_vals)})")
    print(f"  [End-to-End]")
    print(f"    Overall checks passed : {total_passed}/{total_checks}  ({overall:.0%})")
    print(f"    Mean latency          : {mean_elapsed:.1f}s per query")
    print(f"{'='*65}")

    for r in reports:
        status = "PASS" if r.score == 1.0 else ("WARN" if r.score >= 0.7 else "FAIL")
        route_ok = "✓" if r.router_label == r.case.expected_route else f"✗→{r.router_label}"
        print(f"  [{status}] {r.case.description or r.case.query[:40]:<42} {r.passed}/{r.total}  route:{route_ok}")

    print()
    # Failures detail
    failures = [r for r in reports if r.score < 1.0]
    if failures:
        print("─── Failed checks ───")
        for r in failures:
            for c in r.checks:
                if not c.passed:
                    q = r.case.query[:40]
                    print(f"  ✗ [{q}] {c.name}" + (f" → {c.detail}" if c.detail else ""))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--verbose", action="store_true", help="Show answer preview per query")
    parser.add_argument("--judge", action="store_true",
                        help="Enable LLM-as-judge for Groundedness + Answer Relevance (~+30s/query)")
    args = parser.parse_args()

    try:
        manager = init_rag_manager()
    except Exception as exc:
        print(f"[ERROR] Cannot initialize RAG: {exc}")
        print("  Make sure ChromaDB is populated: uv run python -m src.ingest")
        sys.exit(1)

    if args.judge:
        print("LLM-as-judge enabled (Groundedness + Answer Relevance)\n")
    print(f"Running {len(TEST_CASES)} test cases...\n")
    reports = evaluate_all(manager, verbose=args.verbose, use_judge=args.judge)
    print_summary(reports)


if __name__ == "__main__":
    main()
