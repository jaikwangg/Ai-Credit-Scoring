"""
RAG Synthesis Quality Comparison
=================================
Tests 5 configurations on the same query and prints side-by-side results.

Configs:
  Baseline  — temperature=0.1,  compact,        current prompt
  Option A  — temperature=0.1,  compact,        + synthesis instruction
  Option B  — temperature=0.3,  compact,        current prompt
  Option C  — temperature=0.1,  tree_summarize, current prompt
  All       — temperature=0.3,  tree_summarize, + synthesis instruction

Run:
    uv run python examples/rag_synthesis_comparison.py
"""

from __future__ import annotations

import sys
import io
import os
import textwrap
import time

# Force UTF-8 output on Windows (Thai characters)
if hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# ── query used across all configs ─────────────────────────────────────────────
QUERY = "ลูกค้าที่มีหนี้ค้างชำระควรทำอย่างไรเพื่อปรับปรุงประวัติเครดิต"

# ── prompt templates ──────────────────────────────────────────────────────────
from llama_index.core.prompts import PromptTemplate

PROMPT_CURRENT = PromptTemplate(
    """You are a retrieval QA assistant for Thai home loan and refinance documents.
Use only the retrieved context. Never use outside knowledge.

Rules:
1) Answer in the same language as the user's question.
2) The answer must be at least 2 sentences, unless context is insufficient.
3) Never use the phrase "According to the document".
4) If context is insufficient, output exactly: "ไม่พบข้อมูลในเอกสารที่มีอยู่"
5) If you cite policy conditions, fees, rates, periods, or numeric values, include this format in the answer: "แหล่งข้อมูล: <doc title>".
6) Do not guess numbers, fees, rates, dates, or eligibility conditions.
7) Do not copy blank form placeholders (for example underscores or empty template fields).

Context:
---------------------
{context_str}
---------------------

Question: {query_str}
Answer:"""
)

PROMPT_WITH_SYNTHESIS = PromptTemplate(
    """You are a helpful advisor for Thai home loan and refinance customers.
Use the retrieved context below as your knowledge base.

Rules:
1) Answer in the same language as the user's question.
2) Synthesize the information in your own words — do not copy sentences verbatim from the context.
3) Rephrase, summarize, and organize the advice into a coherent response.
4) If you cite policy conditions, fees, rates, or numeric values, mention the source document.
5) Do not guess numbers, fees, rates, dates, or eligibility conditions not in the context.
6) If context is insufficient, output exactly: "ไม่พบข้อมูลในเอกสารที่มีอยู่"
7) Use a friendly, advisory tone — imagine explaining to the customer directly.

Context:
---------------------
{context_str}
---------------------

Question: {query_str}
Answer:"""
)

REFINE_TEMPLATE = PromptTemplate(
    """Original question: {query_str}
Current answer: {existing_answer}

Additional context:
---------------------
{context_msg}
---------------------

Refine only if the new context clearly improves correctness.
Keep the same language as the question.
Never invent missing details.
Refined answer:"""
)


# ── index loader (mirrors rag_bridge.py) ──────────────────────────────────────
def load_index():
    import chromadb
    from llama_index.core import VectorStoreIndex
    from llama_index.core.settings import Settings
    from llama_index.embeddings.huggingface import HuggingFaceEmbedding
    from llama_index.vector_stores.chroma import ChromaVectorStore
    from config.settings import settings as cfg

    print("Loading embedding model …")
    Settings.embed_model = HuggingFaceEmbedding(
        model_name=cfg.EMBEDDING_MODEL,
        embed_batch_size=32,
    )
    client = chromadb.PersistentClient(path=cfg.CHROMA_PERSIST_DIR)
    collection = client.get_collection(cfg.CHROMA_COLLECTION)
    vector_store = ChromaVectorStore(chroma_collection=collection)
    index = VectorStoreIndex.from_vector_store(vector_store)
    print(f"Index loaded — collection: {cfg.CHROMA_COLLECTION}\n")
    return index


# ── build one RetrieverQueryEngine with custom params ─────────────────────────
def build_engine(index, llm, prompt: PromptTemplate, response_mode: str):
    from llama_index.core.retrievers import VectorIndexRetriever
    from llama_index.core.query_engine import RetrieverQueryEngine
    from llama_index.core.response_synthesizers import get_response_synthesizer
    from config.settings import settings as cfg

    retriever = VectorIndexRetriever(
        index=index,
        similarity_top_k=cfg.SIMILARITY_TOP_K,
    )
    synthesizer = get_response_synthesizer(
        response_mode=response_mode,
        llm=llm,
        text_qa_template=prompt,
        refine_template=REFINE_TEMPLATE,
    )
    return RetrieverQueryEngine(
        retriever=retriever,
        response_synthesizer=synthesizer,
    )


# ── run one config and return (answer, elapsed_sec) ───────────────────────────
def run_config(index, temperature: float, prompt: PromptTemplate, response_mode: str) -> tuple[str, float]:
    from llama_index.llms.ollama import Ollama
    from config.settings import settings as cfg

    llm = Ollama(
        model=cfg.OLLAMA_MODEL,
        base_url=cfg.OLLAMA_BASE_URL,
        temperature=temperature,
        request_timeout=180.0,
    )
    engine = build_engine(index, llm, prompt, response_mode)

    t0 = time.perf_counter()
    response = engine.query(QUERY)
    elapsed = time.perf_counter() - t0

    return str(response).strip(), round(elapsed, 1)


# ── pretty print helper ───────────────────────────────────────────────────────
SEP = "=" * 72

def print_result(label: str, description: str, answer: str, elapsed: float):
    print(SEP)
    print(f"  {label}")
    print(f"  {description}")
    print(f"  Time: {elapsed}s")
    print(SEP)
    wrapped = textwrap.fill(answer, width=70, subsequent_indent="  ")
    print(wrapped)
    print()


# ── main ──────────────────────────────────────────────────────────────────────
def main():
    print(SEP)
    print("  RAG Synthesis Comparison")
    print(f"  Query: {QUERY}")
    print(SEP)
    print()

    try:
        index = load_index()
    except Exception as exc:
        print(f"[ERROR] Could not load index: {exc}")
        print("Make sure Chroma is populated (uv run python -m src.ingest) and Ollama is running.")
        sys.exit(1)

    configs = [
        {
            "label": "BASELINE",
            "desc": "temperature=0.1 | compact | current prompt (control group)",
            "temperature": 0.1,
            "prompt": PROMPT_CURRENT,
            "response_mode": "compact",
        },
        {
            "label": "OPTION A — Synthesis Prompt",
            "desc": "temperature=0.1 | compact | synthesis instruction added",
            "temperature": 0.1,
            "prompt": PROMPT_WITH_SYNTHESIS,
            "response_mode": "compact",
        },
        {
            "label": "OPTION B — Higher Temperature",
            "desc": "temperature=0.3 | compact | current prompt",
            "temperature": 0.3,
            "prompt": PROMPT_CURRENT,
            "response_mode": "compact",
        },
        {
            "label": "OPTION C — Tree Summarize",
            "desc": "temperature=0.1 | tree_summarize | current prompt",
            "temperature": 0.1,
            "prompt": PROMPT_CURRENT,
            "response_mode": "tree_summarize",
        },
        {
            "label": "ALL COMBINED",
            "desc": "temperature=0.3 | tree_summarize | synthesis prompt",
            "temperature": 0.3,
            "prompt": PROMPT_WITH_SYNTHESIS,
            "response_mode": "tree_summarize",
        },
    ]

    results = []
    for i, cfg_item in enumerate(configs, 1):
        print(f"[{i}/{len(configs)}] Running {cfg_item['label']} …")
        try:
            answer, elapsed = run_config(
                index,
                cfg_item["temperature"],
                cfg_item["prompt"],
                cfg_item["response_mode"],
            )
        except Exception as exc:
            answer = f"[ERROR] {exc}"
            elapsed = 0.0
        results.append((cfg_item["label"], cfg_item["desc"], answer, elapsed))
        print(f"       Done ({elapsed}s)\n")

    print("\n")
    print(SEP)
    print("  RESULTS COMPARISON")
    print(SEP)
    print()

    for label, desc, answer, elapsed in results:
        print_result(label, desc, answer, elapsed)

    # ── scoring rubric ───────────────────────────────────────────────────────
    print(SEP)
    print("  EVALUATION RUBRIC (manual review)")
    print(SEP)
    rubric = [
        ("Verbatim copy", "Does the answer copy sentences word-for-word from context?"),
        ("Own words",     "Does it rephrase/summarize instead of copying?"),
        ("Coherence",     "Reads as a unified response, not a list of fragments?"),
        ("Accuracy",      "No hallucinated numbers/fees/conditions?"),
        ("Completeness",  "Covers the key advice the customer needs?"),
        ("Tone",          "Friendly and advisory vs. clinical/document-like?"),
    ]
    for dimension, question in rubric:
        print(f"  {dimension:14s}  {question}")
    print()
    print("Score each config 1-5 per dimension to find the best option.")
    print()


if __name__ == "__main__":
    main()
