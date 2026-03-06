"""Query and chat engines for LlamaIndex."""

import logging
import os
import re
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from llama_index.core import VectorStoreIndex
from llama_index.core.chat_engine import CondenseQuestionChatEngine, SimpleChatEngine
from llama_index.core.postprocessor import SimilarityPostprocessor
from llama_index.core.postprocessor.types import BaseNodePostprocessor
from llama_index.core.prompts import PromptTemplate
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.response_synthesizers import get_response_synthesizer
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.schema import NodeWithScore, QueryBundle
from llama_index.core.settings import Settings
from llama_index.core.vector_stores.types import MetadataFilters
from llama_index.llms.ollama import Ollama

from config.settings import settings
from src.document_parser import CLEANING_VERSION
from src.rag.logging import log_rag_debug_event, log_retrieval_event
from src.rag.router import build_metadata_filters, metadata_matches_route, route_query
from src.rag.validator import (
    CLOSE_ACCOUNT_CLARIFICATION_MESSAGE,
    NO_ANSWER_MESSAGE,
    ROUTE_MUST_HAVE,
    needs_close_account_clarification,
    validate_nodes,
)

logger = logging.getLogger(__name__)

TEXT_QA_TEMPLATE = PromptTemplate(
    """You are a retrieval QA assistant for Thai home loan and refinance documents.
Use only the retrieved context. Never use outside knowledge.

Rules:
1) Answer in the same language as the user's question.
2) The answer must be at least 2 sentences, unless context is insufficient.
3) Never use the phrase \"According to the document\".
4) If context is insufficient, output exactly: \"ไม่พบข้อมูลในเอกสารที่มีอยู่\"
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


def _safe_score(node: Any) -> float:
    """Convert node score to comparable float value."""
    score = getattr(node, "score", None)
    if score is None:
        return float("-inf")
    try:
        return float(score)
    except (TypeError, ValueError):
        return float("-inf")


def _extract_node_text(node: Any) -> str:
    """Extract node text from NodeWithScore-like objects."""
    text = getattr(node, "text", None)
    if isinstance(text, str):
        return text

    inner_node = getattr(node, "node", None)
    if inner_node is not None:
        inner_text = getattr(inner_node, "text", None)
        if isinstance(inner_text, str):
            return inner_text
        if hasattr(inner_node, "get_content"):
            try:
                return inner_node.get_content()
            except Exception:  # pragma: no cover - defensive
                return ""

    return ""


def _extract_node_metadata(node: Any) -> Dict[str, Any]:
    """Extract metadata from NodeWithScore-like objects."""
    metadata = getattr(node, "metadata", None)
    if isinstance(metadata, dict):
        return metadata

    inner_node = getattr(node, "node", None)
    if inner_node is not None:
        inner_metadata = getattr(inner_node, "metadata", None)
        if isinstance(inner_metadata, dict):
            return inner_metadata

    return {}


def _nodes_to_log_records(nodes: List[Any]) -> List[Dict[str, Any]]:
    """Convert nodes to structured logging entries."""
    records = []
    for node in nodes:
        metadata = _extract_node_metadata(node)
        records.append(
            {
                "doc_title": metadata.get("title", metadata.get("file_name", "Unknown")),
                "score": _safe_score(node),
            }
        )
    return records


def _nodes_to_debug_records(nodes: List[Any], limit: int = 25) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    for node in nodes[:limit]:
        metadata = _extract_node_metadata(node)
        records.append(
            {
                "title": metadata.get("title", metadata.get("file_name", "Unknown")),
                "category": metadata.get("category", "Uncategorized"),
                "doc_kind": metadata.get("doc_kind", "unknown"),
                "score": _safe_score(node),
            }
        )
    return records


def _env_flag(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _route_cutoff(router_label: str, configured_cutoff: float) -> float:
    route_caps = {
        "interest_structure": 0.20,
        "fee_structure": 0.23,
        "refinance": 0.25,
        "hardship_support": 0.25,
        "policy_requirement": 0.28,
        "general_info": 0.30,
    }
    return min(configured_cutoff, route_caps.get(router_label, 0.30))


def _apply_similarity_cutoff(nodes: List[Any], cutoff: float) -> List[Any]:
    kept = [node for node in nodes if _safe_score(node) >= cutoff]
    if kept:
        return kept
    # Prevent aggressive cutoffs from dropping all evidence for table-heavy Thai docs.
    return nodes


def _final_context_char_count(nodes: List[Any]) -> int:
    return sum(len(_extract_node_text(node)) for node in nodes)


def _node_has_fingerprint(node: Any) -> bool:
    metadata = _extract_node_metadata(node)
    if str(metadata.get("cleaning_version", "")).strip() == CLEANING_VERSION:
        return True
    text = _extract_node_text(node)
    return f"CLEANING_VERSION: {CLEANING_VERSION}" in text


def _route_bonus_terms(router_label: str) -> List[str]:
    return {
        "policy_requirement": ["คุณสมบัติ", "เอกสาร", "รายได้ขั้นต่ำ", "สัญชาติ", "อาชีพ", "เงื่อนไข"],
        "interest_structure": ["ดอกเบี้ย", "fixed", "floating", "mrr", "%", "ปีแรก"],
        "fee_structure": ["ค่าธรรมเนียม", "ค่าปรับ", "จดจำนอง", "ปิดบัญชี", "ค่าใช้จ่าย"],
        "refinance": ["รีไฟแนนซ์", "บ้านแลกเงิน", "mortgage power"],
        "hardship_support": ["ผ่อนไม่ไหว", "ปรับโครงสร้างหนี้", "พักชำระ", "มาตรการ", "โควิด", "น้ำท่วม"],
    }.get(router_label, [])


STRICT_ROUTE_ALLOWED_CATEGORIES = {
    "policy_requirement": {"policy_requirement"},
    "interest_structure": {"interest_structure"},
    "fee_structure": {"fee_structure"},
    "refinance": {"refinance"},
    "hardship_support": {"hardship_support", "consumer_guideline"},
}

STRICT_ROUTE_MUST_HAVE = ROUTE_MUST_HAVE

GLOBAL_BLOCKLIST = (
    "ndid",
    "พร้อมเพย์",
    "กรมสรรพากร",
    "ภาษี",
    "bizchannel@cimb",
    "cimb thai app",
    "cimb thai connect",
)

ROUTE_BLOCKLIST = {
    "refinance": GLOBAL_BLOCKLIST,
    "interest_structure": GLOBAL_BLOCKLIST,
    "fee_structure": GLOBAL_BLOCKLIST,
    "policy_requirement": GLOBAL_BLOCKLIST,
}

EXPLICIT_BLOCK_ALLOW_TERMS = ("ndid", "พร้อมเพย์", "กรมสรรพากร", "ภาษี", "tax")


def _question_allows_blocked_terms(question: str) -> bool:
    q = (question or "").lower()
    return any(term in q for term in EXPLICIT_BLOCK_ALLOW_TERMS)


def _node_match_text(node: Any) -> str:
    metadata = _extract_node_metadata(node)
    joined_meta = " ".join(
        str(metadata.get(key, ""))
        for key in ("title", "file_name", "category", "doc_kind", "topic", "topic_tags")
    )
    return f"{_extract_node_text(node)} {joined_meta}".lower()


def _strict_route_filter(
    question: str,
    nodes: List[Any],
    router_label: str,
) -> tuple[List[Any], int]:
    """Stage-2 strict filter: route category + must-have + domain blocklist."""
    if not nodes:
        return [], 0

    allow_blocked = _question_allows_blocked_terms(question)
    allowed_categories = STRICT_ROUTE_ALLOWED_CATEGORIES.get(router_label, set())
    must_have = STRICT_ROUTE_MUST_HAVE.get(router_label, ())
    route_blocklist = ROUTE_BLOCKLIST.get(router_label, ())

    filtered: List[Any] = []
    blocked_count = 0

    for node in nodes:
        metadata = _extract_node_metadata(node)
        category = str(metadata.get("category", "")).lower()
        text = _node_match_text(node)

        if (
            router_label != "general_info"
            and allowed_categories
            and category
            and category not in allowed_categories
            and not metadata_matches_route(metadata, router_label)
        ):
            blocked_count += 1
            continue

        if not allow_blocked and any(token in text for token in GLOBAL_BLOCKLIST):
            blocked_count += 1
            continue

        if must_have and not any(token in text for token in must_have):
            blocked_count += 1
            continue

        if route_blocklist and not allow_blocked and any(token in text for token in route_blocklist):
            blocked_count += 1
            continue

        filtered.append(node)

    return sorted(filtered, key=_safe_score, reverse=True), blocked_count


def _rerank_nodes(question: str, nodes: List[Any], router_label: str) -> List[Any]:
    terms = _route_bonus_terms(router_label)
    if not terms:
        return sorted(nodes, key=_safe_score, reverse=True)

    q = (question or "").lower()
    active_terms = [term for term in terms if term.lower() in q] or terms

    def _rank(node: Any) -> float:
        text = _extract_node_text(node).lower()
        bonus_hits = sum(1 for term in active_terms if term.lower() in text)
        bonus = min(0.60, bonus_hits * 0.20)
        return _safe_score(node) + bonus

    return sorted(nodes, key=_rank, reverse=True)


def _extract_policy_hint(nodes: List[Any]) -> Optional[str]:
    policy_terms = ("คุณสมบัติ", "เอกสาร", "รายได้ขั้นต่ำ", "สัญชาติ", "อาชีพ", "เงื่อนไข")
    for node in nodes:
        text = _extract_node_text(node)
        metadata = _extract_node_metadata(node)
        title = metadata.get("title", metadata.get("file_name", "Unknown"))
        for raw_line in text.splitlines():
            line = raw_line.strip()
            if len(line) < 12:
                continue
            lowered = line.lower()
            if any(term in lowered for term in policy_terms):
                line = re.sub(r"[_\s]{2,}", " ", line).strip(" .")
                if not line:
                    continue
                return (
                    f"{line}. แหล่งข้อมูล: {title}. "
                    "กรุณาตรวจสอบเงื่อนไขล่าสุดกับธนาคารอีกครั้ง."
                )
    return None


def _contains_thai(text: str) -> bool:
    return bool(re.search(r"[\u0E00-\u0E7F]", text))


def _estimate_sentence_count(text: str) -> int:
    """
    Estimate sentence-like units for Thai/English answers.

    We count both punctuation-separated clauses and newline/bullet rows
    to avoid undercounting list-style answers (common for rate sheets).
    """
    if not text or not text.strip():
        return 0

    normalized = text.replace("\r", "\n")

    # Avoid splitting decimal numbers such as 6.59.
    raw_parts = re.split(r"\n+|(?<!\d)[.!?;。！？]+(?!\d)", normalized)
    parts = []
    for raw in raw_parts:
        cleaned = re.sub(r"^[\-\*\u2022\s]+", "", raw.strip())
        # Keep only meaningful chunks that have Thai/English letters or digits.
        if cleaned and re.search(r"[\u0E00-\u0E7FA-Za-z0-9]", cleaned):
            parts.append(cleaned)

    return len(parts) if parts else 1


def _normalize_answer_text(
    answer_text: str,
    question: str,
    source_nodes: List[Any],
) -> str:
    """
    Enforce response policy after synthesis without changing API contract.
    """
    text = (answer_text or "").strip()
    if not text or text.lower() == "empty response":
        return NO_ANSWER_MESSAGE

    if text == NO_ANSWER_MESSAGE:
        return text

    text = re.sub(
        r"(?i)\baccording to (the )?(document|provided information)\b[:,]?\s*",
        "",
        text,
    ).strip()

    # Some models append fallback text even when valid evidence exists.
    if NO_ANSWER_MESSAGE in text and text != NO_ANSWER_MESSAGE:
        text = text.replace(NO_ANSWER_MESSAGE, " ").strip(" .\n\t")
        text = re.sub(r"\s{2,}", " ", text).strip()
        if not text:
            return NO_ANSWER_MESSAGE

    # Remove weak "no information" fragments when actual sources are present.
    if source_nodes and "ไม่พบข้อมูล" in text and text != NO_ANSWER_MESSAGE:
        text = re.sub(r"\(?\s*ไม่พบข้อมูล[^)\n]*\)?", " ", text).strip(" .\n\t")
        text = re.sub(r"\s{2,}", " ", text).strip()
        if not text:
            return NO_ANSWER_MESSAGE

    primary_title = None
    if source_nodes:
        metadata = _extract_node_metadata(source_nodes[0])
        primary_title = metadata.get("title", metadata.get("file_name", "Unknown"))

    lower_text = text.lower()
    cites_numeric = bool(re.search(r"\d|[%฿]", text))
    cites_policy = any(
        keyword in lower_text
        for keyword in ("policy", "requirement", "eligibility", "เงื่อนไข", "คุณสมบัติ")
    )
    needs_doc_title = (cites_numeric or cites_policy) and primary_title

    if needs_doc_title and primary_title not in text:
        if _contains_thai(question):
            text = f"{text} แหล่งข้อมูล: {primary_title}"
        else:
            text = f"{text} (Source document: {primary_title})"

    if source_nodes and _estimate_sentence_count(text) < 2:
        if _contains_thai(question):
            if primary_title:
                if f"แหล่งข้อมูล: {primary_title}" not in text:
                    text = f"{text} แหล่งข้อมูล: {primary_title}."
                else:
                    text = f"{text} กรุณาตรวจสอบเงื่อนไขล่าสุดกับธนาคารอีกครั้ง."
            else:
                text = f"{text} กรุณาตรวจสอบรายละเอียดเพิ่มเติมกับธนาคาร."
        else:
            if primary_title:
                text = f"{text} This is based on {primary_title}."
            else:
                text = f"{text} Please verify details with the bank."

    return text.strip()


def format_source_display(metadata: dict) -> dict:
    """
    Format metadata for display with fallbacks

    Args:
        metadata: Document metadata dictionary

    Returns:
        Formatted dictionary with title, category, source_url, institution, publication_date
        - title: Uses metadata title, falls back to file_name, then "Unknown"
        - category: Uses metadata category, falls back to "Uncategorized"
        - Other fields: Passed through as-is (may be None)
    """
    return {
        "title": metadata.get("title", metadata.get("file_name", "Unknown")),
        "category": metadata.get("category", "Uncategorized"),
        "source_url": metadata.get("source_url"),
        "institution": metadata.get("institution"),
        "publication_date": metadata.get("publication_date"),
        "doc_kind": metadata.get("doc_kind", "unknown"),
    }


class MetadataRoutePostprocessor(BaseNodePostprocessor):
    """Apply route-aware metadata filtering for all vector stores."""

    router_label: str = "general_info"

    @classmethod
    def class_name(cls) -> str:
        return "MetadataRoutePostprocessor"

    def _postprocess_nodes(
        self,
        nodes: List[NodeWithScore],
        query_bundle: Optional[QueryBundle] = None,
    ) -> List[NodeWithScore]:
        if self.router_label == "general_info":
            return nodes

        return [
            node
            for node in nodes
            if metadata_matches_route(_extract_node_metadata(node), self.router_label)
        ]


class RelevanceValidatorPostprocessor(BaseNodePostprocessor):
    """Filter out cross-domain nodes and rerank by score."""

    router_label: str = "general_info"

    @classmethod
    def class_name(cls) -> str:
        return "RelevanceValidatorPostprocessor"

    def _postprocess_nodes(
        self,
        nodes: List[NodeWithScore],
        query_bundle: Optional[QueryBundle] = None,
    ) -> List[NodeWithScore]:
        question = query_bundle.query_str if query_bundle else ""
        validated = validate_nodes(
            question=question,
            nodes=nodes,
            router_label=self.router_label,
        )
        return sorted(validated, key=_safe_score, reverse=True)


def _build_llm():
    """
    LLM factory — single place to swap providers.

    Priority: USE_GEMINI > USE_OLLAMA > OpenAI

    To migrate to Gemini 2/2.5 Flash, set USE_GEMINI=true and GEMINI_API_KEY in .env.
    Required package: pip install llama-index-llms-gemini
    """
    if settings.USE_GEMINI:
        # ------------------------------------------------------------------
        # Gemini path (activate with USE_GEMINI=true in .env)
        # Supports: gemini-2.0-flash, gemini-2.5-flash-preview-04-17
        # ------------------------------------------------------------------
        from llama_index.llms.gemini import Gemini  # noqa: PLC0415

        logger.info("LLM provider: Gemini (model=%s)", settings.GEMINI_MODEL)
        return Gemini(
            model=settings.GEMINI_MODEL,
            api_key=settings.GEMINI_API_KEY,
            temperature=0.1,
        )

    if settings.USE_OLLAMA:
        # ------------------------------------------------------------------
        # Ollama path (default — local model, no API key needed)
        # ------------------------------------------------------------------
        logger.info(
            "LLM provider: Ollama (model=%s, url=%s)",
            settings.OLLAMA_MODEL,
            settings.OLLAMA_BASE_URL,
        )
        return Ollama(
            model=settings.OLLAMA_MODEL,
            base_url=settings.OLLAMA_BASE_URL,
            temperature=0.1,
            request_timeout=120.0,
        )

    # ------------------------------------------------------------------
    # OpenAI path (fallback)
    # ------------------------------------------------------------------
    from llama_index.llms.openai import OpenAI  # noqa: PLC0415

    logger.info("LLM provider: OpenAI (model=%s)", settings.MODEL_NAME)
    return OpenAI(
        model=settings.MODEL_NAME,
        temperature=0.1,
        api_key=settings.OPENAI_API_KEY,
    )


class QueryEngineManager:
    """Manage query and chat engines."""

    def __init__(self, index: VectorStoreIndex):
        self.index = index
        self._bm25_nodes_cache: Optional[List] = None

        self.llm = _build_llm()

        # Set global Settings.llm so response synthesizer uses it
        Settings.llm = self.llm

        self.similarity_top_k = settings.SIMILARITY_TOP_K
        self.similarity_cutoff = settings.SIMILARITY_CUTOFF
        self.response_mode = settings.RESPONSE_MODE

    def _get_nodes_for_bm25(self) -> List:
        """Load all nodes from the vector store for BM25 index (cached after first call)."""
        if self._bm25_nodes_cache is not None:
            return self._bm25_nodes_cache

        from llama_index.core.schema import TextNode

        nodes: List = []

        # Chroma path: fetch via collection.get()
        vector_store = getattr(self.index, "_vector_store", None)
        chroma_collection = getattr(vector_store, "_collection", None)
        if chroma_collection is not None:
            try:
                result = chroma_collection.get(include=["documents", "metadatas"])
                for doc_id, doc_text, metadata in zip(
                    result["ids"], result["documents"], result["metadatas"]
                ):
                    if doc_text:
                        nodes.append(
                            TextNode(text=doc_text, id_=doc_id, metadata=metadata or {})
                        )
                logger.info("BM25: loaded %d nodes from Chroma", len(nodes))
            except Exception as exc:
                logger.warning("BM25: could not load Chroma nodes: %s", exc)
        else:
            # Fallback: SimpleVectorStore / FAISS — try in-memory docstore
            docstore = getattr(self.index, "docstore", None)
            if docstore and hasattr(docstore, "docs"):
                nodes = list(docstore.docs.values())
                logger.info("BM25: loaded %d nodes from docstore", len(nodes))

        self._bm25_nodes_cache = nodes
        return nodes

    def _build_retriever(
        self,
        similarity_top_k: int,
        metadata_filters: Optional[MetadataFilters],
    ):
        """Return a hybrid (BM25 + vector) retriever or fall back to vector-only."""
        retriever_kwargs: Dict[str, Any] = {
            "index": self.index,
            "similarity_top_k": similarity_top_k,
        }
        if metadata_filters is not None and settings.VECTOR_STORE_TYPE == "chroma":
            retriever_kwargs["filters"] = metadata_filters
        elif metadata_filters is not None:
            logger.debug(
                "Skipping retriever-level metadata filters for VECTOR_STORE_TYPE=%s",
                settings.VECTOR_STORE_TYPE,
            )

        vector_retriever = VectorIndexRetriever(**retriever_kwargs)

        if not _env_flag("RAG_HYBRID_SEARCH", default=False):
            return vector_retriever

        try:
            from llama_index.retrievers.bm25 import BM25Retriever
            from llama_index.core.retrievers import QueryFusionRetriever

            bm25_nodes = self._get_nodes_for_bm25()
            if not bm25_nodes:
                logger.warning("BM25: no nodes available, using vector-only retriever")
                return vector_retriever

            bm25_retriever = BM25Retriever.from_defaults(
                nodes=bm25_nodes,
                similarity_top_k=similarity_top_k,
            )
            logger.info(
                "Hybrid search enabled: vector+BM25, top_k=%d, nodes=%d",
                similarity_top_k,
                len(bm25_nodes),
            )
            return QueryFusionRetriever(
                retrievers=[vector_retriever, bm25_retriever],
                similarity_top_k=similarity_top_k,
                num_queries=1,
                mode="reciprocal_rerank",
                use_async=False,
            )
        except ImportError:
            logger.warning(
                "BM25 retriever not installed. "
                "Run: pip install llama-index-retrievers-bm25"
            )
            return vector_retriever
        except Exception as exc:
            logger.warning("BM25 setup failed, using vector-only: %s", exc)
            return vector_retriever

    def create_query_engine(
        self,
        similarity_top_k: Optional[int] = None,
        response_mode: Optional[str] = None,
        use_postprocessor: bool = True,
        router_label: str = "general_info",
        metadata_filters: Optional[MetadataFilters] = None,
    ) -> RetrieverQueryEngine:
        """
        Create a query engine

        Args:
            similarity_top_k: Number of similar documents to retrieve
            response_mode: Response synthesis mode
            use_postprocessor: Whether to use similarity postprocessor
            router_label: Routed domain label for query
            metadata_filters: Optional vector store metadata filters

        Returns:
            Configured query engine
        """
        similarity_top_k = similarity_top_k or self.similarity_top_k
        response_mode = response_mode or self.response_mode

        logger.info(
            "Creating query engine with top_k=%s, mode=%s, route=%s",
            similarity_top_k,
            response_mode,
            router_label,
        )

        retriever = self._build_retriever(similarity_top_k, metadata_filters)

        # Create response synthesizer
        response_synthesizer = get_response_synthesizer(
            response_mode=response_mode,
            llm=self.llm,
            text_qa_template=TEXT_QA_TEMPLATE,
            refine_template=REFINE_TEMPLATE,
        )

        # Postprocessors execute before synthesis.
        postprocessors: List[BaseNodePostprocessor] = [
            MetadataRoutePostprocessor(router_label=router_label),
            RelevanceValidatorPostprocessor(router_label=router_label),
        ]
        disable_sim_cutoff = _env_flag("RAG_DISABLE_SIM_CUTOFF", default=False)
        if use_postprocessor and not disable_sim_cutoff:
            adaptive_cutoff = _route_cutoff(router_label, self.similarity_cutoff)
            postprocessors.append(SimilarityPostprocessor(similarity_cutoff=adaptive_cutoff))
        elif use_postprocessor and disable_sim_cutoff:
            logger.info("RAG_DISABLE_SIM_CUTOFF=true, skipping SimilarityPostprocessor")

        # Create query engine
        query_engine = RetrieverQueryEngine(
            retriever=retriever,
            response_synthesizer=response_synthesizer,
            node_postprocessors=postprocessors,
        )

        return query_engine

    def create_chat_engine(
        self,
        chat_mode: str = "condense_question",
        similarity_top_k: Optional[int] = None,
        verbose: bool = True,
    ):
        """
        Create a chat engine

        Args:
            chat_mode: Chat mode ("condense_question", "simple", "context")
            similarity_top_k: Number of similar documents to retrieve
            verbose: Whether to show verbose output

        Returns:
            Configured chat engine
        """
        similarity_top_k = similarity_top_k or self.similarity_top_k

        logger.info("Creating chat engine with mode=%s", chat_mode)

        if chat_mode == "condense_question":
            # Create query engine for chat
            query_engine = self.create_query_engine(similarity_top_k=similarity_top_k)

            chat_engine = CondenseQuestionChatEngine.from_defaults(
                query_engine=query_engine, llm=self.llm, verbose=verbose
            )
        elif chat_mode == "simple":
            chat_engine = SimpleChatEngine.from_defaults(llm=self.llm, verbose=verbose)
        else:
            # Default to condense_question
            chat_engine = self.create_chat_engine(
                chat_mode="condense_question",
                similarity_top_k=similarity_top_k,
                verbose=verbose,
            )

        return chat_engine

    def query(
        self,
        question: str,
        similarity_top_k: Optional[int] = None,
        response_mode: Optional[str] = None,
        include_sources: bool = True,
    ) -> Dict[str, Any]:
        """
        Query the index

        Args:
            question: Query question
            similarity_top_k: Number of similar documents to retrieve
            response_mode: Response synthesis mode
            include_sources: Whether to include source information

        Returns:
            Dictionary with response and metadata
        """
        router_label = route_query(question)
        metadata_filters = build_metadata_filters(router_label)
        requested_top_k = similarity_top_k or self.similarity_top_k
        retrieval_top_k = 30
        final_top_k = 8
        disable_sim_cutoff = _env_flag("RAG_DISABLE_SIM_CUTOFF", default=False)
        route_cutoff = _route_cutoff(router_label, self.similarity_cutoff)

        logger.info("Querying: %s (route=%s)", question, router_label)

        query_engine = self.create_query_engine(
            similarity_top_k=retrieval_top_k,
            response_mode=response_mode,
            use_postprocessor=False,
            router_label=router_label,
            metadata_filters=metadata_filters,
        )

        retrieved_nodes: List[Any] = []
        route_filtered_nodes: List[Any] = []
        strict_filtered_nodes: List[Any] = []
        validated_nodes: List[Any] = []
        final_nodes: List[Any] = []
        response: Any = None
        answer_text = NO_ANSWER_MESSAGE
        source_nodes: List[Any] = []
        manual_pipeline_used = False
        domain_drift_blocked_count = 0

        retriever = getattr(query_engine, "retriever", None)
        synthesizer = getattr(query_engine, "response_synthesizer", None)

        try:
            if retriever is not None and hasattr(retriever, "retrieve"):
                maybe_nodes = retriever.retrieve(question)
                if isinstance(maybe_nodes, (list, tuple)):
                    retrieved_nodes = list(maybe_nodes)
        except Exception as exc:  # pragma: no cover - logging best effort
            logger.debug("Unable to capture pre-filter retrieval nodes: %s", exc)

        if retrieved_nodes:
            if router_label == "general_info":
                route_filtered_nodes = list(retrieved_nodes)
            else:
                route_filtered_nodes = [
                    node
                    for node in retrieved_nodes
                    if metadata_matches_route(_extract_node_metadata(node), router_label)
                ]
                if not route_filtered_nodes:
                    route_filtered_nodes = list(retrieved_nodes)

            strict_filtered_nodes, blocked_stage2 = _strict_route_filter(
                question=question,
                nodes=route_filtered_nodes,
                router_label=router_label,
            )
            domain_drift_blocked_count += blocked_stage2

            validated_nodes = validate_nodes(
                question=question,
                nodes=strict_filtered_nodes,
                router_label=router_label,
            )
            validated_nodes = _rerank_nodes(question, validated_nodes, router_label)

            if len(validated_nodes) < 2:
                answer_text = NO_ANSWER_MESSAGE
                response = NO_ANSWER_MESSAGE
                source_nodes = []
                manual_pipeline_used = True

            if not manual_pipeline_used and needs_close_account_clarification(question, validated_nodes):
                answer_text = CLOSE_ACCOUNT_CLARIFICATION_MESSAGE
                source_nodes = validated_nodes[:final_top_k]
                response = answer_text
                manual_pipeline_used = True
            elif not manual_pipeline_used:
                if validated_nodes:
                    if disable_sim_cutoff:
                        final_nodes = list(validated_nodes)
                    else:
                        final_nodes = _apply_similarity_cutoff(validated_nodes, route_cutoff)

                    final_nodes = _rerank_nodes(question, final_nodes, router_label)[:final_top_k]

                if final_nodes and synthesizer is not None and hasattr(synthesizer, "synthesize"):
                    response = synthesizer.synthesize(question, final_nodes)
                    source_nodes = getattr(response, "source_nodes", []) or final_nodes
                    answer_text = _normalize_answer_text(str(response), question, source_nodes)
                    manual_pipeline_used = True

        if not manual_pipeline_used:
            if validated_nodes == [] and retrieved_nodes:
                response = NO_ANSWER_MESSAGE
                answer_text = NO_ANSWER_MESSAGE
            else:
                response = query_engine.query(question)
                source_nodes = getattr(response, "source_nodes", []) or []
                answer_text = _normalize_answer_text(str(response), question, source_nodes)

                if (
                    answer_text == NO_ANSWER_MESSAGE
                    and final_nodes
                    and synthesizer is not None
                    and hasattr(synthesizer, "synthesize")
                ):
                    retry_response = synthesizer.synthesize(question, final_nodes)
                    retry_sources = getattr(retry_response, "source_nodes", []) or final_nodes
                    retry_answer = _normalize_answer_text(
                        str(retry_response),
                        question,
                        retry_sources,
                    )
                    if retry_answer != NO_ANSWER_MESSAGE:
                        response = retry_response
                        source_nodes = retry_sources
                        answer_text = retry_answer

        if answer_text != CLOSE_ACCOUNT_CLARIFICATION_MESSAGE and not source_nodes and validated_nodes:
            # Use validated nodes for source visibility when synthesis path did not attach sources.
            source_nodes = validated_nodes[:final_top_k]

        if (
            router_label == "policy_requirement"
            and answer_text == NO_ANSWER_MESSAGE
            and final_nodes
        ):
            policy_hint = _extract_policy_hint(final_nodes)
            if policy_hint:
                answer_text = policy_hint
                response = policy_hint
                source_nodes = final_nodes[:final_top_k]

        result = {
            "question": question,
            "answer": answer_text,
            "response": response,
            "router_label": router_label,
            "domain_drift_blocked_count": domain_drift_blocked_count,
            "retrieved_node_count": len(retrieved_nodes),
            "validated_node_count": len(validated_nodes),
        }

        if include_sources and source_nodes:
            sources = []
            for node in source_nodes:
                text = _extract_node_text(node)
                metadata = _extract_node_metadata(node)
                source_info = {
                    "content": text[:200] + "..." if len(text) > 200 else text,
                    "metadata": format_source_display(metadata),
                    "score": getattr(node, "score", None),
                }
                sources.append(source_info)
            result["sources"] = sources

        final_context_char_count = _final_context_char_count(source_nodes)
        used_fingerprint_found = any(_node_has_fingerprint(node) for node in source_nodes)
        rag_debug_event = {
            "question": question,
            "router_label": router_label,
            "topk_before_filter": _nodes_to_debug_records(retrieved_nodes),
            "topk_after_filter": _nodes_to_debug_records(strict_filtered_nodes or route_filtered_nodes),
            "topk_after_validator": _nodes_to_debug_records(validated_nodes),
            "final_context_char_count": final_context_char_count,
            "answer_preview": answer_text[:240],
            "used_fingerprint_found": used_fingerprint_found,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        log_rag_debug_event(rag_debug_event)

        if _env_flag("RAG_DEBUG", default=False):
            print(
                "[RAG_DEBUG] "
                f"route={router_label} before={len(retrieved_nodes)} "
                f"after_filter={len(route_filtered_nodes)} "
                f"after_validator={len(validated_nodes)} "
                f"final_context_chars={final_context_char_count} "
                f"fingerprint={used_fingerprint_found}"
            )

        log_retrieval_event(
            {
                "question": question,
                "router_label": router_label,
                "retrieved": _nodes_to_log_records(retrieved_nodes),
                "filtered": _nodes_to_log_records(source_nodes),
                "final_answer_length": len(answer_text),
                "is_no_answer": answer_text.strip() == NO_ANSWER_MESSAGE,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
        )

        return result

    def chat(
        self,
        message: str,
        chat_history: Optional[List[Dict[str, str]]] = None,
        chat_mode: str = "condense_question",
        similarity_top_k: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Chat with the index

        Args:
            message: Chat message
            chat_history: Previous chat history
            chat_mode: Chat mode to use
            similarity_top_k: Number of similar documents to retrieve

        Returns:
            Dictionary with response and metadata
        """
        logger.info("Chat message: %s", message)

        chat_engine = self.create_chat_engine(
            chat_mode=chat_mode, similarity_top_k=similarity_top_k
        )

        # If chat history is provided, you might need to handle it differently
        # depending on the chat engine implementation
        response = chat_engine.chat(message)

        result = {
            "message": message,
            "answer": str(response),
            "response": response,
        }

        # Add source information if available
        if hasattr(response, "source_nodes"):
            sources = []
            for node in response.source_nodes:
                text = _extract_node_text(node)
                metadata = _extract_node_metadata(node)
                source_info = {
                    "content": text[:200] + "..." if len(text) > 200 else text,
                    "metadata": format_source_display(metadata),
                    "score": getattr(node, "score", None),
                }
                sources.append(source_info)
            result["sources"] = sources

        return result

    def get_query_suggestions(self, topic: str, num_suggestions: int = 5) -> List[str]:
        """
        Get query suggestions based on a topic

        Args:
            topic: Topic to generate suggestions for
            num_suggestions: Number of suggestions to generate

        Returns:
            List of suggested queries
        """
        prompt = f"""
        Based on the topic "{topic}", generate {num_suggestions} specific and useful questions
        that would be good for querying a document database. Make the questions specific and actionable.

        Format each question on a new line.
        """

        try:
            response = self.llm.complete(prompt)
            suggestions = [line.strip() for line in str(response).split("\n") if line.strip()]
            return suggestions[:num_suggestions]
        except Exception as e:
            logger.error("Error generating suggestions: %s", e)
            return []

    def explain_response(self, response) -> Dict[str, Any]:
        """
        Explain the response with detailed information

        Args:
            response: Query response object

        Returns:
            Dictionary with explanation details
        """
        explanation = {
            "answer": str(response),
            "type": type(response).__name__,
        }

        if hasattr(response, "source_nodes"):
            explanation["num_sources"] = len(response.source_nodes)
            explanation["sources"] = []

            for i, node in enumerate(response.source_nodes):
                text = _extract_node_text(node)
                metadata = _extract_node_metadata(node)
                source_info = {
                    "index": i,
                    "content_preview": text[:100] + "..." if len(text) > 100 else text,
                    "metadata": format_source_display(metadata),
                    "score": getattr(node, "score", None),
                }
                explanation["sources"].append(source_info)

        return explanation
