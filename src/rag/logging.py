"""Structured retrieval logging utilities."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Union

logger = logging.getLogger(__name__)

DEFAULT_LOG_PATH = Path("logs/retrieval_logs.jsonl")
DEFAULT_DEBUG_LOG_PATH = Path("logs/rag_debug.jsonl")


def log_retrieval_event(
    event: Dict[str, Any],
    log_path: Union[str, Path] = DEFAULT_LOG_PATH,
) -> None:
    """Append one retrieval event as JSONL."""
    target = Path(log_path)
    target.parent.mkdir(parents=True, exist_ok=True)

    payload = dict(event)
    payload.setdefault("timestamp", datetime.now(timezone.utc).isoformat())

    try:
        with target.open("a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")
    except Exception as exc:  # pragma: no cover - IO best effort
        logger.warning("Failed to write retrieval log: %s", exc)


def log_rag_debug_event(
    event: Dict[str, Any],
    log_path: Union[str, Path] = DEFAULT_DEBUG_LOG_PATH,
) -> None:
    """Append one RAG debug event as JSONL."""
    target = Path(log_path)
    target.parent.mkdir(parents=True, exist_ok=True)

    payload = dict(event)
    payload.setdefault("timestamp", datetime.now(timezone.utc).isoformat())

    try:
        with target.open("a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")
    except Exception as exc:  # pragma: no cover - IO best effort
        logger.warning("Failed to write rag debug log: %s", exc)
