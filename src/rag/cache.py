"""Thread-safe LRU cache with per-entry TTL for RAG query results.

Uses only stdlib — no external dependencies.

Usage:
    from src.rag.cache import get_cache

    cache = get_cache()
    result = cache.get(question, top_k=5)
    if result is None:
        result = expensive_rag_call(question)
        cache.set(question, result, top_k=5)
"""
from __future__ import annotations

import threading
import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Optional

_DEFAULT_MAX_SIZE = 256
_DEFAULT_TTL_SECONDS = 3600.0  # 1 hour


@dataclass
class _CacheStats:
    hits: int = 0
    misses: int = 0
    evictions: int = 0

    @property
    def total(self) -> int:
        return self.hits + self.misses

    @property
    def hit_rate(self) -> float:
        return round(self.hits / self.total, 4) if self.total > 0 else 0.0


class QueryCache:
    """Thread-safe LRU cache with per-entry TTL.

    - **LRU eviction**: least-recently-used entry is dropped when `max_size` is exceeded.
    - **TTL expiry**: entries are considered stale after `ttl_seconds` and treated as misses.
    - **Thread-safe**: a single lock guards all state mutations.
    """

    def __init__(self, max_size: int = _DEFAULT_MAX_SIZE, ttl_seconds: float = _DEFAULT_TTL_SECONDS) -> None:
        self.max_size = max_size
        self.ttl = ttl_seconds
        self._store: OrderedDict[str, tuple[Any, float]] = OrderedDict()
        self._lock = threading.Lock()
        self._stats = _CacheStats()

    # ── Public API ────────────────────────────────────────────────────────────

    def get(self, question: str, top_k: Optional[int] = None) -> Optional[Any]:
        """Return cached result or None on miss / expiry."""
        key = self._make_key(question, top_k)
        with self._lock:
            if key not in self._store:
                self._stats.misses += 1
                return None
            value, expires_at = self._store[key]
            if time.monotonic() > expires_at:
                del self._store[key]
                self._stats.misses += 1
                return None
            self._store.move_to_end(key)  # mark as recently used
            self._stats.hits += 1
            return value

    def set(self, question: str, value: Any, top_k: Optional[int] = None) -> None:
        """Store a result. Evicts LRU entry if over capacity."""
        key = self._make_key(question, top_k)
        expires_at = time.monotonic() + self.ttl
        with self._lock:
            if key in self._store:
                self._store.move_to_end(key)
            self._store[key] = (value, expires_at)
            while len(self._store) > self.max_size:
                self._store.popitem(last=False)
                self._stats.evictions += 1

    def clear(self) -> int:
        """Remove all entries. Returns number of entries cleared."""
        with self._lock:
            count = len(self._store)
            self._store.clear()
            self._stats = _CacheStats()
            return count

    def stats(self) -> dict:
        """Return current cache statistics (thread-safe snapshot)."""
        with self._lock:
            now = time.monotonic()
            expired = sum(1 for _, (_, exp) in self._store.items() if now > exp)
            return {
                "size": len(self._store),
                "max_size": self.max_size,
                "ttl_seconds": self.ttl,
                "expired_entries": expired,
                "hits": self._stats.hits,
                "misses": self._stats.misses,
                "evictions": self._stats.evictions,
                "hit_rate": self._stats.hit_rate,
            }

    # ── Internals ─────────────────────────────────────────────────────────────

    @staticmethod
    def _make_key(question: str, top_k: Optional[int]) -> str:
        normalized = " ".join(question.lower().split())
        return f"{normalized}|{top_k}"


# ── Global singleton ──────────────────────────────────────────────────────────

_cache = QueryCache(max_size=_DEFAULT_MAX_SIZE, ttl_seconds=_DEFAULT_TTL_SECONDS)


def get_cache() -> QueryCache:
    """Return the global RAG query cache singleton."""
    return _cache
