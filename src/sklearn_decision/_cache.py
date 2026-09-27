"""Answer cache: one row per (model namespace, question spec, state)."""
from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
import time
from collections.abc import Mapping

from ._answers import Answer, answer_from_json, answer_to_json

__all__ = ["AnswerCache", "answer_key", "canon", "question_key", "clear_memory_cache"]

# Process-wide store behind every cache_path=None cache. Keys include the
# model namespace, so estimators can share it safely; clones, refits and
# grid-search candidates reuse answers without writing anything to disk.
_MEMORY: dict[str, tuple[Answer, str]] = {}


def clear_memory_cache() -> None:
    """Drop every answer held by in-memory (cache_path=None) caches."""
    _MEMORY.clear()


def canon(obj) -> str:
    """Canonical JSON: the identity of a state or question spec."""
    return json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":"), default=str)


def question_key(spec: Mapping) -> str:
    """The identity of a question spec, for cache keys.

    Choice options keep their order: answers are stored by position, and
    the order is part of what the model sees, so the same options listed in
    a different order are a different question. (``canon`` alone would sort
    them away and hand one ordering's answers to the other.)
    """
    d = dict(spec)
    if isinstance(d.get("criteria"), Mapping):
        d["criteria"] = [[k, v] for k, v in d["criteria"].items()]
    return canon(d)


def answer_key(namespace: str, qspec_key: str, state_key: str) -> str:
    return hashlib.sha256(f"{namespace}\x1f{qspec_key}\x1f{state_key}".encode()).hexdigest()


class AnswerCache:
    """SQLite-backed when ``path`` is set, otherwise the process-wide
    in-memory store (see :func:`clear_memory_cache`).

    The connection opens lazily and is dropped on pickling, so estimators that
    hold a cache stay picklable and cloneable.
    """

    def __init__(self, path: str | None):
        self.path = path
        self._conn: sqlite3.Connection | None = None
        self._lock = threading.Lock()

    def _connect(self) -> sqlite3.Connection:
        if self._conn is None:
            conn = sqlite3.connect(self.path, check_same_thread=False)
            conn.execute(
                "CREATE TABLE IF NOT EXISTS answers ("
                "key TEXT PRIMARY KEY, answer TEXT NOT NULL, "
                "model_version TEXT, created REAL)"
            )
            conn.commit()
            self._conn = conn
        return self._conn

    def get_many(self, keys: list[str]) -> dict[str, tuple[Answer, str]]:
        if not self.path:
            return {k: _MEMORY[k] for k in keys if k in _MEMORY}
        out: dict[str, tuple[Answer, str]] = {}
        with self._lock:
            conn = self._connect()
            for i in range(0, len(keys), 900):  # sqlite parameter limit
                chunk = keys[i : i + 900]
                rows = conn.execute(
                    f"SELECT key, answer, model_version FROM answers WHERE key IN ({','.join('?' * len(chunk))})",
                    chunk,
                ).fetchall()
                for k, a, v in rows:
                    out[k] = (answer_from_json(json.loads(a)), v)
        return out

    def put_many(self, rows: list[tuple[str, Answer, str]]) -> None:
        if not rows:
            return
        if not self.path:
            for k, a, v in rows:
                _MEMORY[k] = (a, v)
            return
        now = time.time()
        with self._lock:
            conn = self._connect()
            conn.executemany(
                "INSERT OR REPLACE INTO answers VALUES (?, ?, ?, ?)",
                [(k, json.dumps(answer_to_json(a)), v, now) for k, a, v in rows],
            )
            conn.commit()

    def close(self) -> None:
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_conn"] = None
        state.pop("_lock", None)
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._lock = threading.Lock()
