"""StatCache — derive-once values keyed on their source files' stats.

The repo's one caching idiom for "recompute only when the files changed":
an entry is valid while every source file's (st_mtime_ns, st_size) matches,
so atomic-rename writers (jsonl.atomic_write, store.save_embedding_matrix)
invalidate entries naturally. Values are shared across callers — treat them
as read-only; a mutating caller must bypass the cache and read fresh.

Unbounded by design: users key per (video, model) with small values. Give a
hot path its own instance rather than sharing one namespace.
"""

from __future__ import annotations

import os
import threading
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import TypeVar

T = TypeVar("T")


class StatCache:
    def __init__(self) -> None:
        self._entries: dict = {}
        self._lock = threading.Lock()

    def get(self, key, paths: Sequence[Path], compute: Callable[[], T]) -> T:
        """The cached value for ``key``, recomputed when any path's stat moved.

        ``compute`` runs outside the lock (it may be slow — a parse, a model
        pass); concurrent misses may compute twice, last write wins. A missing
        path raises FileNotFoundError to the caller, unchanged.
        """
        stats = tuple((s.st_mtime_ns, s.st_size) for s in (os.stat(p) for p in paths))
        with self._lock:
            hit = self._entries.get(key)
            if hit is not None and hit[0] == stats:
                return hit[1]
        value = compute()
        with self._lock:
            self._entries[key] = (stats, value)
        return value
