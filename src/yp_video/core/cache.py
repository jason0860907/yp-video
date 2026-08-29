"""StatCache — derive-once values keyed on their source files' stats.

The repo's one caching idiom for "recompute only when the files changed":
an entry is valid while every source file's (st_mtime_ns, st_size) matches,
so atomic-rename writers (jsonl.atomic_write, store.save_embedding_matrix)
invalidate entries naturally. Values are shared across callers — treat them
as read-only; a mutating caller must bypass the cache and read fresh.

Instances are unbounded by default. A cache that owns large derived values can
set ``max_source_bytes``; entries then form an LRU weighted by the total source
file size recorded in their stat keys. Source bytes are deliberately the
weight: they are stable and cheap to obtain, while recursively sizing an
arbitrary Python value is slow and unreliable. The owner calibrates its budget
against resident-memory measurements for that one value type.
"""

from __future__ import annotations

import os
import threading
from collections import OrderedDict
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import TypeVar

T = TypeVar("T")


class StatCache:
    def __init__(self, *, max_source_bytes: int | None = None) -> None:
        if max_source_bytes is not None and max_source_bytes <= 0:
            raise ValueError("max_source_bytes must be positive")
        self._max_source_bytes = max_source_bytes
        self._source_bytes = 0
        self._entries: OrderedDict = OrderedDict()
        self._lock = threading.Lock()

    def get(self, key, paths: Sequence[Path], compute: Callable[[], T]) -> T:
        """The cached value for ``key``, recomputed when any path's stat moved.

        ``compute`` runs outside the lock (it may be slow — a parse, a model
        pass); concurrent misses may compute twice, last write wins. A missing
        path raises FileNotFoundError to the caller, unchanged.
        """
        stats = tuple((s.st_mtime_ns, s.st_size) for s in (os.stat(p) for p in paths))
        source_bytes = sum(size for _mtime, size in stats)
        with self._lock:
            hit = self._entries.get(key)
            if hit is not None and hit[0] == stats:
                self._entries.move_to_end(key)
                return hit[1]

            # Make room before constructing a potentially large value. If
            # eviction waited until after compute(), a miss would hold both
            # old and new object graphs and defeat the memory bound.
            stale = self._entries.pop(key, None)
            if stale is not None:
                self._source_bytes -= stale[2]
            while (
                self._max_source_bytes is not None
                and self._entries
                and self._source_bytes + source_bytes > self._max_source_bytes
            ):
                _evicted_key, evicted = self._entries.popitem(last=False)
                self._source_bytes -= evicted[2]
        value = compute()
        with self._lock:
            # Another thread may have filled this key while compute ran.
            previous = self._entries.pop(key, None)
            if previous is not None:
                self._source_bytes -= previous[2]
            self._entries[key] = (stats, value, source_bytes)
            self._source_bytes += source_bytes
            # Keep the value just requested even if it alone exceeds the
            # budget. Evicting it immediately would turn every access into a
            # full recomputation; the next distinct entry replaces it.
            while (
                self._max_source_bytes is not None
                and self._source_bytes > self._max_source_bytes
                and len(self._entries) > 1
            ):
                _evicted_key, evicted = self._entries.popitem(last=False)
                self._source_bytes -= evicted[2]
        return value
