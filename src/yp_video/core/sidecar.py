"""The JSON-sidecar mechanism actor labels and player names both run on.

One file per video, small enough to rewrite whole: readers go through a
StatCache, writers hold one re-entrant lock and re-read fresh before
mutating, and every write is an atomic replace. Two packages used to carry
word-for-word copies of that machinery — same comments included — around
different payload schemas. The schema (parse/serialize) stays with its
owner; only the mechanism lives here.
"""

from __future__ import annotations

import json
import threading
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import TypeVar

from yp_video.core.cache import StatCache
from yp_video.core.jsonl import atomic_write

T = TypeVar("T")


class JsonSidecar:
    """One sidecar-file family (``<stem>_something.json``)."""

    def __init__(self, path_of: Callable[[str], Path]):
        self._path_of = path_of
        # Serializes read-modify-write; the UI can land two picks back to back.
        self._lock = threading.RLock()
        # Readers go through the cache; writers re-read fresh under the lock.
        self._cache: StatCache = StatCache()

    def path(self, stem: str) -> Path:
        return self._path_of(stem)

    @contextmanager
    def transaction(self) -> Iterator[None]:
        """Hold the file's lock across a multi-file operation."""
        with self._lock:
            yield

    def read_fresh(self, stem: str) -> dict:
        """The raw JSON payload straight from disk ({} when absent)."""
        path = self.path(stem)
        if not path.exists():
            return {}
        return json.loads(path.read_text(encoding="utf-8"))

    def cached(self, stem: str, parse: Callable[[dict], T]) -> T:
        """``parse`` applied to the current payload, cached on the file stat."""
        path = self.path(stem)
        if not path.exists():
            return parse({})
        return self._cache.get(stem, [path], lambda: parse(self.read_fresh(stem)))

    def write(self, stem: str, payload: dict | None) -> None:
        """Atomically replace the file — or, with ``None``, delete it."""
        path = self.path(stem)
        if payload is None:
            path.unlink(missing_ok=True)
            return
        with atomic_write(path) as file:
            json.dump(payload, file, ensure_ascii=False, indent=1)
