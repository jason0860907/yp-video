"""Shared JSONL read/write utilities for _meta-header JSONL files."""

import json
import os
import threading
from collections import OrderedDict
from contextlib import contextmanager
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Iterable, Iterator, TextIO


def _dumps(obj: dict) -> str:
    return json.dumps(obj, ensure_ascii=False) + "\n"


@contextmanager
def atomic_write(path: Path) -> Iterator[TextIO]:
    """Open a temp file for writing and rename it over ``path`` on success.

    A concurrent reader sees the old or the new file — never a half-written
    one. On failure the temp file is removed and nothing changes.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, prefix=f"{path.name}.", suffix=".tmp", delete=False
    ) as f:
        try:
            yield f
        except BaseException:
            os.unlink(f.name)
            raise
    os.replace(f.name, path)


def read_jsonl(path: Path) -> tuple[dict, list[dict]]:
    """Read a JSONL file with a _meta header line.

    Returns:
        (meta, records) — meta dict (without _meta key) and list of record dicts.
    """
    with open(path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    if not lines:
        return {}, []

    meta = json.loads(lines[0])
    meta.pop("_meta", None)

    records = []
    for line in lines[1:]:
        line = line.strip()
        if line:
            records.append(json.loads(line))

    return meta, records


# Parsed-file cache for the frequently re-read jsonls. Keyed by (mtime, size),
# so the atomic rewrites in write_jsonl invalidate entries naturally. Sized to
# hold every cut's annotation file AND reid/tracks jsonl at once —
# /reid/videos touches ALL of them per page load, and an LRU smaller than the
# working set thrashes on every request.
_READ_CACHE_SIZE = 256
_read_cache: OrderedDict[Path, tuple[tuple[int, int], dict, list[dict]]] = OrderedDict()
_read_cache_lock = threading.Lock()


def read_jsonl_cached(path: Path) -> tuple[dict, list[dict]]:
    """``read_jsonl`` behind a small mtime-keyed LRU.

    Every caller receives the SAME meta/record objects — treat them as
    read-only. Callers that mutate (or feed a read-modify-write) must use
    ``read_jsonl`` directly.
    """
    stat = path.stat()
    key = (stat.st_mtime_ns, stat.st_size)
    with _read_cache_lock:
        hit = _read_cache.get(path)
        if hit and hit[0] == key:
            _read_cache.move_to_end(path)
            return hit[1], hit[2]
    meta, records = read_jsonl(path)
    with _read_cache_lock:
        _read_cache[path] = (key, meta, records)
        _read_cache.move_to_end(path)
        while len(_read_cache) > _READ_CACHE_SIZE:
            _read_cache.popitem(last=False)
    return meta, records


def write_jsonl(path: Path, meta: dict, records: Iterable[dict]) -> None:
    """Write a JSONL file with a _meta header line plus records in one go.

    The meta dict is augmented with `"_meta": True` if missing.
    Use this for one-shot writes (e.g. converters); training loops that append
    rows over time should use ``write_meta_header`` + ``append_jsonl`` instead.

    The write is atomic (see ``atomic_write``).
    """
    meta_out = {"_meta": True, **meta} if "_meta" not in meta else meta
    with atomic_write(path) as f:
        f.write(_dumps(meta_out))
        for rec in records:
            f.write(_dumps(rec))


def write_meta_header(path: Path, meta: dict) -> None:
    """Truncate ``path`` and write a single _meta header line.

    Used at training start; subsequent epoch entries get appended via
    ``append_jsonl``.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    meta_out = {"_meta": True, **meta} if "_meta" not in meta else meta
    with open(path, "w", encoding="utf-8") as f:
        f.write(_dumps(meta_out))


def append_jsonl(path: Path, record: dict) -> None:
    """Append a single record to an existing JSONL file."""
    with open(path, "a", encoding="utf-8") as f:
        f.write(_dumps(record))
