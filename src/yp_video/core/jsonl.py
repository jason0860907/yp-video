"""Shared JSONL read/write utilities for _meta-header JSONL files."""

import json
from pathlib import Path
from typing import Iterable


def _dumps(obj: dict) -> str:
    return json.dumps(obj, ensure_ascii=False) + "\n"


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


def write_jsonl(path: Path, meta: dict, records: Iterable[dict]) -> None:
    """Write a JSONL file with a _meta header line plus records in one go.

    The meta dict is augmented with `"_meta": True` if missing.
    Use this for one-shot writes (e.g. converters); training loops that append
    rows over time should use ``write_meta_header`` + ``append_jsonl`` instead.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    meta_out = {"_meta": True, **meta} if "_meta" not in meta else meta
    with open(path, "w", encoding="utf-8") as f:
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
