"""Shared JSONL read/write utilities for _meta-header JSONL files."""

import json
from pathlib import Path


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
