"""Per-video, per-mode "labeling is finished" flags.

The Label page's Done button is a human verdict no counts can derive — a
video with unreviewed events may still be as done as it will ever get, and a
fully-covered one may need another pass. What marking Done *implies*
(ReID's confirm-auto-actors, association's standing endorsement) belongs to
the endpoints that set it — this module only stores the verdict.

All verdicts live in one ledger, ``label-done.jsonl``: one line per video
carrying every mode's flag. One file is what makes the verdicts a single
R2 object (config.R2_CATEGORIES) instead of a directory of sidecars that no
backup covered. It is kilobytes, so every write rewrites it whole, and the
web app mirrors each write to R2 through ``ledger.on_write`` (wired in
app.py — this module knows nothing about storage).
"""

from __future__ import annotations

import threading
from collections.abc import Callable
from pathlib import Path

from yp_video.config import LABEL_DONE_FILE
from yp_video.core.cache import StatCache
from yp_video.core.jsonl import read_jsonl, write_jsonl

#: Modes whose Done flag lives here.
MODES = ("rally", "action", "association", "reid")


class Ledger:
    """The one file: ``{stem: {mode: True}}`` — only True flags are stored."""

    def __init__(self, path: Path, on_write: Callable[[Path], None] | None = None):
        self.path = path
        #: Called with the path after every successful rewrite.
        self.on_write = on_write
        # Serializes read-modify-write; the UI can land two clicks back to back.
        self._lock = threading.RLock()
        # Readers go through the stat cache; writers re-read fresh under the lock.
        self._cache: StatCache = StatCache()

    def _read_fresh(self) -> dict[str, dict[str, bool]]:
        if not self.path.exists():
            return {}
        _meta, rows = read_jsonl(self.path)
        return {
            row["video"]: {m: True for m in MODES if row.get(m)}
            for row in rows
            if isinstance(row.get("video"), str)
        }

    def flags(self) -> dict[str, dict[str, bool]]:
        if not self.path.exists():
            return {}
        return self._cache.get("ledger", [self.path], self._read_fresh)

    def set(self, stem: str, mode: str, done: bool) -> dict[str, bool]:
        with self._lock:
            ledger = self._read_fresh()
            flags = ledger.get(stem, {})
            if done:
                flags[mode] = True
            else:
                flags.pop(mode, None)
            if flags:
                ledger[stem] = flags
            else:
                ledger.pop(stem, None)
            write_jsonl(
                self.path,
                {"modes": list(MODES)},
                ({"video": s, **{m: True for m in MODES if f.get(m)}} for s, f in sorted(ledger.items())),
            )
            if self.on_write:
                self.on_write(self.path)
            return {m: flags.get(m, False) for m in MODES}


ledger = Ledger(LABEL_DONE_FILE)


def load(stem: str) -> dict[str, bool]:
    """Every mode's flag for ``stem`` (missing modes are False)."""
    flags = ledger.flags().get(stem, {})
    return {m: flags.get(m, False) for m in MODES}


def is_done(stem: str, mode: str) -> bool:
    return mode in ledger.flags().get(stem, {})


def set_done(stem: str, mode: str, done: bool) -> dict[str, bool]:
    """Persist one mode's flag."""
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
    return ledger.set(stem, mode, done)
