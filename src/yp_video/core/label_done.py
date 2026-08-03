"""Per-video, per-mode "labeling is finished" flags.

The Label page's Done button is a human verdict no counts can derive — a
video with unreviewed events may still be as done as it will ever get, and a
fully-covered one may need another pass. One sidecar per video holds every
mode's flag: ``{"rally": true, "reid": true}``. What marking Done *implies*
(ReID's confirm-auto-actors, association's standing endorsement) belongs to
the endpoints that set it — this module only stores the verdict.
"""

from __future__ import annotations

from yp_video.config import LABEL_DONE_DIR
from yp_video.core.sidecar import JsonSidecar

#: Modes whose Done flag lives here.
MODES = ("rally", "action", "association", "reid")

_sidecar = JsonSidecar(lambda stem: LABEL_DONE_DIR / f"{stem}_done.json")


def load(stem: str) -> dict[str, bool]:
    """Every mode's flag for ``stem`` (missing modes are False)."""
    return _sidecar.cached(stem, _parse)


def is_done(stem: str, mode: str) -> bool:
    return load(stem).get(mode, False)


def set_done(stem: str, mode: str, done: bool) -> dict[str, bool]:
    """Persist one mode's flag; the file disappears when nothing is done."""
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
    with _sidecar.transaction():
        flags = {k: v for k, v in _sidecar.read_fresh(stem).items() if v}
        if done:
            flags[mode] = True
        else:
            flags.pop(mode, None)
        _sidecar.write(stem, flags or None)
    return {m: flags.get(m, False) for m in MODES}


def _parse(payload: dict) -> dict[str, bool]:
    return {m: bool(payload.get(m)) for m in MODES}
