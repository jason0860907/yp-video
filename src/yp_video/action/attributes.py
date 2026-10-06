"""Per-touch attributes the SPOT side / jump heads train on.

Stored action labels hold neither attribute for most touches, so the training
snapshot fills them in, a stored value always winning:

- ``side`` — the court side the touching player stood on. The team that won
  a rally serves the next one, so the previous rally's ``winner`` says which
  side served, and the touch order (``rules.touch_sides``) carries it through
  the rally. Ends change between sets and the winner label can be wrong, so
  the anchor is checked against where each serve was struck: a video's serves
  must split cleanly by anchored side along the camera's court axis, and a
  rally whose serve sits on the other side's half is left out.
- ``jump`` — whether the touching player was off the floor: spikes and blocks
  are, receives are not; serves and sets go either way and supervise only
  where a value is stored.
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections.abc import Mapping, Sequence
from pathlib import Path

from yp_video.action.rules import touch_sides
from yp_video.config import ACTION_ANNOTATIONS_DIR
from yp_video.contracts.action import DEFAULT_FPS, LABEL_FILE_GLOB, LABEL_FILE_SUFFIX
from yp_video.core.jsonl import read_jsonl
from yp_video.core.rallies import load_rallies

JUMP_DEFAULTS = {"spike": True, "block": True, "receive": False}
#: The two camera axes; per axis the side whose serves sit at the larger
#: image coordinate on that axis is learned per video (a near server's toss
#: can be struck higher in the frame than a far one's).
AXES = {"lr": ("left", "right", 0), "nf": ("far", "near", 1)}
AXIS_OF = {side: axis for axis, (a, b, _) in AXES.items() for side in (a, b)}
#: Share of a video's anchored serves that must sit on their anchor's half
#: for the video to supervise side at all (median video: 0.98).
MIN_SERVE_AGREEMENT = 0.8


def derive_sides(
    rallies: Sequence[Mapping], events: Sequence[Mapping], fps: float
) -> tuple[list[str | None], dict]:
    """Side per event (aligned with ``events``) and how the video fared."""
    sides: list[str | None] = [None] * len(events)
    axes = [AXIS_OF[r["winner"]] for r in rallies if r.get("winner")]
    if not axes:
        return sides, {"status": "no winner labels"}
    axis = max(set(axes), key=axes.count)
    low, high, coord = AXES[axis]

    order = sorted(range(len(events)), key=lambda i: events[i]["frame"])
    anchored = []  # (anchor side, serve coordinate, event positions, sides)
    for previous, rally in zip([None, *rallies[:-1]], rallies):
        positions = [
            i for i in order
            if rally["start"] <= events[i]["frame"] / fps <= rally["end"]
        ]
        anchor = previous and previous.get("winner")
        if not positions or anchor not in (low, high):
            continue
        rally_sides = touch_sides([events[i] for i in positions], anchor)
        if rally_sides[0] is None:
            continue
        anchored.append(
            (anchor, float(events[positions[0]]["xy"][coord]), positions, rally_sides)
        )

    by_side = {
        side: [c for anchor, c, _, _ in anchored if anchor == side] for side in (low, high)
    }
    if not by_side[low] or not by_side[high]:
        return sides, {"status": "serves from one side only", "rallies": len(anchored)}
    low_median, high_median = (statistics.median(by_side[s]) for s in (low, high))
    middle = (low_median + high_median) / 2
    high_is_larger = high_median > low_median
    agrees = [((c > middle) == high_is_larger) == (anchor == high) for anchor, c, _, _ in anchored]
    agreement = sum(agrees) / len(agrees)
    report = {
        "axis": axis,
        "rallies": len(rallies),
        "anchored_rallies": len(anchored),
        "serve_agreement": round(agreement, 3),
    }
    if agreement < MIN_SERVE_AGREEMENT:
        return sides, {**report, "status": "serve positions disagree with winners"}
    for (_, _, positions, rally_sides), agree in zip(anchored, agrees):
        if agree:
            for i, side in zip(positions, rally_sides):
                sides[i] = side
    return sides, {**report, "status": "ok", "kept_rallies": sum(agrees)}


def attribute_defaults(
    rallies: Sequence[Mapping], events: Sequence[Mapping], fps: float
) -> tuple[list[dict], dict]:
    """What each event's side / jump is when nothing is stored — ``{"side",
    "jump"}`` per event, None where unknown — and the side report."""
    sides, report = derive_sides(rallies, events, fps)
    return [
        {"side": side, "jump": JUMP_DEFAULTS.get(event["label"])}
        for event, side in zip(events, sides)
    ], report


def with_attributes(
    stem: str, fps: float, events: Sequence[Mapping]
) -> tuple[list[dict], dict]:
    """``events`` with ``side`` / ``jump`` filled where known, and the side
    derivation's report. A stored value wins; an unknown stays absent."""
    defaults, report = attribute_defaults(load_rallies(stem), events, fps)
    out = []
    for event, default in zip(events, defaults):
        event = dict(event)
        for key, value in default.items():
            if key not in event and value is not None:
                event[key] = value
        out.append(event)
    return out, report


def _report(directory: Path) -> None:
    totals = {"events": 0, "side": 0, "jump": 0}
    for path in sorted(directory.glob(LABEL_FILE_GLOB)):
        meta, events = read_jsonl(path)
        stem = path.name.removesuffix(LABEL_FILE_SUFFIX)
        labelled, report = with_attributes(stem, float(meta.get("fps") or DEFAULT_FPS), events)
        counts = {
            "events": len(labelled),
            "side": sum("side" in e for e in labelled),
            "jump": sum("jump" in e for e in labelled),
        }
        for key, value in counts.items():
            totals[key] += value
        print(json.dumps({"video": stem, **counts, **report}, ensure_ascii=False))
    print(json.dumps({"total": totals}))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Per-video side / jump coverage of the action labels."
    )
    parser.add_argument("directory", type=Path, nargs="?", default=ACTION_ANNOTATIONS_DIR)
    _report(parser.parse_args().directory)


if __name__ == "__main__":
    main()
