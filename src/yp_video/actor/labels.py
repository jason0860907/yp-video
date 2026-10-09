"""The durable human verdict on who performed each action event.

One event, one label — ``videos/association/annotations/<stem>_actors.json``:

    {"version": 3,
     "actors": {
       "<e>": {"verdict": "manual", "frame": 812, "box": [x0, y0, x1, y1]},
       "<e>": {"verdict": "confirmed_auto", "frame": 40, "box": [...]},
       "<e>": {"verdict": "occluded"}
     }}

A label answers one question: which person on screen performed this event.
The answer is a box in pixels on ``frame`` — written on the event's own
frame, and picked from the RF-DETR Seg 2XLarge dense pass (person/dense.py),
the box style the person head learns. Nothing in it points at a tracklet or
a detection: those are derived data that a re-run renumbers or replaces,
while a box on a frame means the same person forever.

``frame`` is kept because an action annotation may later move its event a
frame or two. A label whose frame is no longer its event's is followed there
through the dense boxes (actor/box_style.event_box) — never taken as is, and
never guessed when the walk loses the person.

The verdict IS the state, and says who chose the box:

- ``manual``          the user picked this person.
- ``occluded``        nobody in frame is the actor. No box exists to record.
- ``confirmed_auto``  the user endorsed the automatic pick (by naming the
                      crop, or reviewing the video). The box snapshots what
                      they endorsed, so later re-extraction cannot silently
                      reinterpret the endorsement.

Only the first two override the automatic pick (``ActorLabel.overrides_auto``)
— a confirmation agrees with it by definition. All three are training truth
for the person/action head (yp-spot scripts/prepare_person_action.py).

Player identity is a different label with a different lifetime, and a
different directory: ``videos/reid/annotations/<stem>_players.json`` (see
reid/store.py). The two are written under separate locks, so naming a player
never blocks fixing an actor.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

from yp_video.config import ASSOCIATION_ANNOTATIONS_DIR
from yp_video.core.sidecar import JsonSidecar

SCHEMA_VERSION = 3
#: The name this package owns inside the shared annotations directory.
#: Public so a caller can count or list actor-labelled videos without
#: re-spelling the suffix and drifting from it.
LABEL_SUFFIX = "_actors.json"

Box = tuple[float, float, float, float]


class ActorVerdict(str, Enum):
    MANUAL = "manual"
    OCCLUDED = "occluded"
    CONFIRMED_AUTO = "confirmed_auto"


@dataclass(frozen=True)
class ActorLabel:
    """One event's human actor verdict, and the box behind it.

    Occluded carries neither ``frame`` nor ``box``; the other verdicts carry
    both. Constructing anything else raises, so no reader has to ask.
    """

    verdict: ActorVerdict
    #: The frame ``box`` is on — the event's when it was written.
    frame: int | None = None
    #: The actor's pixel xyxy on ``frame``.
    box: Box | None = None

    def __post_init__(self) -> None:
        occluded = self.verdict is ActorVerdict.OCCLUDED
        if occluded != (self.box is None) or occluded != (self.frame is None):
            raise ValueError(
                f"A {self.verdict.value} label must carry "
                + ("neither a frame nor a box" if occluded else "both a frame and a box")
            )

    @property
    def overrides_auto(self) -> bool:
        """Whether extraction must replace the automatic pick with this."""
        return self.verdict is not ActorVerdict.CONFIRMED_AUTO

    def payload(self) -> dict:
        out: dict = {"verdict": self.verdict.value}
        if self.box is not None:
            out["frame"] = int(self.frame)
            out["box"] = [round(float(value), 1) for value in self.box]
        return out

    @classmethod
    def from_payload(cls, payload: Mapping) -> "ActorLabel":
        """Parse one entry; anything but the shape ``payload`` writes raises."""
        extra = set(payload) - {"verdict", "frame", "box"}
        if extra:
            raise ValueError(f"Unknown actor label fields: {sorted(extra)}")
        frame = payload.get("frame")
        if frame is not None and (not isinstance(frame, int) or frame < 0):
            raise ValueError(f"Invalid actor label frame: {frame!r}")
        box = payload.get("box")
        return cls(
            verdict=ActorVerdict(payload["verdict"]),
            frame=frame,
            box=None if box is None else _box_from(box),
        )


def _box_from(value: object) -> Box:
    """A four-corner box from stored JSON; anything else raises."""
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        raise ValueError(f"Not a box: {value!r}")
    x0, y0, x1, y1 = (float(v) for v in value)
    return x0, y0, x1, y1


def actors_path(stem: str) -> Path:
    return ASSOCIATION_ANNOTATIONS_DIR / f"{stem}{LABEL_SUFFIX}"


def labeled_stems() -> list[str]:
    """Every video carrying actor labels, sorted."""
    if not ASSOCIATION_ANNOTATIONS_DIR.exists():
        return []
    return sorted(
        path.name[: -len(LABEL_SUFFIX)]
        for path in ASSOCIATION_ANNOTATIONS_DIR.glob(f"*{LABEL_SUFFIX}")
    )


# Late-bound so tests patching ``actors_path`` redirect the store too.
_store = JsonSidecar(lambda stem: actors_path(stem))

#: Hold the label file across a multi-file actor transaction (see the lock
#: order contract in extraction/actor_fix.py).
write_transaction = _store.transaction


def _parse(data: dict) -> dict[str, ActorLabel]:
    """Every label in one file ({} when absent); another schema raises."""
    if not data:
        return {}
    if data.get("version") != SCHEMA_VERSION:
        raise ValueError(f"Actor labels version {data.get('version')!r}, expected {SCHEMA_VERSION}")
    return {
        str(event_id): ActorLabel.from_payload(payload)
        for event_id, payload in data["actors"].items()
    }


def _read(stem: str) -> dict[str, ActorLabel]:
    return _parse(_store.read_fresh(stem))


def _write(stem: str, labels: dict[str, ActorLabel]) -> None:
    if not labels:
        _store.write(stem, None)
        return
    _store.write(
        stem,
        {
            "version": SCHEMA_VERSION,
            "actors": {
                event_id: labels[event_id].payload()
                for event_id in sorted(labels)
            },
        },
    )


def load(stem: str) -> dict[str, ActorLabel]:
    """Every actor label for one video. Cached — SHARED, read-only."""
    return _store.cached(stem, _parse)


def save(stem: str, event_id: str, label: ActorLabel | None) -> None:
    """Set (or with ``label=None`` clear) one event's verdict."""
    with _store.transaction():
        labels = _read(stem)
        if label is None:
            labels.pop(event_id, None)
        else:
            labels[event_id] = label
        _write(stem, labels)


def confirm_auto(stem: str, confirmations: dict[str, ActorLabel]) -> list[str]:
    """Record endorsements of the automatic pick, never overwriting a fix.

    A manual or occluded verdict is the stronger statement: the user looked
    at that event and disagreed with the machine. Bulk confirmation must not
    undo it, so existing labels win.

    Returns the events actually confirmed, decided under the same lock as the
    write — a caller comparing before and after could not report that
    honestly.
    """
    with _store.transaction():
        labels = _read(stem)
        added = sorted(set(confirmations) - set(labels))
        if added:
            labels.update({event_id: confirmations[event_id] for event_id in added})
            _write(stem, labels)
        return added
