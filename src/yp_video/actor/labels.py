"""The durable human verdict on who performed each action event.

One event, one label — ``videos/association/annotations/<stem>_actors.json``:

    {"version": 2,
     "actors": {
       "<e>": {"verdict": "manual", "track": "12:3", "box": [...], "frame": 812},
       "<e>": {"verdict": "manual", "box": [x0,y0,x1,y1], "frame": 3011},
       "<e>": {"verdict": "occluded"},
       "<e>": {"verdict": "confirmed_auto", "track": "12:7", "box": [...], "frame": 40}
     }}

A label names a TRACKLET when one was picked, and a bare box otherwise. The
tracklet is the better answer — it survives a box that jitters, it spans the
frames where the actor is occluded, and it re-resolves deterministically on
re-extraction instead of being IoU-matched back onto a fresh detection that
may be somebody else. The box stays either way, but its job changes:

- with ``track``  the box is the ANCHOR — where the human clicked. It is what
                  re-resolves the label if tracking is ever re-run and the
                  ids are renumbered (``track_id`` restarts per rally, so
                  re-tracking WILL renumber).
- without         the box is the answer itself: today's behaviour, kept for
                  the events no tracklet reaches (~7%) and for videos tracked
                  before instance masks existed.

The verdict IS the state. Nothing infers "the user marked this occluded"
from a missing box or "this was a manual pick" from the presence of one:
those inversions are how two records of the same fact drift apart.

The three verdicts differ in who chose the box and whether extraction must
act on it:

- ``manual``          the user picked this person. Re-extraction replays it
                      (see extraction/pipeline.py). For a box label, ``frame``
                      set = the actor was undetected on the event frame and
                      the user clicked them on a nearby one, and
                      ``snap=False`` = embed the box exactly as drawn.
- ``occluded``        nobody in frame is the actor. No box exists to record.
- ``confirmed_auto``  the user endorsed the automatic pick by assigning the
                      crop an identity and marking the video done. ``box``
                      snapshots what they endorsed, so later re-extraction
                      cannot silently reinterpret the endorsement.

Only the first two override the automatic pick (``ActorLabel.overrides_auto``)
— a confirmation agrees with it by definition. All three are training truth
for the learned ranker (see actor/dataset.py).

Player identity is a different label with a different lifetime, and now a
different directory: ``videos/reid/annotations/<stem>_players.json`` (see
reid/store.py). The two are written under separate locks, so naming a player
never blocks fixing an actor.
"""

from __future__ import annotations

import json
import threading
from collections.abc import Iterable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

from yp_video.actor.resolution import ActorResolution, actor_resolution
from yp_video.config import ASSOCIATION_ANNOTATIONS_DIR
from yp_video.tracklets.geometry import TrackRef
from yp_video.core.cache import StatCache
from yp_video.core.jsonl import atomic_write

SCHEMA_VERSION = 2
#: The name this package owns inside the shared annotations directory.
#: Public so a caller can count or list actor-labelled videos without
#: re-spelling the suffix and drifting from it.
LABEL_SUFFIX = "_actors.json"


class ActorVerdict(str, Enum):
    MANUAL = "manual"
    OCCLUDED = "occluded"
    CONFIRMED_AUTO = "confirmed_auto"


@dataclass(frozen=True)
class ActorLabel:
    """One event's human actor verdict, and the truth behind it."""

    verdict: ActorVerdict
    #: The tracklet the human picked. When set, this is the answer and ``box``
    #: is only the anchor that can re-derive it.
    track: TrackRef | None = None
    box: tuple[float, float, float, float] | None = None
    #: Where the box was drawn. For a box label a value different from the
    #: event's frame means a cross-frame pick.
    frame: int | None = None
    #: False when no stored detection is this player, so an IoU snap could
    #: only attach an occluder. Meaningless for a tracklet label — the track
    #: already names the person.
    snap: bool = True

    @property
    def overrides_auto(self) -> bool:
        """Whether extraction must replace the automatic pick with this."""
        return self.verdict is not ActorVerdict.CONFIRMED_AUTO

    @property
    def is_tracklet(self) -> bool:
        return self.track is not None

    def payload(self) -> dict:
        """The JSON form — defaults stay absent so the file reads clean."""
        out: dict = {"verdict": self.verdict.value}
        if self.track is not None:
            out["track"] = self.track.key
        if self.box is not None:
            out["box"] = [round(float(value), 1) for value in self.box]
        if self.frame is not None:
            out["frame"] = int(self.frame)
        if not self.snap:
            out["snap"] = False
        return out

    @classmethod
    def from_payload(cls, payload: object) -> "ActorLabel | None":
        """Parse one entry; None when it is unreadable, never a guess."""
        if not isinstance(payload, dict):
            return None
        try:
            verdict = ActorVerdict(str(payload.get("verdict")))
        except ValueError:
            return None
        return cls(
            verdict=verdict,
            track=_track_from(payload.get("track")),
            box=box_from(payload.get("box")),
            frame=(
                int(payload["frame"])
                if isinstance(payload.get("frame"), int)
                else None
            ),
            snap=payload.get("snap") is not False,
        )


def _track_from(value: object) -> TrackRef | None:
    """Parse a "rally:track" key; unreadable is absent, never a guess."""
    if not isinstance(value, str):
        return None
    try:
        return TrackRef.parse(value)
    except (ValueError, TypeError):
        return None


def box_from(value: object) -> tuple[float, float, float, float] | None:
    """A four-corner box from stored JSON, or None when it isn't one."""
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        return None
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


# Serializes read-modify-write; the UI can land two picks back to back.
_lock = threading.RLock()
# Readers go through the cache; writers re-read under the lock.
_cache: StatCache = StatCache()


@contextmanager
def write_transaction() -> Iterator[None]:
    """Hold the label file across a multi-file actor transaction."""
    with _lock:
        yield


def _read(stem: str) -> dict[str, ActorLabel]:
    path = actors_path(stem)
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    labels = {}
    for event_id, payload in (data.get("actors") or {}).items():
        label = ActorLabel.from_payload(payload)
        if label is not None:
            labels[str(event_id)] = label
    return labels


def _write(stem: str, labels: dict[str, ActorLabel]) -> None:
    path = actors_path(stem)
    if not labels:
        path.unlink(missing_ok=True)
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with atomic_write(path) as file:
        json.dump(
            {
                "version": SCHEMA_VERSION,
                "actors": {
                    event_id: labels[event_id].payload()
                    for event_id in sorted(labels)
                },
            },
            file,
            ensure_ascii=False,
            indent=1,
        )


def load(stem: str) -> dict[str, ActorLabel]:
    """Every actor label for one video. Cached — SHARED, read-only."""
    path = actors_path(stem)
    if not path.exists():
        return {}
    return _cache.get(stem, [path], lambda: _read(stem))


def save(stem: str, event_id: str, label: ActorLabel | None) -> None:
    """Set (or with ``label=None`` clear) one event's verdict."""
    with _lock:
        labels = _read(stem)
        if label is None:
            labels.pop(event_id, None)
        else:
            labels[event_id] = label
        _write(stem, labels)


def confirmations_for(
    records: Iterable[Mapping[str, object]],
) -> dict[str, ActorLabel]:
    """Every automatic answer a human could endorse, as the label it would be.

    A policy gives two kinds of answer worth agreeing with, and agreeing with
    each means a different verdict:

    - it PICKED somebody → ``confirmed_auto``. The box is snapshotted so a
      later re-extraction cannot quietly reinterpret what was endorsed.
    - it said NOBODY IS VISIBLE → ``occluded``. That is a real verdict, not a
      confirmation of a pick, because there is no pick to confirm — and it is
      the training truth the NONE head is scored on.

    Only an explicit occlusion counts, never a mere abstention: ``untracked``
    means the model believes somebody acted and tracking has no box for them,
    which re-running tracking may fix and a verdict would bury. A rule policy
    that never abstains produces neither.

    WHO may endorse them is the caller's question, and the two labeling pages
    answer it differently: naming the crop (ReID Label) and reviewing the
    video (Association Label) are both evidence a human looked.
    """
    out: dict[str, ActorLabel] = {}
    for record in records:
        try:
            resolution = actor_resolution(record)
        except ValueError:
            continue  # unmigrated record; never guess what it was
        if resolution is ActorResolution.AUTO:
            box = box_from(record.get("actor_box"))
            if box is None:
                continue
            frame = record.get("frame")
            out[str(record["id"])] = ActorLabel(
                verdict=ActorVerdict.CONFIRMED_AUTO,
                box=box,
                frame=frame if isinstance(frame, int) else None,
            )
        elif resolution is ActorResolution.UNRESOLVED and _says_occluded(record):
            out[str(record["id"])] = ActorLabel(ActorVerdict.OCCLUDED)
    return out


def _says_occluded(record: Mapping[str, object]) -> bool:
    """Whether the policy's own answer was "nobody is visible here"."""
    diagnostic = record.get("association")
    return (
        isinstance(diagnostic, Mapping) and diagnostic.get("kind") == "occluded"
    )


def confirm_auto(stem: str, confirmations: dict[str, ActorLabel]) -> list[str]:
    """Record endorsements of the automatic pick, never overwriting a fix.

    A manual or occluded verdict is the stronger statement: the user looked
    at that event and disagreed with the machine. Bulk confirmation must not
    undo it, so existing labels win.

    Returns the events actually confirmed, decided under the same lock as the
    write — a caller comparing before and after could not report that
    honestly.
    """
    with _lock:
        labels = _read(stem)
        added = sorted(set(confirmations) - set(labels))
        if added:
            labels.update({event_id: confirmations[event_id] for event_id in added})
            _write(stem, labels)
        return added
