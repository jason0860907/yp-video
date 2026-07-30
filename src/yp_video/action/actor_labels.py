"""Build the actor-candidate supervision yp-spot trains its actor head on.

The action label file says WHEN something happened and WHERE the ball was
touched; who did it lives two packages away, in ``actor`` (the human verdict)
and ``tracklets`` (who was on court). This module is the only place the three
meet, and it exists so the training-label exporter stays a copier.

The unit of supervision is a CHOICE, not a box: the candidate set is every
tracklet with a box on the event frame, and the target names one of them. That
mirrors what the model is asked to do at inference — and it is why an answer
that no candidate matches is reported as ``untracked`` rather than forced onto
the nearest body.

Boxes leave here normalized against the source frame size, matching the ``xy``
contact point in the action labels: pixels would break the moment the frame
cache height changes, and every resize downstream is per-axis linear.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence

from yp_video.actor import labels as actor_labels
from yp_video.actor.labels import ActorLabel, ActorVerdict
from yp_video.contracts.action import ACTOR_WINDOW_OFFSETS, ActorTargetKind
from yp_video.core.jsonl import read_jsonl_cached
from yp_video.extraction.store import records_path
from yp_video.tracklets.store import tracks_path


def _frame_size(stem: str) -> tuple[int, int] | None:
    path = records_path(stem)
    if not path.exists():
        return None
    meta, _records = read_jsonl_cached(path)
    size = meta.get("frame_size")
    if not size or len(size) != 2 or not all(size):
        return None
    return int(size[0]), int(size[1])


def _normalized(
    box: Sequence[float] | None, width: int, height: int
) -> list[float] | None:
    """A box in [0, 1], or None where tracking has none for that frame."""
    if box is None:
        return None
    x0, y0, x1, y1 = (float(v) for v in box)
    if x1 <= x0 or y1 <= y0:
        return None
    return [
        round(min(max(x0 / width, 0.0), 1.0), 5),
        round(min(max(y0 / height, 0.0), 1.0), 5),
        round(min(max(x1 / width, 0.0), 1.0), 5),
        round(min(max(y1 / height, 0.0), 1.0), 5),
    ]


def track_paths(stem: str) -> dict[str, dict[int, Sequence[float]]]:
    """tracklet key → {frame: box}, the whole video."""
    path = tracks_path(stem)
    if not path.exists():
        return {}
    _meta, tracklets = read_jsonl_cached(path)
    return {
        f"{tracklet['rally_id']}:{tracklet['track_id']}": {
            int(frame): box
            for frame, box in zip(tracklet["frames"], tracklet["boxes"])
        }
        for tracklet in tracklets
    }


def candidates_on(
    paths: Mapping[str, Mapping[int, Sequence[float]]], frame: int
) -> list[str]:
    """The tracklets with a box on this exact frame, in a stable order.

    Membership is decided on the event frame alone. A window would let a
    tracklet that had already vanished before the contact re-enter as a
    candidate, and the model would be asked to rule out someone who was not
    there; the window's other offsets only add history for a player already
    established as present.
    """
    return sorted(key for key, boxes in paths.items() if frame in boxes)


def candidates_only(stem: str, events: Iterable[dict]) -> list[dict]:
    """The candidate set per event, with no answer attached.

    What inference needs, and built by the same code that builds training's,
    so the model is asked the question it was taught. A row carries no
    ``target_kind`` at all — absent means unlabelled, and inventing one here
    would put a guess where supervision goes.
    """
    size = _frame_size(stem)
    if size is None:
        return []
    width, height = size
    paths = track_paths(stem)
    rows: list[dict] = []
    for event in events:
        frame = int(event.get("frame", -1))
        keys = candidates_on(paths, frame)
        if not keys:
            continue
        rows.append(
            {
                "id": str(event.get("id")),
                "frame": frame,
                "contact": event.get("xy"),
                "contact_visible": bool(event.get("visible", True)),
                "candidates": [
                    {
                        "track": key,
                        "boxes": [
                            _normalized(paths[key].get(frame + offset), width, height)
                            for offset in ACTOR_WINDOW_OFFSETS
                        ],
                    }
                    for key in keys
                ],
            }
        )
    return rows


def build(stem: str, events: Iterable[dict]) -> tuple[list[dict], dict[str, int]]:
    """The actor-candidate rows for one video's action events, plus a tally.

    Events nobody has ruled on produce NO row at all — this file carries
    supervision, and an event's absence from it is what "unlabelled" means.
    """
    verdicts = actor_labels.load(stem)
    tally = {kind.value: 0 for kind in ActorTargetKind}
    tally["unlabelled"] = 0
    tally["legacy_box"] = 0
    if not verdicts:
        tally["unlabelled"] = sum(1 for _ in events)
        return [], tally

    size = _frame_size(stem)
    paths = track_paths(stem)
    rows: list[dict] = []

    for event in events:
        label = verdicts.get(str(event.get("id")))
        if label is None:
            tally["unlabelled"] += 1
            continue
        if label.verdict is not ActorVerdict.OCCLUDED and label.track is None:
            # A verdict from before tracklet labelling existed: it picked a
            # detection box, so it names a person but not a tracklet. Skipped
            # rather than called `untracked`, because nothing about the frame
            # made it untracked — the label format did, and there is no visual
            # evidence for the model to learn that from.
            tally["legacy_box"] += 1
            continue

        frame = int(event.get("frame", -1))
        keys = candidates_on(paths, frame) if size else []
        width, height = size or (1, 1)
        payload = [
            {
                "track": key,
                "boxes": [
                    _normalized(paths[key].get(frame + offset), width, height)
                    for offset in ACTOR_WINDOW_OFFSETS
                ],
            }
            for key in keys
        ]

        if label.verdict is ActorVerdict.OCCLUDED:
            kind, target = ActorTargetKind.OCCLUDED, None
        else:
            index = keys.index(label.track.key) if label.track.key in keys else None
            if index is None:
                # Somebody acted and the candidate set does not contain them:
                # tracking dropped the player, or never reached this frame.
                kind, target = ActorTargetKind.UNTRACKED, None
            else:
                kind, target = ActorTargetKind.TRACK, index

        tally[kind.value] += 1
        rows.append(
            {
                "id": str(event.get("id")),
                "frame": frame,
                "contact": event.get("xy"),
                "contact_visible": bool(event.get("visible", True)),
                "candidates": payload,
                "target_kind": kind.value,
                **({"target": target} if target is not None else {}),
            }
        )
    return rows, tally
