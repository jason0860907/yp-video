"""Re-decide every automatic actor pick in a video, in place.

Extraction answers "who performed this action" as a side effect of detecting
people, which means changing the answer used to mean paying for detection
again — minutes of GPU per video to re-run a decision that needs no GPU at
all. The detections are already in the record; only the choice among them is
being revisited. This module revisits it.

Two rules make the job safe to run on a labeled video:

- A human verdict is never touched. Not overwritten, not re-derived, not
  "refreshed" — an event a person has ruled on leaves this module unread.
- Only events whose pick actually MOVED are re-cropped. A policy that agrees
  with the previous one costs a jsonl rewrite and nothing else, so re-running
  is cheap and idempotent rather than a full re-encode.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from yp_video.actor import labels as actor_labels
from yp_video.actor.policy import ActorPick, ActorPolicy, EventContext
from yp_video.actor.resolution import ActorResolution
from yp_video.core.jsonl import read_jsonl, read_jsonl_cached, write_jsonl
from yp_video.core.progress import ProgressFn
from yp_video.extraction.links import resolve_track
from yp_video.extraction.store import crop_dir, masked_crop_dir, records_path
from yp_video.person.detector import PersonBox
from yp_video.tracklets.store import tracks_path

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class ReassociationCounts:
    """What the run did, in the terms the job card reports."""

    events: int = 0
    #: Left alone because a human had already ruled on them.
    labeled: int = 0
    changed: int = 0
    unchanged: int = 0
    #: The policy declined to answer where the previous one had picked someone.
    abstained: int = 0
    #: Decided, but the pick could not be turned into pixels.
    unresolvable: int = 0

    def payload(self) -> dict:
        return {
            "events": self.events,
            "labeled": self.labeled,
            "changed": self.changed,
            "unchanged": self.unchanged,
            "abstained": self.abstained,
            "unresolvable": self.unresolvable,
        }


def _update(record: dict, **fields) -> bool:
    """Assign only the fields that differ; report whether any did.

    The records file's mtime is what marks every embedding matrix for this
    video stale. Rewriting it when nothing moved — a fully reviewed video, or
    a second run of the same policy — would order a full re-embed to produce
    byte-identical vectors, so "did anything actually change" has to be
    tracked rather than assumed.
    """
    changed = False
    for key, value in fields.items():
        if value is _ABSENT:
            if key in record:
                del record[key]
                changed = True
        elif record.get(key) != value:
            record[key] = value
            changed = True
    return changed


class _Absent:
    """Marks a field that must not be present, as distinct from one set to
    None — ``crop_frame`` and ``track`` are absent or meaningful, never null."""

    def __repr__(self) -> str:
        return "<absent>"


_ABSENT = _Absent()


def _target(
    stem: str, record: dict, pick: ActorPick
) -> tuple[tuple[float, float, float, float], int, bool] | None:
    """Where the pick says to crop: (box, frame, may-snap).

    A tracklet is re-resolved through the masks exactly as a tracklet LABEL
    is — same function, so an automatic tracklet pick and a hand-placed one
    cannot drift into cropping different pixels for the same tracklet.
    """
    if pick.track is not None:
        resolved = resolve_track(stem, record, pick.track)
        return (
            (resolved.box, resolved.frame, resolved.snap)
            if resolved is not None
            else None
        )
    if pick.box is None:
        return None
    return pick.box, record["frame"], False


def reassociate_video(
    video_path: Path,
    policy: ActorPolicy,
    *,
    on_progress: ProgressFn | None = None,
) -> dict:
    """Re-run ``policy`` over one video's extracted records."""
    import cv2

    stem = video_path.stem
    path = records_path(stem)
    if not path.exists():
        raise FileNotFoundError(f"No extraction records for {stem}")

    tracklets: list[dict] = []
    if policy.needs_tracklets:
        tracks = tracks_path(stem)
        if not tracks.exists():
            raise FileNotFoundError(
                f"{policy.name} needs tracklets; {stem} has not been tracked"
            )
        _tmeta, tracklets = read_jsonl_cached(tracks)

    meta, records = read_jsonl(path)
    frame_w, frame_h = meta.get("frame_size") or [0, 0]
    verdicts = actor_labels.load(stem)

    # Pass one: decide. No video is opened until it is known which frames are
    # actually needed.
    pending: list[tuple[int, dict, tuple, int, bool]] = []
    counts = {"events": 0, "labeled": 0, "unchanged": 0, "abstained": 0, "unresolvable": 0}
    dirty = False
    for row, record in enumerate(records):
        counts["events"] += 1
        if str(record.get("id")) in verdicts:
            counts["labeled"] += 1
            continue

        xy = record.get("xy")
        context = EventContext(
            frame=int(record["frame"]),
            contact=(
                (float(xy[0]) * frame_w, float(xy[1]) * frame_h)
                if xy and frame_w and frame_h
                else None
            ),
            visible=bool(record.get("visible", True)),
            detections=record.get("detections") or [],
            tracklets=tracklets,
        )
        pick = policy.decide(context)
        dirty |= _update(
            record,
            association=pick.diagnostic or _ABSENT,
            candidates=pick.candidates,
        )

        if not pick.decided:
            if record.get("box") is not None:
                counts["abstained"] += 1
                dirty |= _clear(record)
            else:
                counts["unchanged"] += 1
            continue

        target = _target(stem, record, pick)
        if target is None:
            counts["unresolvable"] += 1
            dirty |= _clear(record)
            continue

        box, src_frame, may_snap = target
        settled = _same_pick(record, box, src_frame, pick, frame_w, frame_h)
        # The tracklet reference is the pick; the box is only where it lands
        # today. Written after the comparison, which reads the old one.
        dirty |= _update(
            record,
            track=pick.track.key if pick.track is not None else _ABSENT,
        )
        if settled:
            counts["unchanged"] += 1
            continue
        pending.append((row, record, box, src_frame, may_snap))

    # Pass one is pure bookkeeping over records already in memory; the crops
    # are the minutes. So the re-crop pass IS the progress, and this is its
    # phase start rather than a fraction of some weighted whole.
    if on_progress is not None:
        on_progress(0, len(pending), f"{len(pending)} picks moved; re-cropping...")

    # Pass two: cut only what moved. Sorted by frame so the seeks run forward.
    changed = 0
    if pending:
        capture = cv2.VideoCapture(str(video_path))
        try:
            for index, (row, record, box, src_frame, may_snap) in enumerate(
                sorted(pending, key=lambda item: item[3])
            ):
                capture.set(cv2.CAP_PROP_POS_FRAMES, src_frame)
                ok, frame_img = capture.read()
                if not ok:
                    log.warning(
                        "Could not decode frame %s of %s; leaving the previous pick",
                        src_frame,
                        video_path.name,
                    )
                    continue
                if _recrop(
                    stem,
                    record,
                    frame_img,
                    box,
                    src_frame,
                    may_snap,
                    frame_w,
                    frame_h,
                ):
                    changed += 1
                if on_progress is not None:
                    on_progress(
                        index + 1,
                        len(pending),
                        f"re-cropping {index + 1}/{len(pending)}",
                    )
        finally:
            capture.release()

    if dirty or changed:
        write_jsonl(path, {**meta, "association_policy": policy.name}, records)
    counts["changed"] = changed
    counts["unchanged"] += len(pending) - changed
    return ReassociationCounts(**counts).payload()


def _clear(record: dict) -> bool:
    """Back to an unresolved event — the state extraction uses for "nobody"."""
    changed = _update(
        record,
        status="miss",
        box=None,
        actor_box=None,
        score=None,
        crop=None,
        keypoints=None,
        resolution=ActorResolution.UNRESOLVED.value,
        crop_frame=_ABSENT,
        track=_ABSENT,
    )
    if changed:
        record["actor_revision"] = int(record.get("actor_revision") or 0) + 1
    return changed


def _same_pick(
    record: dict,
    box,
    src_frame: int,
    pick: ActorPick,
    frame_w: int,
    frame_h: int,
) -> bool:
    """Whether the new answer is the one already stored.

    Compared against the CLAMPED box, because that is what gets stored: a
    player at the edge of frame has a detector box running past it, and
    comparing the raw box to the clamped one made every such event look like
    it had moved — re-cropping the same pixels on every run, forever.
    """
    from yp_video.extraction.pipeline import _clamp_box

    current = record.get("actor_box")
    if current is None or int(record.get("crop_frame") or record["frame"]) != src_frame:
        return False
    if pick.track is not None and record.get("track") != pick.track.key:
        return False
    stored = _clamp_box(box, frame_w, frame_h)
    return all(int(a) == int(b) for a, b in zip(current, stored))


def _recrop(
    stem: str,
    record: dict,
    frame_img,
    box,
    src_frame: int,
    may_snap: bool,
    frame_w: int,
    frame_h: int,
) -> bool:
    from yp_video.extraction.pipeline import _attach_person, _snap_to_detection

    previous = record.get("crop")
    cross_frame = src_frame != record["frame"]
    # Snapping recovers the detector's own score and keypoints. Where it is
    # vetoed the box goes through bare, and that is the point: the veto means
    # the masks found no stored detection that IS this player, so any box
    # close enough to snap to would be the occluder.
    person = (
        _snap_to_detection(record.get("detections") or [], list(box))
        if may_snap and not cross_frame
        else None
    ) or PersonBox(xyxy=box, score=0.0)

    xy = record.get("xy")
    anchor = (
        (float(xy[0]) * frame_w, float(xy[1]) * frame_h)
        if xy and not cross_frame
        else ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)
    )
    record["actor_revision"] = int(record.get("actor_revision") or 0) + 1
    crop = _attach_person(
        record,
        frame_img,
        person,
        anchor[0],
        anchor[1],
        frame_w,
        frame_h,
        crop_dir(stem),
        suffix=f"_p{record['actor_revision']}",
    )
    if crop is None:
        _clear(record)
        return False
    record["status"] = "ok"
    record["resolution"] = ActorResolution.AUTO.value
    if cross_frame:
        record["crop_frame"] = src_frame
    else:
        record.pop("crop_frame", None)
    # The crop filename carries the revision so the browser cannot serve a
    # stale image; the superseded file is derived data with no reader left.
    if previous and previous != record.get("crop"):
        (crop_dir(stem) / previous).unlink(missing_ok=True)
        (masked_crop_dir(stem) / previous).unlink(missing_ok=True)
    return True
