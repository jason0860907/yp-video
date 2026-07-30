"""Re-decide every automatic actor pick in a video, in place.

Extraction answers "who performed this action" as a side effect of detecting
people, which means changing the answer used to mean paying for detection
again — minutes of GPU per video to re-run a decision that needs no GPU at
all. The detections are already in the record; only the choice among them is
being revisited. This module revisits it.

Two rules make the job safe to run on a labeled video:

- A human verdict is never re-decided. Its crop may be rebuilt when the crop
  geometry contract changes, but the person's answer remains authoritative.
- Only events whose pick actually MOVED are re-cropped. A policy that agrees
  with the previous one costs a jsonl rewrite and nothing else, so re-running
  is cheap and idempotent rather than a full re-encode.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass
from pathlib import Path

from yp_video.actor import labels as actor_labels
from yp_video.actor.labels import ActorVerdict
from yp_video.actor.policy import ActorPick, ActorPolicy, EventContext
from yp_video.actor.resolution import ActorResolution
from yp_video.core.jsonl import read_jsonl, write_jsonl
from yp_video.core.progress import ProgressFn
from yp_video.extraction.cropping import (
    CROP_SCHEMA_VERSION,
    CropTarget,
    clamp_box,
    crop_target,
    cut,
    label_target,
    person_for,
)
from yp_video.extraction.store import (
    crop_dir,
    labelable,
    masked_crop_dir,
    records_path,
)
from yp_video.tracklets.store import (
    TrackMasks,
    open_track_masks,
    tracklet_index,
    tracks_path,
)

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class _Pending:
    """One crop still to cut, and what the record should say once it is.

    An automatic pick and a materialized human verdict both land here, and
    they mean different things afterwards — ``multi`` is "more than one
    candidate was plausible", which is never true of a person pointing.
    """

    row: int
    record: dict
    target: CropTarget
    resolution: ActorResolution
    #: How many candidates the policy weighed. 1 for a human verdict.
    candidates: int


@dataclass
class ReassociationCounts:
    """What the run did, in the terms the job card reports.

    Tallied straight onto the fields. Counting into a plain dict first meant
    a mistyped key surfaced as a TypeError when the dataclass was finally
    built — at the END of a job that had already spent minutes re-cropping.
    """

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
        return asdict(self)


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
    stem: str, record: dict, pick: ActorPick, masks: TrackMasks | None
) -> CropTarget | None:
    """Where the pick says to crop.

    A box answer never snaps: it already IS one of the stored detections (the
    rule chooses among them), so there is nothing to snap onto. An unresolvable
    tracklet answers nothing at all — unlike a human's label, a policy is free
    to abstain and be asked again.
    """
    return crop_target(
        stem,
        record,
        pick.track,
        CropTarget(pick.box, record["frame"], snap=False)
        if pick.box is not None
        else None,
        masks=masks,
    )


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

    tracks_index = None
    masks = None
    if policy.needs_tracklets:
        tracks = tracks_path(stem)
        if not tracks.exists():
            raise FileNotFoundError(
                f"{policy.name} needs tracklets; {stem} has not been tracked"
            )
        tracks_index = tracklet_index(stem)
        # Held open for the whole video: every event reads the same archive,
        # and a policy that ignores outlines never unpacks a single entry.
        masks = open_track_masks(stem)

    meta, records = read_jsonl(path)
    current = {
        str(record["id"]): record
        for record in labelable(records, stem, float(meta.get("fps") or 0))
    }
    frame_w, frame_h = meta.get("frame_size") or [0, 0]
    verdicts = actor_labels.load(stem)

    # Pass one: decide. No video is opened until it is known which frames are
    # actually needed.
    pending: list[_Pending] = []
    counts = ReassociationCounts()
    dirty = False
    # The archive is only needed while deciding; pass two re-crops from
    # the video, so it is closed as soon as the last event is scored.
    try:
        for row, record in enumerate(records):
            action = current.get(str(record.get("id")))
            if action is None:
                continue
            counts.events += 1
            label = verdicts.get(str(record.get("id")))
            if label is not None and _is_materialized(record, label):
                # A human verdict already turned into pixels is never
                # re-decided — that is the whole point of having ruled on it.
                counts.labeled += 1
                continue
            if label is not None and label.verdict is ActorVerdict.OCCLUDED:
                counts.labeled += 1
                dirty |= _update(
                    record, resolution=ActorResolution.OCCLUDED.value
                )
                continue
            if label is not None and label.overrides_auto:
                # A manual pick with nothing cut for it yet: detection stores
                # no crop, so a freshly detected video arrives here with the
                # verdict on file and no pixels behind it.
                target = label_target(stem, record, label, masks)
                if target is not None:
                    pending.append(
                        _Pending(row, record, target, ActorResolution.MANUAL, 1)
                    )
                    continue
                counts.labeled += 1
                continue
            # Everything else runs the policy — INCLUDING a confirmed_auto
            # event with no crop. That verdict says "the automatic answer was
            # right", so computing it is what honouring the label means;
            # skipping it would leave the video's endorsed picks blank.

            xy = action.get("xy")
            context = EventContext(
                frame=int(action["frame"]),
                event_id=str(record.get("id")),
                contact=(
                    (float(xy[0]) * frame_w, float(xy[1]) * frame_h)
                    if xy and frame_w and frame_h
                    else None
                ),
                visible=bool(action.get("visible", True)),
                detections=record.get("detections") or [],
                tracks=tracks_index,
                masks=masks,
            )
            pick = policy.decide(context)
            dirty |= _update(
                record,
                association=pick.diagnostic or _ABSENT,
                candidates=pick.candidates,
            )

            if not pick.decided:
                if record.get("box") is not None:
                    counts.abstained += 1
                    dirty |= _clear(record)
                else:
                    counts.unchanged += 1
                continue

            target = _target(stem, record, pick, masks)
            if target is None:
                counts.unresolvable += 1
                dirty |= _clear(record)
                continue

            settled = _same_pick(record, target, pick, frame_w, frame_h)
            # The tracklet reference is the pick; the box is only where it lands
            # today. Written after the comparison, which reads the old one.
            dirty |= _update(
                record,
                track=pick.track.key if pick.track is not None else _ABSENT,
            )
            if settled:
                counts.unchanged += 1
                continue
            pending.append(
                _Pending(row, record, target, ActorResolution.AUTO, pick.candidates)
            )

    finally:
        if masks is not None:
            masks.close()

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
            for index, item in enumerate(
                sorted(pending, key=lambda p: p.target.frame)
            ):
                capture.set(cv2.CAP_PROP_POS_FRAMES, item.target.frame)
                ok, frame_img = capture.read()
                if not ok:
                    log.warning(
                        "Could not decode frame %s of %s; leaving the previous pick",
                        item.target.frame,
                        video_path.name,
                    )
                    continue
                if _recrop(stem, item, frame_w, frame_h, frame_img):
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
        # The outcome counts belong to whoever decided them. Detection writes
        # how many people it found; this writes how many events ended up with
        # an actor, which is what the work lists read off the header instead
        # of parsing every record.
        write_jsonl(
            path,
            {
                **meta,
                "association_policy": policy.name,
                **{
                    key: sum(
                        1 for r in records
                        if str(r.get("id")) in current
                        and r.get("status") == key
                    )
                    for key in ("ok", "multi", "miss")
                },
            },
            records,
        )
    counts.changed = changed
    counts.unchanged += len(pending) - changed
    return counts.payload()


def _is_materialized(record: dict, label) -> bool:
    """Whether this verdict already has the pixels it asks for.

    Occlusion asks for none — "nobody is the actor" is complete the moment
    the resolution says so.
    """
    if label.verdict is ActorVerdict.OCCLUDED:
        return record.get("resolution") == ActorResolution.OCCLUDED.value
    return (
        record.get("crop") is not None
        and record.get("crop_schema") == CROP_SCHEMA_VERSION
    )


def _clear(record: dict) -> bool:
    """Back to an unresolved event — the state extraction uses for "nobody"."""
    changed = _update(
        record,
        status="miss",
        box=None,
        actor_box=None,
        score=None,
        crop=None,
        crop_schema=_ABSENT,
        resolution=ActorResolution.UNRESOLVED.value,
        crop_frame=_ABSENT,
        track=_ABSENT,
        keypoints=_ABSENT,
    )
    if changed:
        record["actor_revision"] = int(record.get("actor_revision") or 0) + 1
    return changed


def _same_pick(
    record: dict,
    target: CropTarget,
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
    current = record.get("actor_box")
    if (
        current is None
        or record.get("crop_schema") != CROP_SCHEMA_VERSION
        or int(record.get("crop_frame") or record["frame"]) != target.frame
    ):
        return False
    if pick.track is not None and record.get("track") != pick.track.key:
        return False
    stored = clamp_box(target.box, frame_w, frame_h)
    return all(int(a) == int(b) for a, b in zip(current, stored))


def _recrop(stem: str, item: _Pending, frame_w: int, frame_h: int, frame_img) -> bool:
    record = item.record
    previous = record.get("crop")
    xy = record.get("xy")
    record["actor_revision"] = int(record.get("actor_revision") or 0) + 1
    crop = cut(
        record,
        frame_img,
        person_for(record, item.target),
        source_frame=item.target.frame,
        contact=(
            (float(xy[0]) * frame_w, float(xy[1]) * frame_h) if xy else None
        ),
        frame_size=(frame_w, frame_h),
        out_dir=crop_dir(stem),
        suffix=f"_p{record['actor_revision']}",
    )
    if crop is None:
        _clear(record)
        return False
    # "multi" is what the board's amber ring reads: the policy had more than
    # one plausible answer here. Flattening it to "ok" hid every ambiguous
    # pick from the person reviewing them.
    record["status"] = "ok" if item.candidates <= 1 else "multi"
    record["resolution"] = item.resolution.value
    # The crop filename carries the revision so the browser cannot serve a
    # stale image; the superseded file is derived data with no reader left.
    if previous and previous != record.get("crop"):
        (crop_dir(stem) / previous).unlink(missing_ok=True)
        (masked_crop_dir(stem) / previous).unlink(missing_ok=True)
    return True
