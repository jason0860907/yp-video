"""One video in, anonymous player-pairing suggestions out.

Every stage already exists and answers one question (tracking: who is on
court over time; extraction: who is on each action frame; association: who
acted; embedding + clustering: who looks like whom). What did not exist is a
caller that runs them in order without a person clicking through the Film
Room — this module is that caller, plus the one genuinely new piece: an
exporter that groups unit-level appearances at the model's calibrated cutoff
and picks representative full frames so a UI can confirm who each person is.

Runs against the VIDEOS_DIR layout like everything else. A caller that wants
an isolated run (the selfhost worker) stages a minimal layout in a scratch
directory and points ``YP_VIDEOS_DIR`` at it in a subprocess — the same
process-boundary pattern yp-spot and yp-reid already use — with
``YP_REID_CHECKPOINTS_DIR`` kept on the real checkpoint store.

This orchestration belongs in the extraction roof because it is the only layer
allowed to combine tracking, actor association, extraction and ReID. The
``__main__`` CLI is the subprocess entry point. It reports progress as
``PROGRESS <percent> 100 <message>`` lines on stdout, one phase-weighted
number so the caller needs no knowledge of the stages.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

from yp_video.core.progress import ProgressFn
from yp_video.reid.embedder import DEFAULT_EMBEDDER, threshold_calibration

#: Overall-percent band per phase, tuned to measured cost: dense tracking is
#: the GPU bill (~14.5 ms/frame over every rally frame); everything after it
#: is seconds to a few minutes.
_BANDS = {
    "tracking": (0, 62),
    "detecting": (62, 68),
    "associating": (68, 80),
    "embedding": (80, 94),
    "clustering": (94, 99),
}


@dataclass(frozen=True)
class UnitImage:
    """One complete source-video frame with this appearance boxed."""

    path: Path
    #: The person's box within this image, as [x0, y0, x1, y1] in 0–1.
    box: tuple[float, float, float, float]


@dataclass(frozen=True)
class _UnitFrames:
    key: str
    event_ids: tuple[str, ...]
    images: tuple[UnitImage, ...]


@dataclass(frozen=True)
class IdentifyUnit:
    """One person's appearances within the video, as far as tracking can tell — a
    tracklet, or a lone action when tracking lost them.

    THE thing a jersey number gets attached to. The linkage tree groups units
    for display; correcting one member never changes any other unit.
    """

    key: str
    #: Action events this unit performed — the join back to the analysis result.
    event_ids: tuple[str, ...]
    #: Representative photos, best-first (nearest this unit's own centroid).
    images: tuple[UnitImage, ...]


@dataclass(frozen=True)
class IdentifyResult:
    embedder: str
    units: tuple[IdentifyUnit, ...]
    #: scipy average-linkage tree over unit centroids. The app cuts this tree
    #: locally while the user moves the suggestion-granularity slider.
    linkage: tuple[tuple[float, float, float, float], ...]
    #: Calibrated distance band for the active embedder.
    threshold: dict[str, float]


def identify_players(
    video_path: Path,
    *,
    embedder: str = DEFAULT_EMBEDDER,
    association_checkpoint: Path | None = None,
    tracking_stride: int = 1,
    reps_per_unit: int = 3,
    on_progress: ProgressFn | None = None,
) -> IdentifyResult:
    """Track → detect → associate → embed → group one video, end to end.

    Prerequisites on disk (the same ones the Film Room stages require): rally
    spans (core/rallies.py) and an action annotation file
    (extraction/store.action_annotation_path). Raises when either is missing.

    ``association_checkpoint`` selects the yp-spot actor head; ``None`` falls
    back to the geometric rule policy, which needs no model but picks the
    wrong player more often.
    """
    # Deferred imports: this module is also imported for its dataclasses by
    # code that must not pull the GPU stack in.
    from yp_video.actor.policy import RulePolicy, SpotPlan
    from yp_video.extraction import links
    from yp_video.extraction.pipeline import detect_video, embed_video, load_events
    from yp_video.extraction.reassociate import reassociate_video
    from yp_video.reid import identity
    from yp_video.tracklets.tracking import track_video

    stem = video_path.stem
    events = load_events(stem)
    if not events:
        raise ValueError(f"No action events for {stem} — run Action Predict first")

    track_video(
        video_path,
        stride=tracking_stride,
        event_frames={int(e["frame"]) for e in events},
        on_progress=_banded(on_progress, "tracking"),
    )
    detect_video(video_path, on_progress=_banded(on_progress, "detecting"))

    plan = SpotPlan(association_checkpoint) if association_checkpoint else RulePolicy()
    associate_cb = _banded(on_progress, "associating")
    policy = plan.build(video_path, on_progress=associate_cb)
    reassociate_video(video_path, policy, on_progress=associate_cb)

    embed_video(stem, models=[embedder], on_progress=_banded(on_progress, "embedding"))

    if on_progress:
        on_progress(_BANDS["clustering"][0], 100, "creating pairing suggestions...")
    unit_links = links.track_keys(stem)
    records, matrix = identity.load_embeddings(stem, model=embedder)
    tracked = identity.build_units(records, unit_links)

    # Photos first, suggestions second. Units with nothing on disk are dropped
    # before grouping so the fixed suggestion ids describe only shipped units.
    exported, kept = _with_images(stem, video_path, records, matrix, tracked, reps_per_unit)
    unit_matrix = identity.unit_centroids(kept, matrix)
    calibration = threshold_calibration(embedder)
    tree = identity.linkage_tree(unit_matrix)
    identified = tuple(
        IdentifyUnit(
            key=unit.key,
            event_ids=unit.event_ids,
            images=unit.images,
        )
        for unit in exported
    )

    if on_progress:
        on_progress(100, 100, f"{len(identified)} appearances suggested")
    return IdentifyResult(
        embedder=embedder,
        units=identified,
        linkage=tuple(tuple(float(value) for value in row) for row in tree)
        if tree is not None
        else (),
        threshold={key: float(value) for key, value in calibration.items()},
    )


def _banded(on_progress: ProgressFn | None, phase: str) -> ProgressFn | None:
    """A stage's (done, total, msg) mapped into the overall percent band."""
    if on_progress is None:
        return None
    lo, hi = _BANDS[phase]

    def cb(done: int, total: int, msg: str) -> None:
        fraction = done / total if total else 1.0
        on_progress(int(round(lo + (hi - lo) * fraction)), 100, f"{phase} · {msg}")

    return cb


#: Full frames stay cheap to upload while preserving their exact source ratio.
_FRAME_LONG_EDGE = 448


def _with_images(
    stem: str,
    video_path: Path,
    records: list[dict],
    matrix,
    units,
    reps_per_unit: int,
) -> tuple[list[_UnitFrames], list]:
    """Attach representative photos to each unit; drop the ones with none.

    Photos are scored against the UNIT's own centroid, so each independently
    correctable appearance gets its own best evidence. Several photos of one
    unit is exactly what "I can't tell who this is" expands to.

    Each photo is the complete source-video frame, downscaled proportionally,
    and carries the person's box normalized against that complete frame.
    Frames are visited in order, since a decoder seeking backwards is the slow
    case.

    Returns the exported units and matching `Unit` objects in the same order,
    so linkage leaf indices stay aligned with the shipped appearances.
    """
    import cv2
    import numpy as np

    # Pick first, decode second: one ordered pass over the frames we settled
    # on beats seeking per unit.
    picks: dict[str, list[dict]] = {}
    wanted: set[int] = set()
    for unit in units:
        centroid = matrix[list(unit.rows)].mean(axis=0)
        centroid /= np.linalg.norm(centroid) + 1e-12
        scored = [
            (float(matrix[row] @ centroid), records[row])
            for row in unit.rows
            if records[row].get("box") and records[row].get("frame") is not None
        ]
        if not scored:
            continue
        scored.sort(key=lambda c: -c[0])
        chosen = [record for _sim, record in scored[:reps_per_unit]]
        picks[unit.key] = chosen
        wanted.update(int(record["frame"]) for record in chosen)

    images = _decode_frames(video_path, wanted)

    out_dir = _image_dir(stem)
    out_dir.mkdir(parents=True, exist_ok=True)
    exported: list[_UnitFrames] = []
    kept: list = []
    for unit in units:
        chosen = picks.get(unit.key)
        if not chosen:
            continue
        images_for_unit: list[UnitImage] = []
        for record in chosen:
            frame = images.get(int(record["frame"]))
            if frame is None:
                continue
            rendered = _full_frame(frame, record["box"])
            if rendered is None:
                continue
            image, box = rendered
            path = out_dir / f"{unit.key.replace(':', '_')}_{record['frame']}.jpg"
            if cv2.imwrite(str(path), image, [cv2.IMWRITE_JPEG_QUALITY, 82]):
                images_for_unit.append(UnitImage(path=path, box=box))
        if not images_for_unit:
            continue
        exported.append(
            _UnitFrames(
                key=unit.key,
                event_ids=tuple(unit.event_ids),
                images=tuple(images_for_unit),
            )
        )
        kept.append(unit)
    return exported, kept


def _image_dir(stem: str) -> Path:
    """Where identify's complete human-review frames live."""
    from yp_video.config import EXTRACTION_DIR

    return EXTRACTION_DIR / "identify-frames" / stem


def _decode_frames(video_path: Path, wanted: set[int]) -> dict:
    """The requested frames, read in ascending order."""
    import cv2

    images: dict = {}
    if not wanted:
        return images
    capture = cv2.VideoCapture(str(video_path))
    try:
        for index in sorted(wanted):
            capture.set(cv2.CAP_PROP_POS_FRAMES, index)
            ok, frame = capture.read()
            if ok:
                images[index] = frame
    finally:
        capture.release()
    return images


def _full_frame(frame, box):
    """Downscale a complete frame and normalize a valid, clamped person box."""
    import cv2

    height, width = frame.shape[:2]
    if width <= 0 or height <= 0:
        return None
    try:
        values = tuple(float(v) for v in box)
    except (TypeError, ValueError):
        return None
    if len(values) != 4 or not all(math.isfinite(value) for value in values):
        return None
    x0, y0, x1, y1 = values
    x0, x1 = max(0.0, x0), min(float(width), x1)
    y0, y1 = max(0.0, y0), min(float(height), y1)
    if x1 <= x0 or y1 <= y0:
        return None
    normalized = (
        x0 / width,
        y0 / height,
        x1 / width,
        y1 / height,
    )
    image = frame
    longest = max(height, width)
    if longest > _FRAME_LONG_EDGE:
        scale = _FRAME_LONG_EDGE / longest
        image = cv2.resize(
            frame,
            (max(1, round(width * scale)), max(1, round(height * scale))),
            interpolation=cv2.INTER_AREA,
        )
    return image, normalized


def _main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--assoc-checkpoint", type=Path, default=None)
    parser.add_argument("--embedder", default=DEFAULT_EMBEDDER)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--reps-per-unit", type=int, default=3)
    args = parser.parse_args()

    def report(done: int, total: int, msg: str) -> None:
        print(f"PROGRESS {done} {total} {msg}", flush=True)

    result = identify_players(
        args.video,
        embedder=args.embedder,
        association_checkpoint=args.assoc_checkpoint,
        tracking_stride=args.stride,
        reps_per_unit=args.reps_per_unit,
        on_progress=report,
    )
    payload = {
        "version": 4,
        "video": args.video.stem,
        "embedder": result.embedder,
        "threshold": result.threshold,
        "linkage": [list(row) for row in result.linkage],
        "units": [
            {
                "key": u.key,
                "events": list(u.event_ids),
                "images": [
                    {"path": str(image.path), "box": list(image.box)}
                    for image in u.images
                ],
            }
            for u in result.units
        ],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    _main()
