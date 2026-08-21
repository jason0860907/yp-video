"""One video in, anonymous player clusters out — the batch identity pipeline.

Every stage already exists and answers one question (tracking: who is on
court over time; extraction: who is on each action frame; association: who
acted; embedding + clustering: who looks like whom). What did not exist is a
caller that runs them in order without a person clicking through the Film
Room — this module is that caller, plus the one genuinely new piece: an
exporter that flattens unit-level cluster labels into per-event assignments
and picks representative crops so a UI can show "this person" without a name.

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
class UnitCrop:
    """One photo of a unit, cut wide enough to place the person.

    A tight crop of a blurred player is often unidentifiable — the same body
    in the same kit from any angle. Keeping the surroundings (who they are
    next to, where on court, which side of the net) is usually what settles
    it, so the cut is the person's box grown several times over and the box
    itself rides along, normalized, for the viewer to draw.
    """

    path: Path
    #: The person's box within this image, as [x0, y0, x1, y1] in 0–1.
    box: tuple[float, float, float, float]


@dataclass(frozen=True)
class IdentifyUnit:
    """One person's crops within the video, as far as tracking can tell — a
    tracklet, or a lone action when tracking lost them.

    THE thing a jersey number gets attached to. A unit is stable for the life
    of one identify run whatever cutoff the viewer picks, which is why the
    export ships units and a tree instead of clusters: re-cutting the tree
    changes how units are grouped for display, never who a unit is.
    """

    key: str
    #: Action events this unit performed — the join back to the analysis result.
    event_ids: tuple[str, ...]
    #: Representative photos, best-first (nearest this unit's own centroid).
    crops: tuple[UnitCrop, ...]


@dataclass(frozen=True)
class IdentifyResult:
    embedder: str
    #: Leaf order IS linkage leaf order: index i in this tuple is leaf i.
    units: tuple[IdentifyUnit, ...]
    #: scipy (n-1)x4 average-linkage matrix over the unit centroids. Empty
    #: below two units, where there is nothing to merge.
    linkage: tuple[tuple[float, float, float, float], ...]
    #: The embedder's calibrated slider band — consumers must not hardcode a
    #: cosine-distance scale, it moves with every fine-tune.
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
        on_progress(_BANDS["clustering"][0], 100, "grouping players...")
    unit_links = links.track_keys(stem)
    records, matrix = identity.load_embeddings(stem, model=embedder)
    tracked = identity.build_units(records, unit_links)

    # Crops first, tree second. Units with nothing on disk are dropped, and
    # the surviving order becomes the linkage leaf order — building the tree
    # before the filter would leave every leaf index off by the drops.
    exported, kept = _with_crops(stem, video_path, records, matrix, tracked, reps_per_unit)
    unit_matrix = identity.unit_centroids(kept, matrix)
    tree = identity.linkage_tree(unit_matrix)

    if on_progress:
        on_progress(100, 100, f"{len(exported)} appearances grouped")
    return IdentifyResult(
        embedder=embedder,
        units=tuple(exported),
        linkage=tuple(tuple(float(v) for v in row) for row in tree) if tree is not None else (),
        threshold={k: float(v) for k, v in threshold_calibration(embedder).items()},
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


#: How far past the person's own box a photo reaches. Three times the box in
#: each direction puts them in their half of the court with the net and the
#: nearest team-mates in shot — enough context to tell two similar players
#: apart, without shrinking the person past recognition.
_CONTEXT_SCALE = 3.0
#: Long edge of the exported photo. The person occupies roughly a third of it,
#: so this keeps them near the 150px that stayed readable as a tight crop.
_CONTEXT_LONG_EDGE = 448


def _with_crops(
    stem: str,
    video_path: Path,
    records: list[dict],
    matrix,
    units,
    reps_per_unit: int,
) -> tuple[list[IdentifyUnit], list]:
    """Attach representative photos to each unit; drop the ones with none.

    Photos are scored against the UNIT's own centroid rather than a cluster's:
    a cluster is a threshold-dependent view now, and a photo has to identify
    the person whatever the slider says. Several photos of one unit is exactly
    what "I can't tell who this is" expands to, so the old spread-across-units
    pass is gone — a unit already IS one person's appearance.

    Each photo is re-cut from the source video around the person rather than
    reusing the tight extraction crop, and carries the person's box so a
    viewer can draw it. Frames are visited in order, since a decoder seeking
    backwards is the slow case.

    Returns the exported units and the matching `Unit` objects, in the same
    order: the caller turns the second list into the linkage leaves, so the
    two must not drift.
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

    out_dir = _context_dir(stem)
    out_dir.mkdir(parents=True, exist_ok=True)
    exported: list[IdentifyUnit] = []
    kept: list = []
    for unit in units:
        chosen = picks.get(unit.key)
        if not chosen:
            continue
        crops: list[UnitCrop] = []
        for record in chosen:
            frame = images.get(int(record["frame"]))
            if frame is None:
                continue
            cut = _context_cut(frame, record["box"])
            if cut is None:
                continue
            image, box = cut
            path = out_dir / f"{unit.key.replace(':', '_')}_{record['frame']}.jpg"
            if cv2.imwrite(str(path), image, [cv2.IMWRITE_JPEG_QUALITY, 82]):
                crops.append(UnitCrop(path=path, box=box))
        if not crops:
            continue
        exported.append(
            IdentifyUnit(key=unit.key, event_ids=tuple(unit.event_ids), crops=tuple(crops))
        )
        kept.append(unit)
    return exported, kept


def _context_dir(stem: str) -> Path:
    """Where identify's own photos live — beside the extraction crops, not
    among them: these are cut for a human to look at, not for the embedder."""
    from yp_video.config import EXTRACTION_DIR

    return EXTRACTION_DIR / "identify-context" / stem


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


def _context_cut(frame, box):
    """A wide cut around ``box``, plus the box's place inside it (0–1)."""
    import cv2

    height, width = frame.shape[:2]
    x0, y0, x1, y1 = (float(v) for v in box)
    if x1 <= x0 or y1 <= y0:
        return None
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    half_w = (x1 - x0) * _CONTEXT_SCALE / 2
    half_h = (y1 - y0) * _CONTEXT_SCALE / 2
    cx0 = max(0, int(cx - half_w))
    cy0 = max(0, int(cy - half_h))
    cx1 = min(width, int(cx + half_w))
    cy1 = min(height, int(cy + half_h))
    if cx1 - cx0 < 8 or cy1 - cy0 < 8:
        return None

    cut = frame[cy0:cy1, cx0:cx1]
    span_x, span_y = float(cx1 - cx0), float(cy1 - cy0)
    normalized = (
        max(0.0, (x0 - cx0) / span_x),
        max(0.0, (y0 - cy0) / span_y),
        min(1.0, (x1 - cx0) / span_x),
        min(1.0, (y1 - cy0) / span_y),
    )
    longest = max(cut.shape[0], cut.shape[1])
    if longest > _CONTEXT_LONG_EDGE:
        scale = _CONTEXT_LONG_EDGE / longest
        cut = cv2.resize(
            cut,
            (max(1, round(cut.shape[1] * scale)), max(1, round(cut.shape[0] * scale))),
            interpolation=cv2.INTER_AREA,
        )
    return cut, normalized


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
        "version": 2,
        "video": args.video.stem,
        "embedder": result.embedder,
        "threshold": result.threshold,
        "units": [
            {
                "key": u.key,
                "events": list(u.event_ids),
                "crops": [
                    {"path": str(c.path), "box": list(c.box)} for c in u.crops
                ],
            }
            for u in result.units
        ],
        "linkage": [list(row) for row in result.linkage],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    _main()
