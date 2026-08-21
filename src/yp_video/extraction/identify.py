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
    #: Representative crops, best-first (nearest this unit's own centroid).
    crop_paths: tuple[Path, ...]


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
    exported, kept = _with_crops(stem, records, matrix, tracked, reps_per_unit)
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


def _with_crops(
    stem: str,
    records: list[dict],
    matrix,
    units,
    reps_per_unit: int,
) -> tuple[list[IdentifyUnit], list]:
    """Attach representative crops to each unit; drop the ones with none.

    Photos are scored against the UNIT's own centroid rather than a cluster's:
    a cluster is a threshold-dependent view now, and a photo has to identify
    the person whatever the slider says. Several photos of one unit is exactly
    what "I can't tell who this is" expands to, so the old spread-across-units
    pass is gone — a unit already IS one person's appearance.

    Returns the exported units and the matching `Unit` objects, in the same
    order: the caller turns the second list into the linkage leaves, so the
    two must not drift.
    """
    import numpy as np

    from yp_video.extraction.store import crop_dir

    cdir = crop_dir(stem)
    exported: list[IdentifyUnit] = []
    kept: list = []
    for unit in units:
        centroid = matrix[list(unit.rows)].mean(axis=0)
        centroid /= np.linalg.norm(centroid) + 1e-12

        scored: list[tuple[float, Path]] = []
        for row in unit.rows:
            crop = records[row].get("crop")
            if not crop:
                continue
            path = cdir / crop
            if not path.exists():
                continue
            scored.append((float(matrix[row] @ centroid), path))
        if not scored:
            continue
        scored.sort(key=lambda c: -c[0])

        seen: set[Path] = set()
        reps: list[Path] = []
        for _sim, path in scored:
            if len(reps) >= reps_per_unit:
                break
            if path in seen:
                continue
            seen.add(path)
            reps.append(path)

        exported.append(
            IdentifyUnit(
                key=unit.key,
                event_ids=tuple(unit.event_ids),
                crop_paths=tuple(reps),
            )
        )
        kept.append(unit)
    return exported, kept


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
                "crop_paths": [str(path) for path in u.crop_paths],
            }
            for u in result.units
        ],
        "linkage": [list(row) for row in result.linkage],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    _main()
