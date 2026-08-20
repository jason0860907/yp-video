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
class PlayerCluster:
    """One appearance cluster — a person, as far as the embedder can tell."""

    id: str
    #: How many action events this cluster's units cover.
    count: int
    #: Representative crops, best-first (closest to the cluster centroid,
    #: spread across rallies). Absolute paths into EXTRACTION_DIR.
    crop_paths: tuple[Path, ...]


@dataclass(frozen=True)
class ClusterVariant:
    """One granularity of the same embeddings.

    The calibrated per-embedder default is a corpus-level compromise; the
    best cutoff drifts per video (a session with two similar kits merges at
    the default), and re-cutting the linkage costs milliseconds against the
    minutes the GPU stages took. So every run ships three granularities and
    the picker UI lets the user flip between them instead of re-running.
    """

    id: str  # "coarse" | "default" | "fine"
    threshold: float
    clusters: tuple[PlayerCluster, ...]
    #: event id ("f<frame>" / "act_…") → cluster id.
    event_assignments: dict[str, str]


@dataclass(frozen=True)
class IdentifyResult:
    embedder: str
    #: Coarse → fine, with the calibrated default in the middle.
    variants: tuple[ClusterVariant, ...]


def _variant_thresholds(embedder: str) -> dict[str, float]:
    """Coarse / default / fine cutoffs from the embedder's calibrated band.

    Coarse is the band's max (merge-happy), default the calibrated peak, fine
    the midpoint between the peak and the band's min (split-happy) — far
    enough to matter, not so far that every unit is its own cluster.
    """
    band = threshold_calibration(embedder)
    default = float(band["default"])
    return {
        "coarse": float(band["max"]),
        "default": default,
        "fine": round((default + float(band["min"])) / 2, 4),
    }


def identify_players(
    video_path: Path,
    *,
    embedder: str = DEFAULT_EMBEDDER,
    association_checkpoint: Path | None = None,
    tracking_stride: int = 1,
    reps_per_cluster: int = 3,
    on_progress: ProgressFn | None = None,
) -> IdentifyResult:
    """Track → detect → associate → embed → cluster one video, end to end.

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
        on_progress(_BANDS["clustering"][0], 100, "clustering players...")
    unit_links = links.track_keys(stem)
    records, matrix = identity.load_embeddings(stem, model=embedder)
    units, unit_matrix = identity.unit_embeddings(records, matrix, unit_links)

    variants: list[ClusterVariant] = []
    for variant_id, threshold in _variant_thresholds(embedder).items():
        labels = identity.cluster(unit_matrix, threshold=threshold)
        clusters, assignments = _export(
            stem, records, matrix, units, unit_matrix, labels, reps_per_cluster
        )
        variants.append(
            ClusterVariant(
                id=variant_id,
                threshold=threshold,
                clusters=tuple(clusters),
                event_assignments=assignments,
            )
        )
    if on_progress:
        default = next(v for v in variants if v.id == "default")
        on_progress(100, 100, f"{len(default.clusters)} players found")
    return IdentifyResult(embedder=embedder, variants=tuple(variants))


def _banded(on_progress: ProgressFn | None, phase: str) -> ProgressFn | None:
    """A stage's (done, total, msg) mapped into the overall percent band."""
    if on_progress is None:
        return None
    lo, hi = _BANDS[phase]

    def cb(done: int, total: int, msg: str) -> None:
        fraction = done / total if total else 1.0
        on_progress(int(round(lo + (hi - lo) * fraction)), 100, f"{phase} · {msg}")

    return cb


def _export(
    stem: str,
    records: list[dict],
    matrix,
    units,
    unit_matrix,
    labels,
    reps_per_cluster: int,
) -> tuple[list[PlayerCluster], dict[str, str]]:
    """Unit-level cluster labels → per-event assignments + representative crops.

    Representatives are the crops most similar to the cluster centroid,
    greedily spread across units (a person seen in several rallies should be
    recognisable from any of them, and three near-identical frames of one
    swing prove nothing). Clusters whose every crop is missing on disk are
    dropped — there is nothing to show a user.
    """
    import numpy as np

    from yp_video.extraction.store import crop_dir

    cdir = crop_dir(stem)
    by_cluster: dict[int, list[int]] = {}
    for i, label in enumerate(labels):
        by_cluster.setdefault(int(label), []).append(i)

    clusters: list[PlayerCluster] = []
    assignments: dict[str, str] = {}
    for label, unit_indexes in sorted(by_cluster.items()):
        cluster_id = f"c{label}"
        cluster_units = [units[i] for i in unit_indexes]
        for unit in cluster_units:
            for event_id in unit.event_ids:
                assignments[event_id] = cluster_id

        centroid = unit_matrix[unit_indexes].mean(axis=0)
        centroid /= np.linalg.norm(centroid) + 1e-12

        # Candidate crops: every record row in the cluster, scored against the
        # centroid; one representative per unit before a second from any.
        candidates: list[tuple[float, str, Path]] = []
        for unit in cluster_units:
            for row in unit.rows:
                crop = records[row].get("crop")
                if not crop:
                    continue
                path = cdir / crop
                if not path.exists():
                    continue
                sim = float(matrix[row] @ centroid)
                candidates.append((sim, unit.key, path))
        candidates.sort(key=lambda c: -c[0])

        reps: list[Path] = []
        seen_units: set[str] = set()
        for _pass in ("spread", "fill"):
            for sim, unit_key, path in candidates:
                if len(reps) >= reps_per_cluster:
                    break
                if path in reps or (_pass == "spread" and unit_key in seen_units):
                    continue
                reps.append(path)
                seen_units.add(unit_key)
        if not reps:
            for unit in cluster_units:
                for event_id in unit.event_ids:
                    assignments.pop(event_id, None)
            continue

        clusters.append(
            PlayerCluster(
                id=cluster_id,
                count=sum(len(u.event_ids) for u in cluster_units),
                crop_paths=tuple(reps),
            )
        )
    return clusters, assignments


def _main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--assoc-checkpoint", type=Path, default=None)
    parser.add_argument("--embedder", default=DEFAULT_EMBEDDER)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--reps", type=int, default=3)
    args = parser.parse_args()

    def report(done: int, total: int, msg: str) -> None:
        print(f"PROGRESS {done} {total} {msg}", flush=True)

    result = identify_players(
        args.video,
        embedder=args.embedder,
        association_checkpoint=args.assoc_checkpoint,
        tracking_stride=args.stride,
        reps_per_cluster=args.reps,
        on_progress=report,
    )
    payload = {
        "version": 1,
        "video": args.video.stem,
        "embedder": result.embedder,
        "variants": [
            {
                "id": v.id,
                "threshold": v.threshold,
                "clusters": [
                    {
                        "id": c.id,
                        "count": c.count,
                        "crop_paths": [str(p) for p in c.crop_paths],
                    }
                    for c in v.clusters
                ],
                "event_assignments": v.event_assignments,
            }
            for v in result.variants
        ],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    _main()
