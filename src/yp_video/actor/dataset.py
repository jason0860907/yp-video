"""Build immutable learned-association examples from human actor labels."""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from yp_video.actor import labels as actor_labels
from yp_video.actor.features import (
    AssociationFeatures,
    extract_features,
)
from yp_video.actor.labels import ActorVerdict
from yp_video.actor.ranking import AssociationDecision, rule_decision
from yp_video.config import REID_ANNOTATIONS_DIR
from yp_video.core.cache import StatCache
from yp_video.core.jsonl import read_jsonl_cached
from yp_video.extraction.store import RECORDS_DIR, records_path
from yp_video.actor.track_features import (
    TrackFeatures,
    candidates_near,
    extract_track_features,
)
from yp_video.person.detector import (
    iou,
    person_from_detection as _person,
)
from yp_video.tracklets.store import (
    open_track_masks,
    tracklet_index,
    tracks_masks_path,
    tracks_path,
)

MATCH_IOU = 0.5
_dataset_cache: StatCache = StatCache()


@dataclass(frozen=True)
class AssociationExample:
    stem: str
    event_id: str
    features: AssociationFeatures
    production: AssociationDecision
    #: Index into features.ranked; None is the explicit NONE/Occluded class.
    target: int | None
    verdict: ActorVerdict


@dataclass(frozen=True)
class AssociationDataset:
    examples: tuple[AssociationExample, ...]
    labels: dict[str, int]
    skipped: dict[str, int]
    sources: tuple[Path, ...]

    @property
    def stems(self) -> tuple[str, ...]:
        return tuple(sorted({example.stem for example in self.examples}))

    def payload(self) -> dict:
        return {
            "examples": len(self.examples),
            "stems": len(self.stems),
            "labels": self.labels,
            "skipped": self.skipped,
        }


def _source_paths(stems: Sequence[str]) -> list[Path]:
    directories = [REID_ANNOTATIONS_DIR, RECORDS_DIR]
    return [
        *[path for path in directories if path.exists()],
        *[
            path
            for stem in stems
            for path in (
                actor_labels.actors_path(stem),
                records_path(stem),
            )
            if path.exists()
        ],
    ]


def load_dataset(
    stems: Sequence[str] | None = None,
) -> AssociationDataset:
    """Cached dataset repository invalidated by every annotation source."""
    selected = tuple(stems) if stems is not None else tuple(
        actor_labels.labeled_stems()
    )
    sources = _source_paths(selected)
    return _dataset_cache.get(
        selected,
        sources,
        lambda: build_dataset(selected),
    )


def build_dataset(
    stems: Sequence[str] | None = None,
) -> AssociationDataset:
    """Resolve labels to detector candidates without inferring human truth."""
    selected_stems = (
        list(stems) if stems is not None else actor_labels.labeled_stems()
    )
    examples: list[AssociationExample] = []
    verdict_counts: Counter[str] = Counter()
    skipped: Counter[str] = Counter()
    sources: list[Path] = []

    for stem in selected_stems:
        record_file = records_path(stem)
        label_file = actor_labels.actors_path(stem)
        if not record_file.exists() or not label_file.exists():
            continue
        sources.extend((label_file, record_file))
        meta, records = read_jsonl_cached(record_file)
        width, height = meta.get("frame_size") or [0, 0]
        truth = actor_labels.load(stem)

        for record in records:
            event_id = str(record.get("id"))
            label = truth.get(event_id)
            if label is None:
                continue
            verdict_counts[label.verdict.value] += 1
            if not width or not height or not record.get("xy"):
                skipped["missing_contact_geometry"] += 1
                continue
            xy = record["xy"]
            people = [
                _person(detection)
                for detection in (record.get("detections") or [])
            ]
            x, y = float(xy[0]) * width, float(xy[1]) * height
            features = extract_features(people, x, y)

            if label.verdict is ActorVerdict.OCCLUDED:
                target = None
            else:
                if label.frame is not None and label.frame != record.get(
                    "frame"
                ):
                    skipped["cross_frame"] += 1
                    continue
                if label.box is None:
                    skipped["missing_truth_box"] += 1
                    continue
                detections = record.get("detections") or []
                if not detections:
                    skipped["no_detections"] += 1
                    continue
                overlaps = [
                    iou(detection["box"], list(label.box))
                    for detection in detections
                ]
                truth_index = max(
                    range(len(overlaps)), key=overlaps.__getitem__
                )
                if overlaps[truth_index] < MATCH_IOU:
                    skipped["unmatched_detection"] += 1
                    continue
                truth_box = detections[truth_index]["box"]
                target = next(
                    (
                        index
                        for index, candidate in enumerate(features.ranked)
                        if np.allclose(
                            np.asarray(candidate.person.xyxy),
                            np.asarray(truth_box),
                            atol=0.1,
                            rtol=0.0,
                        )
                    ),
                    None,
                )
                if target is None:
                    skipped["candidate_filtered"] += 1
                    continue

            examples.append(
                AssociationExample(
                    stem=stem,
                    event_id=event_id,
                    features=features,
                    production=rule_decision(people, x, y),
                    target=target,
                    verdict=label.verdict,
                )
            )

    return AssociationDataset(
        examples=tuple(examples),
        labels=dict(sorted(verdict_counts.items())),
        skipped=dict(sorted(skipped.items())),
        sources=tuple(sources),
    )


# ── Tracklet-level examples ───────────────────────────────────────
# The same learning problem asked at the unit the label now names. Two skip
# categories the box dataset needed disappear by construction: a tracklet
# spans frames, so a label anchored off the event frame is no longer
# `cross_frame`, and the candidate set is every tracklet alive rather than a
# rule-filtered subset, so `candidate_filtered` has nothing to filter.


@dataclass(frozen=True)
class TrackExample:
    stem: str
    event_id: str
    features: TrackFeatures
    #: Index into features.refs; None is the explicit NONE/Occluded class.
    target: int | None
    verdict: ActorVerdict


@dataclass(frozen=True)
class TrackDataset:
    """The tracklet twin of AssociationDataset.

    Deliberately a separate type rather than a union: ``AssociationExample``
    carries the two rule decisions the box evaluation compares against, and a
    tracklet example has no such thing. Letting one type mean both would put
    the mismatch at runtime instead of here.
    """

    examples: tuple[TrackExample, ...]
    labels: dict[str, int]
    skipped: dict[str, int]
    sources: tuple[Path, ...]

    @property
    def stems(self) -> tuple[str, ...]:
        return tuple(sorted({example.stem for example in self.examples}))

    def payload(self) -> dict:
        return {
            "examples": len(self.examples),
            "stems": len(self.stems),
            "labels": self.labels,
            "skipped": self.skipped,
        }


def build_track_dataset(stems: Sequence[str] | None = None) -> TrackDataset:
    """Tracklet examples, in the shape the training loop already consumes."""
    selected = list(stems) if stems is not None else actor_labels.labeled_stems()
    examples: list[TrackExample] = []
    verdicts: Counter[str] = Counter()
    skipped: Counter[str] = Counter()
    sources: list[Path] = []

    for stem in selected:
        record_file, label_file = records_path(stem), actor_labels.actors_path(stem)
        track_file = tracks_path(stem)
        if not (record_file.exists() and label_file.exists() and track_file.exists()):
            skipped["no_tracking"] += 1
            continue
        mask_file = tracks_masks_path(stem)
        sources.extend((label_file, record_file, track_file))
        if mask_file.exists():
            sources.append(mask_file)
        meta, records = read_jsonl_cached(record_file)
        tracks = tracklet_index(stem)
        width, height = meta.get("frame_size") or [0, 0]
        truth = actor_labels.load(stem)

        # One open archive per video; every event reads the same silhouettes.
        masks = open_track_masks(stem)
        try:
            for record in records:
                event_id = str(record.get("id"))
                label = truth.get(event_id)
                if label is None:
                    continue
                verdicts[label.verdict.value] += 1
                if not width or not height or not record.get("xy"):
                    skipped["missing_contact_geometry"] += 1
                    continue
                xy = record["xy"]
                x, y = float(xy[0]) * width, float(xy[1]) * height
                candidates = candidates_near(
                    tracks, record["frame"], masks=masks
                )
                features = extract_track_features(
                    candidates,
                    x,
                    y,
                    record["frame"],
                    detections=record.get("detections") or [],
                    visible=record.get("visible", True),
                )

                if label.verdict is ActorVerdict.OCCLUDED:
                    target = None
                elif label.track is None:
                    # A box verdict names no tracklet — it is truth about a
                    # person, not about a candidate in this list.
                    skipped["no_tracklet_label"] += 1
                    continue
                else:
                    target = next(
                        (
                            i
                            for i, ref in enumerate(features.refs)
                            if ref == label.track
                        ),
                        None,
                    )
                    if target is None:
                        skipped["target_not_alive"] += 1
                        continue

                examples.append(
                    TrackExample(stem, event_id, features, target, label.verdict)
                )
        finally:
            if masks is not None:
                masks.close()

    return TrackDataset(
        examples=tuple(examples),
        labels=dict(sorted(verdicts.items())),
        skipped=dict(sorted(skipped.items())),
        sources=tuple(sources),
    )
