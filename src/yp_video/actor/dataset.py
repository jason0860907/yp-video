"""Build immutable learned-association examples from human actor labels.

One example per reviewed event: the tracklets alive around it, reduced to
features, and the index of the one the human named. The unit is the TRACKLET
because that is what a verdict names and what a policy must answer with — a
box-level twin of this file existed and was retired with the box ranker.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from yp_video.actor import labels as actor_labels
from yp_video.actor.labels import ActorVerdict
from yp_video.actor.policy import contact_point
from yp_video.actor.review import open_video_events
from yp_video.actor.track_features import (
    TrackFeatures,
    candidates_near,
    extract_track_features,
)
from yp_video.config import REID_ANNOTATIONS_DIR
from yp_video.core.cache import StatCache
from yp_video.extraction.store import (
    RECORDS_DIR,
    action_source_paths,
    records_path,
)
from yp_video.tracklets.store import (
    tracks_masks_path,
    tracks_path,
)

_dataset_cache: StatCache = StatCache()


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
    """One training corpus, and a statement of what it could not use.

    ``skipped`` is part of the value, not a log line: an event dropped for
    `no_tracklet_label` or `target_not_alive` is a labelled event this model
    will never see, and the size of that number decides whether a metric
    computed on the rest means anything.
    """

    examples: tuple[TrackExample, ...]
    labels: dict[str, int]
    skipped: dict[str, int]
    #: Targets that came from resolving a labelled BOX to the tracklet it
    #: overlaps, rather than from a human naming that tracklet. Reported
    #: because it is inferred truth: the resolution is production's own, but
    #: a corpus that is mostly inferred deserves to say so out loud.
    resolved_from_box: int
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
            "resolved_from_box": self.resolved_from_box,
        }


def build_track_dataset(stems: Sequence[str] | None = None) -> TrackDataset:
    """Tracklet examples, in the shape the training loop already consumes."""
    selected = list(stems) if stems is not None else actor_labels.labeled_stems()
    examples: list[TrackExample] = []
    verdicts: Counter[str] = Counter()
    skipped: Counter[str] = Counter()
    sources: list[Path] = []
    resolved = 0

    for stem in selected:
        record_file, label_file = records_path(stem), actor_labels.actors_path(stem)
        track_file = tracks_path(stem)
        if not (record_file.exists() and label_file.exists() and track_file.exists()):
            skipped["no_tracking"] += 1
            continue
        mask_file = tracks_masks_path(stem)
        sources.extend((label_file, record_file, track_file))
        sources.extend(action_source_paths(stem))
        if mask_file.exists():
            sources.append(mask_file)

        # One open archive per video; every event reads the same silhouettes.
        with open_video_events(stem) as video:
            for record in video.records:
                event_id = str(record.get("id"))
                label = video.verdicts.get(event_id)
                if label is None:
                    continue
                verdicts[label.verdict.value] += 1
                contact = contact_point(
                    record.get("xy"), video.width, video.height
                )
                if contact is None:
                    skipped["missing_contact_geometry"] += 1
                    continue
                x, y = contact
                candidates = candidates_near(
                    video.tracks, record["frame"], masks=video.masks
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
                else:
                    named = label.track
                    if named is None:
                        # A box verdict names a PERSON, not a candidate in
                        # this list. Resolving it by overlap is not a liberty:
                        # the same step turns the rule's box into a tracklet
                        # in production (links.py) and in scoring
                        # (evaluate.as_track), so dropping it here was the
                        # anomaly. Confirm snapshots still store the rule's
                        # box with no track key (labels.confirmations_for),
                        # so this path is what keeps every future bulk
                        # confirmation usable as a positive example.
                        named = video.tracks.at_box(
                            label.frame
                            if label.frame is not None
                            else record["frame"],
                            label.box,
                        )
                        if named is None:
                            skipped["unresolved_box_label"] += 1
                            continue
                        resolved += 1
                    target = next(
                        (
                            i
                            for i, ref in enumerate(features.refs)
                            if ref == named
                        ),
                        None,
                    )
                    if target is None:
                        skipped["target_not_alive"] += 1
                        continue

                examples.append(
                    TrackExample(stem, event_id, features, target, label.verdict)
                )

    return TrackDataset(
        examples=tuple(examples),
        labels=dict(sorted(verdicts.items())),
        skipped=dict(sorted(skipped.items())),
        resolved_from_box=resolved,
        sources=tuple(sources),
    )



def source_paths(stems: Sequence[str]) -> list[Path]:
    """Every file a rebuild would read, for cache invalidation.

    Built independently of ``build_track_dataset`` because the cache has to
    know what to watch BEFORE paying to build. A path that does not exist is
    left out: StatCache keys on the stat of what it was given, and a missing
    tracks file becoming present must invalidate through the directory entry
    rather than silently keeping a stale dataset.
    """
    directories = [REID_ANNOTATIONS_DIR, RECORDS_DIR]
    return [
        *[path for path in directories if path.exists()],
        *[
            path
            for stem in stems
            for path in (
                actor_labels.actors_path(stem),
                records_path(stem),
                tracks_path(stem),
                tracks_masks_path(stem),
                *action_source_paths(stem),
            )
            if path.exists()
        ],
    ]


def load_track_dataset(
    stems: Sequence[str] | None = None,
) -> TrackDataset:
    """Cached tracklet dataset, invalidated by every annotation source.

    Building one decompresses a silhouette archive per video, so the page that
    merely wants to SHOW the corpus size must not pay for it twice.
    """
    selected = (
        tuple(stems)
        if stems is not None
        else tuple(actor_labels.labeled_stems())
    )
    return _dataset_cache.get(
        selected,
        source_paths(selected),
        lambda: build_track_dataset(selected),
    )
