"""What a video still needs before the next stage can run.

The stages depend on each other in one order — rallies, then action events
and tracklets, then extraction — but nothing enforced it: a page would let
you start a job that the server answered with a 400 two clicks later, and
some failures surfaced a whole stage away from their cause (a video with no
rally source produced an action file with ``rallies: []``, and only tracking
noticed).

This module answers "what is missing" once so every page can say it before
the click instead of after.
"""

from __future__ import annotations

from dataclasses import dataclass

from yp_video.core.jsonl import read_jsonl_header
from yp_video.core.rallies import rally_fingerprint, rally_sources
from yp_video.extraction.store import action_annotation_path, records_path
from yp_video.tracklets.store import tracks_masks_path, tracks_path

#: Ordered chain. ``blocked_on`` is the first link that is missing.
#: action and tracks are independent of each other — both need only what is
#: above them — but they are listed in the order the UI walks them.
STAGES = ("rallies", "action", "tracks", "records")


@dataclass(frozen=True)
class Prerequisites:
    rally_sources: list[str]
    has_action: bool
    has_tracks: bool
    has_masks: bool
    tracks_stale: bool
    has_records: bool

    @property
    def blocked_on(self) -> str | None:
        """The first unmet stage, or None when the chain is complete."""
        for stage, met in (
            ("rallies", bool(self.rally_sources)),
            ("action", self.has_action),
            ("tracks", self.has_tracks),
            ("records", self.has_records),
        ):
            if not met:
                return stage
        return None

    def payload(self) -> dict:
        return {
            "rally_sources": self.rally_sources,
            "has_action": self.has_action,
            "has_tracks": self.has_tracks,
            "has_masks": self.has_masks,
            "tracks_stale": self.tracks_stale,
            "has_records": self.has_records,
            "blocked_on": self.blocked_on,
        }


def _tracks_stale(stem: str) -> bool:
    """Whether the rallies moved since these tracklets were cut.

    A track key is ``"{rally_id}:{track_id}"`` and rally_id is positional, so
    re-labeling rallies renumbers them and every stored key quietly points
    somewhere else. Tracks written before the fingerprint existed report
    False — unknown, not stale; claiming otherwise would flag every old video.
    """
    path = tracks_path(stem)
    if not path.exists():
        return False
    stored = (read_jsonl_header(path).get("rallies") or {}).get("fingerprint")
    if not stored:
        return False
    return stored != rally_fingerprint(stem)


def prerequisites(stem: str) -> Prerequisites:
    """Everything the pipeline pages need to know about one video."""
    return Prerequisites(
        rally_sources=rally_sources(stem),
        has_action=action_annotation_path(stem) is not None,
        has_tracks=tracks_path(stem).exists(),
        has_masks=tracks_masks_path(stem).exists(),
        tracks_stale=_tracks_stale(stem),
        has_records=records_path(stem).exists(),
    )
