"""The actor policy interface: what reassociation asks, and what it gets back.

A policy answers with a TRACKLET, never with pixels. Turning it into a crop
needs the stored detections, the instance masks and the video,
all of which live a layer up in extraction. The one policy is the joint
person/action head (``actor/person_action.py``).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

from yp_video.tracklets.geometry import TrackRef


@dataclass(frozen=True)
class ActorPick:
    """A policy's answer. The tracklet may be absent — that is an abstention,
    which is a decision, not a failure."""

    track: TrackRef | None = None
    candidates: int = 0
    diagnostic: dict = field(default_factory=dict)

    @property
    def decided(self) -> bool:
        return self.track is not None


class ActorPolicy(Protocol):
    """Named so its answer can be stored beside the record it produced."""

    @property
    def name(self) -> str: ...

    def decide(self, event_id: str) -> ActorPick: ...
