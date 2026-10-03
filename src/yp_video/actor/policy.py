"""The actor policy interface: what reassociation asks, and what it gets back.

A policy answers with a TRACKLET or a BOX, never with pixels. Turning either
into a crop needs the stored detections, the instance masks and the video,
all of which live a layer up in extraction. The one policy is the joint
person/action head (``actor/person_action.py``).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

from yp_video.tracklets.geometry import TrackRef

Box = tuple[float, float, float, float]


@dataclass(frozen=True)
class EventContext:
    """The action event a policy decides for."""

    frame: int
    #: The extraction event id — the frame is not unique, two actions can
    #: share one.
    event_id: str | None = None

    @classmethod
    def for_event(cls, record: dict, *, action: dict | None = None) -> "EventContext":
        """``record`` carries the event id; ``action`` the current frame and
        defaults to ``record`` itself."""
        action = record if action is None else action
        return cls(frame=int(action["frame"]), event_id=str(record.get("id")))


@dataclass(frozen=True)
class ActorPick:
    """A policy's answer. Both references may be absent — that is an abstention,
    which is a decision, not a failure."""

    box: Box | None = None
    track: TrackRef | None = None
    candidates: int = 0
    diagnostic: dict = field(default_factory=dict)

    @property
    def decided(self) -> bool:
        return self.box is not None or self.track is not None


class ActorPolicy(Protocol):
    """Named so its answer can be stored beside the record it produced."""

    @property
    def name(self) -> str: ...

    #: Whether the answers name tracklets, so the caller resolves them through
    #: the tracks and their masks (and refuses an untracked video up front).
    @property
    def needs_tracklets(self) -> bool: ...

    def decide(self, context: EventContext) -> ActorPick: ...
