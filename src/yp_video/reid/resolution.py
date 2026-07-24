"""Actor-resolution domain state shared by extraction and API projection."""

from __future__ import annotations

from enum import Enum
from typing import Mapping


class ActorResolution(str, Enum):
    """How the actor associated with one action event was resolved."""

    UNRESOLVED = "unresolved"
    AUTO = "auto"
    MANUAL = "manual"
    OCCLUDED = "occluded"


def actor_resolution(record: Mapping[str, object]) -> ActorResolution:
    """Return explicit state, normalizing records created before it existed."""
    raw = record.get("resolution")
    try:
        return ActorResolution(raw)
    except (TypeError, ValueError):
        if record.get("box_source") == "manual":
            return (
                ActorResolution.MANUAL
                if record.get("crop")
                else ActorResolution.OCCLUDED
            )
        return ActorResolution.AUTO if record.get("crop") else ActorResolution.UNRESOLVED
