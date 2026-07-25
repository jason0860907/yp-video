"""How one extraction record's actor was resolved.

The machine-side counterpart to a human label (see actor/labels.py): the
label says what the user decided, this says what the record ended up with.
Extraction always writes it explicitly — there is deliberately no inference
from "has a crop" or "was manually picked", because a state derivable two
ways eventually gets derived two different ways.
"""

from __future__ import annotations

from enum import Enum
from typing import Mapping


class ActorResolution(str, Enum):
    """How the actor associated with one action event was resolved."""

    #: Nobody attached: no candidate, or an event still awaiting a pick.
    UNRESOLVED = "unresolved"
    #: The association policy picked this person.
    AUTO = "auto"
    #: A human picked this person.
    MANUAL = "manual"
    #: A human ruled that nobody in frame is the actor.
    OCCLUDED = "occluded"


def actor_resolution(record: Mapping[str, object]) -> ActorResolution:
    """The record's explicit state, or an actionable error."""
    raw = record.get("resolution")
    try:
        return ActorResolution(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"Record {record.get('id')!r} carries no actor resolution "
            f"({raw!r}) — re-run extraction for this video"
        ) from exc
