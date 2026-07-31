"""Which policy decides the actor, as a value rather than an import.

Extraction had exactly one answer to "who performed this action" — the rule,
called inline. There are two now and there will be more the moment someone
trains one, so the choice has to be something a caller can pass, name, and
store next to its output.

A policy answers with a TRACKLET or a BOX, never with pixels. Turning either
into a crop needs the stored detections, the instance masks and the video,
all of which live a layer up in extraction — keeping that out of here is what
lets a policy be evaluated on a laptop with no video file in sight.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol, Sequence

import numpy as np

from yp_video.actor.ranking import RULE_BASED, DecisionReason, rule_decision
from yp_video.core.progress import ProgressFn
from yp_video.person.detector import person_from_detection
from yp_video.tracklets.geometry import TrackletIndex, TrackRef

Box = tuple[float, float, float, float]


def contact_point(
    xy: Sequence[float] | None, width: float, height: float
) -> tuple[float, float] | None:
    """The annotated contact point in pixels, or None without full geometry.

    The one definition: a normalized point with no frame size is no point at
    all. Four call sites used to inline this with three different None checks,
    one of which forgot the frame-size half.
    """
    if not xy or not width or not height:
        return None
    return (float(xy[0]) * width, float(xy[1]) * height)


@dataclass(frozen=True)
class EventContext:
    """Everything a policy may look at for one action event."""

    frame: int
    #: The annotated contact point in pixels, or None when the event has none.
    contact: tuple[float, float] | None
    #: Whether the BALL was visible. An action label says nothing about
    #: whether the PLAYER was — that is the association layer's `occluded`
    #: verdict, and the two are unrelated.
    visible: bool
    #: The extraction event id. A policy that reads a precomputed answer needs
    #: to look it up by something, and the frame is not unique — two actions
    #: can share one.
    event_id: str | None = None
    detections: Sequence[dict] = ()
    #: The video's tracklets, indexed (see tracklets/geometry.TrackletIndex).
    #: Built once per video by the caller — a policy asks it per event, and
    #: the raw list answers "who is near frame N" only by scanning all of it.
    tracks: TrackletIndex | None = None
    #: Tracklet key → that tracklet's silhouettes for the whole video. Absent
    #: on a video tracked before masks existed; a policy that wants outlines
    #: degrades rather than refuses, since the boxes still say where everyone
    #: is.
    masks: Mapping[str, np.ndarray | None] | None = None

    @property
    def contact_usable(self) -> bool:
        """Whether the annotated contact point may be trusted as evidence.

        Two states, neither a missing value: no point at all, and a point on
        an event whose BALL was not visible — annotated from memory, and in
        practice sitting at the frame edge where the ball left. A policy that
        ranks people by distance to it would be ranking them by a guess.

        This says nothing about whether the ACTOR is on screen; a policy that
        does not read the contact point has no business consulting it.
        """
        return self.contact is not None and self.visible

    @classmethod
    def for_event(
        cls,
        record: dict,
        *,
        width: float,
        height: float,
        action: dict | None = None,
        tracks: TrackletIndex | None = None,
        masks: "Mapping[str, np.ndarray | None] | None" = None,
    ) -> "EventContext":
        """The context for one labelable event.

        ``record`` carries the extraction-side fields (id, detections);
        ``action`` the action-side ones (frame, xy, visible) and defaults to
        ``record`` itself — reassociation is the caller that joins a fresh
        action row onto a stored record.
        """
        action = record if action is None else action
        return cls(
            frame=int(action["frame"]),
            event_id=str(record.get("id")),
            contact=contact_point(action.get("xy"), width, height),
            visible=bool(action.get("visible", True)),
            detections=record.get("detections") or [],
            tracks=tracks,
            masks=masks,
        )


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

    #: Whether decide() needs EventContext.tracks filled. The caller uses
    #: this to refuse the job up front instead of silently abstaining on every
    #: event of a video that was never tracked.
    @property
    def needs_tracklets(self) -> bool: ...

    def decide(self, context: EventContext) -> ActorPick: ...


class PolicyPlan(Protocol):
    """A policy a caller can name and refuse before it exists.

    Some policies decide per VIDEO before they can decide per event — the
    yp-spot head runs one subprocess over the frames and only then has an
    answer for anything. A caller still has to name the job and reject an
    untracked video BEFORE that runs, so those two facts are the plan, and
    ``build`` is where the expensive part happens.

    Every ordinary policy is its own plan (see ImmediatePolicy), so a caller
    holds one kind of thing rather than branching on which it got.
    """

    @property
    def name(self) -> str: ...

    @property
    def needs_tracklets(self) -> bool: ...

    def build(
        self, video: Path, on_progress: ProgressFn | None = None
    ) -> ActorPolicy: ...


class ImmediatePolicy:
    """A policy that needs nothing from the video, so it is its own plan."""

    def build(
        self, video: Path, on_progress: ProgressFn | None = None
    ) -> ActorPolicy:
        return self  # type: ignore[return-value]


class RulePolicy(ImmediatePolicy):
    """The geometric rule production has always run on."""

    name = RULE_BASED
    needs_tracklets = False

    def decide(self, context: EventContext) -> ActorPick:
        if not context.contact_usable:
            return ActorPick()
        assert context.contact is not None
        x, y = context.contact
        people = [person_from_detection(d) for d in context.detections]
        decision = rule_decision(people, x, y)
        return ActorPick(
            box=decision.selected.xyxy if decision.selected else None,
            candidates=len(decision.ranked),
            diagnostic=decision.diagnostic(),
        )


class SpotActorPolicy(ImmediatePolicy):
    """The yp-spot actor head's choice, read back per event.

    The model does not run here. It needs the frame pixels and a GPU, both of
    which live behind a subprocess in the other repo, so association reads the
    answers it already produced — running that subprocess is SpotPlan's job.

    ``needs_tracklets`` is True even though this policy never inspects them:
    the answer NAMES a tracklet, so a video without tracking cannot receive
    one, and the caller should refuse the job up front rather than abstain on
    every event.

    The one policy that ignores ``contact_usable``: it looked at the frames,
    not at the annotated point, so a ball nobody could see costs it nothing.
    Refusing those events here would throw away an answer the model already
    gave — and leave them unconfirmable forever, since only a diagnostic makes
    an unresolved event endorsable.
    """

    needs_tracklets = True

    def __init__(self, stem_answers, name: str = "spot"):
        self._answers = stem_answers
        self._name = name

    @property
    def name(self) -> str:
        return f"spot:{self._name}"

    def decide(self, context: EventContext) -> ActorPick:
        if context.event_id is None:
            return ActorPick()
        answer = self._answers.get(context.event_id)
        if answer is None:
            return ActorPick()
        return ActorPick(
            track=answer.track,
            candidates=len(context.tracks) if context.tracks is not None else 0,
            diagnostic={
                "version": self.name,
                "decision": (
                    DecisionReason.SELECTED.value
                    if answer.track is not None
                    else DecisionReason.ABSTAINED.value
                ),
                "kind": answer.kind,
                "confidence": round(answer.confidence, 4),
            },
        )


class SpotPlan:
    """The yp-spot actor head, which cannot answer until it has seen the video.

    Scoring is one subprocess over the frames per VIDEO, so the per-event
    policy does not exist until that has run. What a caller needs beforehand
    is here: a name for the job card, and the fact that the answer NAMES a
    tracklet — so an untracked video is refused up front instead of abstaining
    on every event of it.
    """

    needs_tracklets = True

    def __init__(self, checkpoint: Path):
        self._checkpoint = checkpoint

    @property
    def name(self) -> str:
        return f"spot:{self._checkpoint.parent.name}"

    def build(
        self, video: Path, on_progress: ProgressFn | None = None
    ) -> ActorPolicy:
        # Deferred: scoring pulls in the action package and its checkpoint
        # plumbing, which nothing else in this module needs at import time.
        from yp_video.actor import spot_associate  # noqa: PLC0415

        answers = spot_associate.run(
            video, self._checkpoint, on_progress=on_progress
        )
        return SpotActorPolicy(answers, name=self._checkpoint.parent.name)
