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
from typing import Protocol, Sequence

import numpy as np

from yp_video.actor.features import extract_features
from yp_video.actor.model import FEATURE_SET_TRACK, AssociationModel
from yp_video.actor.ranking import DecisionReason, RULE_BASED, rule_decision
from yp_video.actor.track_features import (
    candidates_near,
    extract_track_features,
)
from yp_video.person.detector import person_from_detection
from yp_video.tracklets.geometry import TrackRef

Box = tuple[float, float, float, float]


@dataclass(frozen=True)
class EventContext:
    """Everything a policy may look at for one action event."""

    frame: int
    #: The annotated contact point in pixels, or None when the event has none.
    contact: tuple[float, float] | None
    visible: bool
    detections: Sequence[dict] = ()
    tracklets: Sequence[dict] = ()
    #: Tracklet key → that tracklet's silhouettes for the whole video. Absent
    #: on a video tracked before masks existed; a policy that wants outlines
    #: degrades rather than refuses, since the boxes still say where everyone
    #: is.
    masks: Mapping[str, np.ndarray | None] | None = None

    @property
    def attributable(self) -> bool:
        """Whether an automatic pick is meaningful here at all.

        Two events get no automatic actor, and neither is a missing value:
        one with no contact point has nothing to attribute, and an INVISIBLE
        one has a point that sits next to somebody who demonstrably did not
        perform the action (the actor is off-screen or hidden — that is what
        makes it invisible). Extraction has always refused both; stating it
        here keeps every policy refusing them the same way instead of each
        rediscovering the rule, or quietly not.
        """
        return self.contact is not None and self.visible


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

    #: Whether decide() needs EventContext.tracklets filled. The caller uses
    #: this to refuse the job up front instead of silently abstaining on every
    #: event of a video that was never tracked.
    @property
    def needs_tracklets(self) -> bool: ...

    def decide(self, context: EventContext) -> ActorPick: ...


class RulePolicy:
    """The geometric rule production has always run on."""

    name = RULE_BASED
    needs_tracklets = False

    def decide(self, context: EventContext) -> ActorPick:
        if not context.attributable:
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


class TrackletPolicy:
    """A learned ranker choosing among the tracklets alive near the event.

    Answers with a tracklet, never a box: which pixels that tracklet means for
    this event is a question the masks answer, and they are not this layer's.
    """

    needs_tracklets = True

    def __init__(self, model: AssociationModel):
        if model.feature_set != FEATURE_SET_TRACK:
            raise ValueError(
                f"{model.name!r} is a {model.feature_set!r} model; "
                f"a tracklet policy needs {FEATURE_SET_TRACK!r}"
            )
        self._model = model

    @property
    def name(self) -> str:
        return f"learned:{self._model.name}"

    def decide(self, context: EventContext) -> ActorPick:
        if not context.attributable:
            return ActorPick()
        assert context.contact is not None
        x, y = context.contact
        candidates = candidates_near(
            context.tracklets, context.frame, masks=context.masks
        )
        features = extract_track_features(
            candidates,
            x,
            y,
            context.frame,
            detections=context.detections,
            visible=context.visible,
        )
        # AssociationModel.decision() speaks PersonBox, which a tracklet is
        # not; only the two probability blocks are shared, so the threshold
        # pair is applied here against an INDEX instead.
        probabilities, none_probability = self._model.probabilities(features)
        top = int(np.argmax(probabilities)) if len(probabilities) else None
        confidence = float(probabilities[top]) if top is not None else None
        selected = (
            top
            if top is not None
            and confidence is not None
            and confidence >= self._model.threshold
            and none_probability < self._model.none_threshold
            else None
        )
        return ActorPick(
            track=features.refs[selected] if selected is not None else None,
            candidates=len(candidates),
            diagnostic={
                "version": self.name,
                "decision": (
                    DecisionReason.SELECTED.value
                    if selected is not None
                    else DecisionReason.NO_CANDIDATE.value
                    if top is None
                    else DecisionReason.AMBIGUOUS.value
                ),
                "candidate_count": len(candidates),
                "confidence": (
                    round(confidence, 4) if confidence is not None else None
                ),
                "none_probability": round(none_probability, 4),
                "top": (
                    {"track": features.refs[top].key}
                    if top is not None
                    else None
                ),
            },
        )


def build_policy(checkpoint: str | None) -> ActorPolicy:
    """``None`` is the rule; anything else names a trained checkpoint."""
    from yp_video.actor import checkpoints

    if checkpoint is None or checkpoint == RULE_BASED:
        return RulePolicy()
    model = checkpoints.load(checkpoint)
    if model.feature_set == FEATURE_SET_TRACK:
        return TrackletPolicy(model)
    return _BoxModelPolicy(model)


class _BoxModelPolicy:
    """A learned ranker over detection boxes — the box contract's twin of
    TrackletPolicy. No checkpoint uses it today; it exists so ``build_policy``
    has no unreachable branch and no silent wrong answer if one is trained."""

    needs_tracklets = False

    def __init__(self, model: AssociationModel):
        self._model = model

    @property
    def name(self) -> str:
        return f"learned:{self._model.name}"

    def decide(self, context: EventContext) -> ActorPick:
        if not context.attributable:
            return ActorPick()
        assert context.contact is not None
        x, y = context.contact
        people = [person_from_detection(d) for d in context.detections]
        features = extract_features(people, x, y)
        decision = self._model.decision(features)
        return ActorPick(
            box=decision.selected.xyxy if decision.selected else None,
            candidates=len(decision.ranked),
            diagnostic=decision.diagnostic(),
        )
