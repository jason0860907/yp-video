"""Application service wrapping the association the extraction pipeline runs.

Extraction associates from DETECTION BOXES: at the point in the pipeline
where a crop is cut, tracking has not necessarily run, so the only question
answerable there is the geometric one. That is why the rule is what lives
behind this boundary and a learned model does not.

A learned model answering the same question does exist, but it names a
tracklet, and reaching it means running Association Predict over a tracked
video (see actor/policy.py) rather than activating something here. There was
once a "shadow" slot on this service that scored a learned BOX ranker
alongside the rule for diagnostics; it was removed with that ranker, because
a slot no checkpoint can ever occupy is not a configuration point.
"""

from __future__ import annotations

from dataclasses import dataclass

from yp_video.actor.ranking import AssociationDecision, rule_decision
from yp_video.person.detector import PersonBox


@dataclass(frozen=True)
class AssociationBundle:
    #: What extraction acted on.
    production: AssociationDecision

    @property
    def production_candidates(self) -> list[PersonBox]:
        return [candidate.person for candidate in self.production.ranked]

    def diagnostic(self) -> dict:
        return self.production.diagnostic()


class ActorAssociationService:
    """One stable pipeline dependency; the policy remains behind it."""

    def associate(
        self,
        people: list[PersonBox],
        x: float,
        y: float,
    ) -> AssociationBundle:
        return AssociationBundle(rule_decision(people, x, y))
