"""Application service coordinating production and shadow association."""

from __future__ import annotations

import logging
from dataclasses import dataclass

from yp_video.actor import checkpoints
from yp_video.actor.features import extract_features
from yp_video.actor.model import FEATURE_SET_BOX, AssociationModel
from yp_video.actor.ranking import AssociationDecision, rule_decision
from yp_video.person.detector import PersonBox

log = logging.getLogger(__name__)

#: The feature contract this service can actually supply. It scores the shadow
#: on the same boxes the rule ranks, so a tracklet model finds nothing here to
#: read — activating one is not a configuration choice, it is a shape error on
#: every event. Serving a track model means giving extraction the tracklets,
#: not relaxing this constant.
SHADOW_FEATURE_SET = FEATURE_SET_BOX


def shadow_rejection(model: AssociationModel) -> str | None:
    """Why this checkpoint cannot be the extraction shadow, or None."""
    if model.feature_set != SHADOW_FEATURE_SET:
        return (
            f"Checkpoint {model.name!r} is trained on the "
            f"{model.feature_set!r} feature set; extraction can only supply "
            f"{SHADOW_FEATURE_SET!r}"
        )
    return None


@dataclass(frozen=True)
class AssociationBundle:
    #: What extraction acted on.
    production: AssociationDecision
    #: The learned ranker's opinion, when one is activated. Diagnostics only —
    #: it never changes the crop.
    learned_shadow: AssociationDecision | None

    @property
    def production_candidates(self) -> list[PersonBox]:
        return [
            candidate.person for candidate in self.production.ranked
        ]

    def diagnostic(self) -> dict:
        return {
            **self.production.diagnostic(),
            "learned": (
                self.learned_shadow.diagnostic()
                if self.learned_shadow is not None
                else None
            ),
        }


class ActorAssociationService:
    """One stable pipeline dependency; policies remain behind this boundary."""

    def __init__(self, learned_shadow: AssociationModel | None = None):
        self._learned_shadow = learned_shadow

    @classmethod
    def from_active_shadow(cls) -> "ActorAssociationService":
        try:
            model = checkpoints.load_active_shadow()
        except (OSError, ValueError, KeyError):
            log.exception(
                "Learned association shadow unavailable; continuing on the rule"
            )
            return cls()
        # Refuse an incompatible shadow once, here, rather than let it raise
        # per event for the length of a video.
        if model is not None and (reason := shadow_rejection(model)):
            log.warning("Ignoring learned association shadow: %s", reason)
            return cls()
        return cls(model)

    def associate(
        self,
        people: list[PersonBox],
        x: float,
        y: float,
    ) -> AssociationBundle:
        production = rule_decision(people, x, y)
        learned = None
        if self._learned_shadow is not None:
            try:
                learned = self._learned_shadow.decision(
                    extract_features(people, x, y)
                )
            except Exception:  # noqa: BLE001 — shadow cannot block production
                log.exception(
                    "Learned association shadow failed; continuing on the rule"
                )
        return AssociationBundle(production, learned)
