from __future__ import annotations

import json
import tempfile
import unittest
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import numpy as np
from fastapi import HTTPException
from pydantic import TypeAdapter

from yp_video.actor import checkpoints as association_checkpoints
from yp_video.actor import labels as actor_labels
from yp_video.actor.features import (
    CANDIDATE_FEATURE_NAMES,
    CONTEXT_FEATURE_NAMES,
    AssociationFeatures,
    extract_features,
)
from yp_video.actor.labels import ActorLabel, ActorVerdict
from yp_video.actor.model import FEATURE_SET_TRACK, AssociationModel
from yp_video.actor.ranking import (
    CandidateSource,
    DecisionReason,
    rank_candidates,
    rule_decision,
)
from yp_video.actor.service import (
    ActorAssociationService,
    shadow_rejection,
)
from yp_video.actor.track_features import (
    TRACK_CANDIDATE_FEATURE_NAMES,
    TRACK_CONTEXT_FEATURE_NAMES,
)
from yp_video.core.cache import StatCache
from yp_video.core.jsonl import write_jsonl
from yp_video.extraction import actor_fix, done
from yp_video.person.detector import PersonBox
from yp_video.web.routers import actor_association as router


def _person(
    *,
    score: float,
    box: tuple[float, float, float, float],
) -> PersonBox:
    return PersonBox(xyxy=box, score=score)


class RulePolicyTests(unittest.TestCase):
    """The rule decides; the candidate ranking does not.

    They used to be two "rules" (V1 and a V2 that also abstained), which made
    the candidate generator look like a competing policy nobody had adopted.
    """

    def test_the_rule_takes_the_best_confident_candidate(self) -> None:
        actor = _person(score=0.9, box=(30, 20, 70, 120))

        decision = rule_decision([actor], 50, 20)

        self.assertIs(decision.selected, actor)
        self.assertEqual(decision.reason, DecisionReason.SELECTED)
        self.assertEqual(decision.version, "rule-based")

    def test_the_rule_ignores_a_low_confidence_person(self) -> None:
        """The 0.1-0.5 band exists to give the human picker more boxes."""
        faint = _person(score=0.2, box=(30, 20, 70, 120))

        self.assertEqual(rule_decision([faint], 50, 20).ranked, ())
        # ...but it stays a candidate the learned ranker may choose.
        self.assertEqual(len(rank_candidates([faint], 50, 20)), 1)

    def test_the_candidate_set_never_drops_a_detected_person(self) -> None:
        """Candidate recall has to be 1.0 or a labeled truth can be
        unreachable — geometry is a negative feature, not a gate."""
        far = _person(score=0.2, box=(400, 400, 450, 550))

        self.assertEqual(rule_decision([far], 10, 10).ranked, ())
        ranked = rank_candidates([far], 10, 10)
        self.assertEqual(len(ranked), 1)
        self.assertIs(ranked[0].source, CandidateSource.OTHER)

    def test_candidates_are_ordered_best_first(self) -> None:
        near = _person(score=0.8, box=(30, 20, 70, 120))
        far = _person(score=0.8, box=(400, 400, 450, 550))

        ranked = rank_candidates([far, near], 50, 20)

        self.assertIs(ranked[0].person, near)
        self.assertLess(ranked[0].cost, ranked[1].cost)


class ActorLabelStoreTests(unittest.TestCase):
    @contextmanager
    def _store(self):
        """The label store pointed at a scratch file, with a cold cache."""
        with tempfile.TemporaryDirectory() as raw_dir:
            path = Path(raw_dir) / "match_actors.json"
            with (
                patch.object(
                    actor_labels, "actors_path", return_value=path
                ),
                patch.object(actor_labels, "_cache", StatCache()),
            ):
                yield path

    def test_verdict_survives_a_round_trip_and_is_never_inferred(
        self,
    ) -> None:
        with self._store() as path:
            actor_labels.save(
                "match",
                "manual-event",
                ActorLabel(
                    ActorVerdict.MANUAL,
                    box=(1, 2, 3, 4),
                    frame=812,
                    snap=False,
                ),
            )
            actor_labels.save(
                "match", "occluded-event", ActorLabel(ActorVerdict.OCCLUDED)
            )

            labels = actor_labels.load("match")
            self.assertEqual(
                labels["manual-event"],
                ActorLabel(
                    ActorVerdict.MANUAL,
                    box=(1.0, 2.0, 3.0, 4.0),
                    frame=812,
                    snap=False,
                ),
            )
            self.assertEqual(
                labels["occluded-event"].verdict, ActorVerdict.OCCLUDED
            )
            self.assertTrue(labels["manual-event"].overrides_auto)

            # Defaults stay out of the file; the verdict never does.
            stored = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(stored["version"], actor_labels.SCHEMA_VERSION)
            self.assertNotIn("snap", stored["actors"]["occluded-event"])
            self.assertEqual(
                stored["actors"]["occluded-event"], {"verdict": "occluded"}
            )

    def test_reverting_clears_the_label_and_the_file(self) -> None:
        with self._store() as path:
            actor_labels.save(
                "match", "event", ActorLabel(ActorVerdict.OCCLUDED)
            )
            actor_labels.save("match", "event", None)

            self.assertEqual(actor_labels.load("match"), {})
            self.assertFalse(path.exists())

    def test_bulk_confirmation_never_overwrites_a_human_fix(self) -> None:
        with self._store():
            actor_labels.save(
                "match", "fixed", ActorLabel(ActorVerdict.OCCLUDED)
            )

            added = actor_labels.confirm_auto(
                "match",
                {
                    "fixed": ActorLabel(
                        ActorVerdict.CONFIRMED_AUTO, box=(1, 2, 3, 4)
                    ),
                    "untouched": ActorLabel(
                        ActorVerdict.CONFIRMED_AUTO, box=(5, 6, 7, 8)
                    ),
                },
            )

            labels = actor_labels.load("match")
            self.assertEqual(added, ['untouched'])
            self.assertEqual(labels["fixed"].verdict, ActorVerdict.OCCLUDED)
            self.assertEqual(
                labels["untouched"].verdict, ActorVerdict.CONFIRMED_AUTO
            )
            self.assertFalse(labels["untouched"].overrides_auto)


class DoneConfirmationTests(unittest.TestCase):
    def test_done_confirms_only_assigned_automatic_actors(self) -> None:
        records = [
            {
                "id": "auto-assigned",
                "resolution": "auto",
                "actor_box": [1, 2, 3, 4],
                "frame": 10,
            },
            {
                "id": "auto-unassigned",
                "resolution": "auto",
                "actor_box": [5, 6, 7, 8],
                "frame": 20,
            },
            {
                "id": "manual-assigned",
                "resolution": "manual",
                "actor_box": [9, 10, 11, 12],
                "frame": 30,
            },
        ]
        with patch.object(
            done.identity,
            "load_assignments",
            return_value={"auto-assigned": "A", "manual-assigned": "B"},
        ):
            confirmable = done.confirmable_actors("match", records)

        self.assertEqual(list(confirmable), ["auto-assigned"])
        self.assertEqual(
            confirmable["auto-assigned"],
            ActorLabel(
                ActorVerdict.CONFIRMED_AUTO, box=(1.0, 2.0, 3.0, 4.0), frame=10
            ),
        )


class AssociationTrainingSelectionTests(unittest.TestCase):
    def test_training_request_requires_an_explicit_video_selection(self) -> None:
        adapter = TypeAdapter(router.AssociationTrainRequest)
        with self.assertRaises(ValueError):
            adapter.validate_python({})
        with self.assertRaises(ValueError):
            adapter.validate_python({"videos": ["only-one.mp4"]})

    def test_only_the_selected_stems_build_the_dataset(self) -> None:
        selected_dataset = type(
            "Dataset",
            (),
            {"stems": ("a", "b")},
        )()
        paths = {
            "a.mp4": Path("/cuts/a.mp4"),
            "b.mp4": Path("/cuts/b.mp4"),
        }
        with (
            patch.object(router, "find_cut", side_effect=paths.get),
            patch.object(
                router.actor_dataset,
                "load_dataset",
                return_value=selected_dataset,
            ) as load,
        ):
            result, resolved = router._selected_training_dataset(
                ["b.mp4", "a.mp4"]
            )

        self.assertIs(result, selected_dataset)
        self.assertEqual(resolved, [paths["b.mp4"], paths["a.mp4"]])
        load.assert_called_once_with(["b", "a"])

    def test_duplicate_video_names_do_not_fake_grouped_validation(self) -> None:
        with patch.object(
            router, "find_cut", return_value=Path("/cuts/a.mp4")
        ):
            with self.assertRaises(HTTPException) as caught:
                router._selected_training_dataset(["a.mp4", "a.mp4"])

        self.assertEqual(caught.exception.status_code, 400)
        self.assertIn("distinct", str(caught.exception.detail))


class FixEndpointTests(unittest.TestCase):
    """The Association Label page's one write, wired end to end.

    Everything below the router is stubbed on purpose: this asserts the
    transport contract (mode → command, response shape, deferred refresh
    scheduled) without letting a test touch a real video's annotations.
    """

    def _fix(self, payload: dict) -> tuple[dict, tuple, list]:
        adapter = TypeAdapter(router.ActorFixRequest)
        applied: list[tuple] = []

        class _Tasks:
            def __init__(self) -> None:
                self.scheduled: list[tuple] = []

            def add_task(self, fn, *args, **kwargs) -> None:
                self.scheduled.append((fn, args, kwargs))

        tasks = _Tasks()
        result = actor_fix.ActorFixResult(
            record={"id": "e1", "actor_revision": 3, "detections": [{"box": [1, 2, 3, 4], "keypoints": [[1, 2, 0.9]]}]},
            refreshing_models=("clip-reid",),
            actor_revision=3,
        )

        def fake_apply(_video, command, *, active_model):
            applied.append((command, active_model))
            return result

        with tempfile.TemporaryDirectory() as raw_dir:
            records = Path(raw_dir) / "match.jsonl"
            records.touch()
            with (
                patch.object(router, "find_cut", return_value=Path(raw_dir) / "match.mp4"),
                patch.object(
                    router.extraction_store, "records_path", return_value=records
                ),
                patch.object(router, "_synchronous_model", return_value="clip-reid"),
                patch.object(router.actor_fix, "apply", side_effect=fake_apply),
                patch.object(
                    router.tracks_store, "tracks_path", return_value=Path(raw_dir) / "none"
                ),
            ):
                response = router.fix(
                    "match.mp4", adapter.validate_python(payload), tasks  # type: ignore[arg-type]
                )
        return response, applied[0], tasks.scheduled

    def test_pick_reaches_the_service_as_a_manual_label(self) -> None:
        response, (command, model), scheduled = self._fix(
            {
                "mode": "pick",
                "event_id": "e1",
                "box": [1, 2, 3, 4],
                "frame": 7,
                "snap": False,
            }
        )

        self.assertEqual(
            command.label,
            ActorLabel(ActorVerdict.MANUAL, box=(1, 2, 3, 4), frame=7, snap=False),
        )
        self.assertEqual(model, "clip-reid")
        self.assertEqual(response["record"]["actor_review"], "manual")
        # Skeletons stay server-side; the picker only ever needed boxes.
        self.assertNotIn("keypoints", response["record"]["detections"][0])
        self.assertIsNone(response["track_link"])
        self.assertEqual(response["refreshing_models"], ("clip-reid",))
        # The matrices not refreshed inline must be scheduled, or they stay
        # silently stale.
        self.assertEqual(len(scheduled), 1)
        self.assertEqual(scheduled[0][2]["expected_revision"], 3)

    def test_revert_reports_the_event_as_unreviewed_again(self) -> None:
        response, (command, _model), _scheduled = self._fix(
            {"mode": "auto", "event_id": "e1"}
        )

        self.assertIsNone(command.label)
        self.assertEqual(response["record"]["actor_review"], "unreviewed")

    def test_a_video_without_records_is_a_404(self) -> None:
        adapter = TypeAdapter(router.ActorFixRequest)
        with tempfile.TemporaryDirectory() as raw_dir:
            with (
                patch.object(router, "find_cut", return_value=Path(raw_dir) / "m.mp4"),
                patch.object(
                    router.extraction_store,
                    "records_path",
                    return_value=Path(raw_dir) / "missing.jsonl",
                ),
                patch.object(router.actor_fix, "apply") as apply,
            ):
                with self.assertRaises(HTTPException) as caught:
                    router.fix(
                        "m.mp4",
                        adapter.validate_python({"mode": "occluded", "event_id": "e1"}),
                        None,  # type: ignore[arg-type]
                    )
        self.assertEqual(caught.exception.status_code, 404)
        apply.assert_not_called()


class ConfirmEndpointTests(unittest.TestCase):
    RECORDS = [
        {"id": "auto-a", "resolution": "auto", "actor_box": [1, 2, 3, 4], "frame": 1},
        {"id": "auto-b", "resolution": "auto", "actor_box": [5, 6, 7, 8], "frame": 2},
        {"id": "fixed", "resolution": "manual", "actor_box": [9, 10, 11, 12], "frame": 3},
        {"id": "miss", "resolution": "unresolved", "frame": 4},
        {
            "id": "model-occluded",
            "resolution": "unresolved",
            "frame": 5,
            "association": {"decision": "abstained", "kind": "occluded"},
        },
    ]

    @contextmanager
    def _video(self):
        with tempfile.TemporaryDirectory() as raw_dir:
            root = Path(raw_dir)
            records = root / "match.jsonl"
            write_jsonl(records, {"video": "match"}, self.RECORDS)
            with (
                patch.object(
                    router.extraction_store, "records_path", return_value=records
                ),
                patch.object(
                    actor_labels, "actors_path", return_value=root / "match_actors.json"
                ),
                patch.object(actor_labels, "_cache", StatCache()),
            ):
                yield

    def test_confirms_automatic_picks_and_leaves_a_human_fix_alone(self) -> None:
        with self._video():
            # A verdict the user already gave: bulk confirmation must not
            # quietly overwrite it with "the machine was right".
            actor_labels.save("match", "auto-b", ActorLabel(ActorVerdict.OCCLUDED))

            response = router.confirm("match.mp4", router.ConfirmRequest())
            labels = actor_labels.load("match")

        self.assertEqual(
            response["confirmed"],
            {"auto-a": "confirmed_auto", "model-occluded": "occluded"},
        )
        self.assertEqual(labels["auto-a"].verdict, ActorVerdict.CONFIRMED_AUTO)
        self.assertEqual(labels["auto-a"].box, (1.0, 2.0, 3.0, 4.0))
        self.assertEqual(labels["auto-b"].verdict, ActorVerdict.OCCLUDED)
        self.assertEqual(
            labels["model-occluded"].verdict, ActorVerdict.OCCLUDED
        )
        self.assertIsNone(labels["model-occluded"].box)
        # A manual fix already had a label; a miss has no box to agree with.
        self.assertNotIn("miss", labels)

    def test_confirming_twice_is_a_no_op(self) -> None:
        with self._video():
            first = router.confirm("match.mp4", router.ConfirmRequest())
            second = router.confirm("match.mp4", router.ConfirmRequest())

        self.assertEqual(
            first["confirmed"],
            {
                "auto-a": "confirmed_auto",
                "auto-b": "confirmed_auto",
                "model-occluded": "occluded",
            },
        )
        self.assertEqual(second["confirmed"], {})

    def test_confirming_model_occluded_returns_the_occluded_verdict(self) -> None:
        """The UI must not turn its `Model: occluded?` hint into Confirmed."""
        with self._video():
            response = router.confirm(
                "match.mp4",
                router.ConfirmRequest(event_ids=["model-occluded"]),
            )
            label = actor_labels.load("match")["model-occluded"]

        self.assertEqual(
            response["confirmed"], {"model-occluded": "occluded"}
        )
        self.assertEqual(label.verdict, ActorVerdict.OCCLUDED)

    def test_a_miss_cannot_be_confirmed(self) -> None:
        """It needs a real verdict; reporting success would be a lie."""
        with self._video():
            with self.assertRaises(HTTPException) as caught:
                router.confirm(
                    "match.mp4", router.ConfirmRequest(event_ids=["auto-a", "miss"])
                )
            self.assertEqual(actor_labels.load("match"), {})

        self.assertEqual(caught.exception.status_code, 400)
        self.assertIn("miss", str(caught.exception.detail))


class LearnedAssociationTests(unittest.TestCase):
    def _model(
        self,
    ) -> tuple[AssociationModel, PersonBox, AssociationFeatures]:
        actor = _person(
            score=0.9,
            box=(30, 20, 70, 120),
        )
        other = _person(
            score=0.8,
            box=(60, 20, 100, 120),
        )
        features = extract_features([actor, other], 50, 20)
        candidate_weights = np.zeros(
            len(CANDIDATE_FEATURE_NAMES), dtype=np.float64
        )
        candidate_weights[
            CANDIDATE_FEATURE_NAMES.index("rank_reciprocal")
        ] = 8.0
        none_weights = np.zeros(
            len(CONTEXT_FEATURE_NAMES), dtype=np.float64
        )
        none_weights[0] = -10.0
        return (
            AssociationModel(
                name="candidate",
                candidate_mean=np.zeros_like(candidate_weights),
                candidate_scale=np.ones_like(candidate_weights),
                context_mean=np.zeros_like(none_weights),
                context_scale=np.ones_like(none_weights),
                candidate_weights=candidate_weights,
                none_weights=none_weights,
                threshold=0.5,
                none_threshold=0.5,
            ),
            actor,
            features,
        )

    def test_ranker_and_none_classifier_have_separate_decisions(self) -> None:
        model, actor, features = self._model()

        decision = model.decision(features)
        self.assertIs(decision.selected, actor)
        self.assertGreater(decision.confidence or 0, 0.5)

        none_weights = model.none_weights.copy()
        none_weights[0] = 10.0
        abstaining = replace(model, none_weights=none_weights)
        self.assertIsNone(abstaining.decision(features).selected)

        restored = AssociationModel.from_payload(model.payload())
        self.assertEqual(restored.name, model.name)
        self.assertTrue(
            np.allclose(restored.candidate_weights, model.candidate_weights)
        )

    def test_candidate_checkpoint_never_activates_implicitly(self) -> None:
        model, _actor, _features = self._model()
        with tempfile.TemporaryDirectory() as raw_dir:
            root = Path(raw_dir)
            with (
                patch.object(
                    association_checkpoints,
                    "CHECKPOINTS_DIR",
                    root / "checkpoints",
                ),
                patch.object(
                    association_checkpoints,
                    "SHADOW_CONFIG",
                    root / "shadow.json",
                ),
                patch.object(
                    association_checkpoints,
                    "_model_cache",
                    StatCache(),
                ),
            ):
                association_checkpoints.save_candidate(
                    model,
                    {
                        "name": model.name,
                        "metrics": {},
                        "training": {"examples": 1, "stems": []},
                    },
                )
                self.assertIsNone(
                    association_checkpoints.active_shadow_name()
                )
                self.assertFalse(
                    association_checkpoints.list_candidates()[0][
                        "active_shadow"
                    ]
                )

                association_checkpoints.set_active_shadow(model.name)
                self.assertEqual(
                    association_checkpoints.active_shadow_name(),
                    model.name,
                )

    def test_broken_shadow_cannot_block_production_rule(self) -> None:
        actor = _person(
            score=0.9,
            box=(30, 20, 70, 120),
        )
        with (
            self.assertLogs(
                "yp_video.actor.service", level="ERROR"
            ),
            patch.object(
                association_checkpoints,
                "load_active_shadow",
                side_effect=ValueError("broken candidate"),
            ),
        ):
            service = ActorAssociationService.from_active_shadow()

        result = service.associate([actor], 50, 20)
        self.assertIs(result.production.selected, actor)
        self.assertIsNone(result.learned_shadow)

    def test_tracklet_model_is_refused_as_the_box_shadow(self) -> None:
        """A track checkpoint loads fine and still cannot serve here.

        The service supplies box features; feeding them to a tracklet model is
        a shape error, so it must be refused once at construction rather than
        raise on every event of a video.
        """
        n_candidate = len(TRACK_CANDIDATE_FEATURE_NAMES)
        n_context = len(TRACK_CONTEXT_FEATURE_NAMES)
        track_model = AssociationModel(
            name="track-shadow",
            candidate_mean=np.zeros(n_candidate),
            candidate_scale=np.ones(n_candidate),
            context_mean=np.zeros(n_context),
            context_scale=np.ones(n_context),
            candidate_weights=np.zeros(n_candidate),
            none_weights=np.zeros(n_context),
            threshold=0.5,
            none_threshold=0.5,
            feature_set=FEATURE_SET_TRACK,
        )
        self.assertIsNotNone(shadow_rejection(track_model))

        actor = _person(score=0.9, box=(30, 20, 70, 120))
        with (
            self.assertLogs("yp_video.actor.service", level="WARNING"),
            patch.object(
                association_checkpoints,
                "load_active_shadow",
                return_value=track_model,
            ),
        ):
            service = ActorAssociationService.from_active_shadow()

        result = service.associate([actor], 50, 20)
        self.assertIs(result.production.selected, actor)
        self.assertIsNone(result.learned_shadow)


if __name__ == "__main__":
    unittest.main()
