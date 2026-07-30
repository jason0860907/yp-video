from __future__ import annotations

import json
import tempfile
import unittest
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import numpy as np
from fastapi import HTTPException
from pydantic import TypeAdapter

from yp_video.actor import checkpoints as association_checkpoints
from yp_video.actor import labels as actor_labels
from yp_video.actor import review as actor_review
from yp_video.actor.labels import ActorLabel, ActorVerdict
from yp_video.actor.model import FEATURE_SET_TRACK, AssociationModel
from yp_video.actor.ranking import DecisionReason, rule_decision
from yp_video.actor.service import ActorAssociationService
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
    """The rule, which is now the only thing in ranking.py.

    It decides and it gates. The wide candidate SET that used to live beside
    it — everyone above the detector floor, geometry as a negative feature —
    existed only to feed a learned box ranker and went with it.
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
        self.assertIsNone(rule_decision([faint], 50, 20).selected)

    def test_the_rule_gates_on_geometry_and_says_so(self) -> None:
        """Out of reach of the padded box is NO_CANDIDATE, not a bad pick."""
        far = _person(score=0.9, box=(400, 400, 450, 550))

        decision = rule_decision([far], 10, 10)

        self.assertEqual(decision.ranked, ())
        self.assertEqual(decision.reason, DecisionReason.NO_CANDIDATE)

    def test_candidates_are_ordered_best_first(self) -> None:
        near = _person(score=0.8, box=(30, 20, 70, 120))
        also = _person(score=0.8, box=(45, 20, 85, 120))

        decision = rule_decision([also, near], 50, 20)

        self.assertIs(decision.ranked[0].person, near)
        self.assertLess(
            decision.ranked[0].geometry_cost,
            decision.ranked[1].geometry_cost,
        )


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


class AssociationReviewProgressTests(unittest.TestCase):
    def test_summary_is_done_over_done_plus_in_progress(self) -> None:
        rows = [
            actor_review.ReviewProgress(3, 3, 0, {"manual": 3}),
            actor_review.ReviewProgress(3, 1, 2, {"occluded": 1}),
            actor_review.ReviewProgress(3, 0, 3, {}),
        ]
        with tempfile.TemporaryDirectory() as raw_dir:
            records = Path(raw_dir) / "records.jsonl"
            records.touch()
            with (
                patch.object(
                    actor_review, "records_path", return_value=records
                ),
                patch.object(
                    actor_review, "read_jsonl_header", return_value={}
                ),
                patch.object(
                    actor_review, "review_progress", side_effect=rows
                ),
            ):
                summary = actor_review.review_summary(
                    ["done", "in-progress", "unlabeled"]
                )

        self.assertEqual(summary.done, 1)
        self.assertEqual(summary.started, 2)


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
    def test_training_request_requires_disjoint_train_and_validation(self) -> None:
        adapter = TypeAdapter(router.AssociationTrainRequest)
        with self.assertRaises(ValueError):
            adapter.validate_python({})
        with self.assertRaises(ValueError):
            adapter.validate_python(
                {
                    "train_videos": ["same.mp4"],
                    "val_videos": ["same.mp4"],
                }
            )

    def test_only_the_selected_videos_build_the_spot_snapshot(self) -> None:
        paths = {
            "a.mp4": Path("/cuts/a.mp4"),
            "b.mp4": Path("/cuts/b.mp4"),
        }
        labels = {
            "a": Path("/labels/a_actions.jsonl"),
            "b": Path("/labels/b_actions.jsonl"),
        }
        with (
            patch.object(router, "find_cut", side_effect=paths.get),
            patch.object(
                router.spot_associate,
                "action_label_path",
                side_effect=labels.get,
            ),
            patch.object(router.actor_labels, "load", return_value={"event": object()}),
            patch.object(router, "read_jsonl_cached", return_value=({}, [{}])),
            patch.object(
                router.spot_actor_labels,
                "build",
                return_value=([{"id": "event"}], {"track": 1}),
            ),
        ):
            result = router._association_training_items(
                ["b.mp4", "a.mp4"]
            )

        self.assertEqual(
            result,
            [(labels["b"], paths["b.mp4"]), (labels["a"], paths["a.mp4"])],
        )

    def test_video_without_actor_review_is_rejected_before_gpu_work(self) -> None:
        with (
            patch.object(router, "find_cut", return_value=Path("/cuts/a.mp4")),
            patch.object(
                router.spot_associate,
                "action_label_path",
                return_value=Path("/labels/a_actions.jsonl"),
            ),
            patch.object(router.actor_labels, "load", return_value={}),
        ):
            with self.assertRaises(HTTPException) as caught:
                router._association_training_items(["a.mp4"])

        self.assertEqual(caught.exception.status_code, 400)
        self.assertIn("Association Label", str(caught.exception.detail))


class NeuralAssociationTrainTests(unittest.IsolatedAsyncioTestCase):
    async def test_train_starts_the_independent_association_runner(self) -> None:
        request = router.AssociationTrainRequest(
            train_videos=["train.mp4"],
            val_videos=["val.mp4"],
            run_name="yp_actor_test",
            backbone="rny002",
        )
        train_item = (Path("/labels/train_actions.jsonl"), Path("/cuts/train.mp4"))
        val_item = (Path("/labels/val_actions.jsonl"), Path("/cuts/val.mp4"))
        start = AsyncMock(return_value={"id": "job"})

        with tempfile.TemporaryDirectory() as raw_dir:
            root = Path(raw_dir)
            with (
                patch.object(router, "ACTION_CHECKPOINTS_DIR", root / "checkpoints"),
                patch.object(router, "SPOT_DIR", root / "yp-spot"),
                patch.object(
                    router,
                    "_association_training_items",
                    side_effect=[[train_item], [val_item]],
                ),
                patch.object(
                    router,
                    "_start_association_training",
                    start,
                ),
            ):
                result = await router.train(request)

        self.assertEqual(result, {"id": "job"})
        self.assertEqual(start.await_args.args[0].backbone, "rny002")
        self.assertEqual(start.await_args.kwargs["train_items"], [train_item])
        self.assertEqual(start.await_args.kwargs["val_items"], [val_item])
        self.assertIsNone(start.await_args.kwargs["init_checkpoint"])

    async def test_train_rejects_a_duplicate_active_job_before_validation(
        self,
    ) -> None:
        request = router.AssociationTrainRequest(
            train_videos=["train.mp4"],
            val_videos=["val.mp4"],
        )
        with (
            patch.object(
                router,
                "_active_job",
                return_value={"name": "Association Train (already-running)"},
            ),
            patch.object(router, "_association_training_items") as prepare,
        ):
            with self.assertRaises(HTTPException) as caught:
                await router.train(request)

        self.assertEqual(caught.exception.status_code, 409)
        self.assertIn("already active", str(caught.exception.detail))
        prepare.assert_not_called()

    def test_history_exposes_each_epoch_in_display_order(self) -> None:
        with tempfile.TemporaryDirectory() as raw_dir:
            path = Path(raw_dir) / "metrics.jsonl"
            path.write_text(
                "\n".join(
                    (
                        json.dumps(
                            {
                                "epoch": 0,
                                "loss": {"train": 2.0, "val": 3.0},
                                "train": {
                                    "player_top1": 0.7,
                                    "overall_exact": 0.6,
                                },
                                "val": {
                                    "player_top1": 0.4,
                                    "overall_exact": 0.3,
                                },
                                "best": True,
                            }
                        ),
                        "not-json",
                    )
                ),
                encoding="utf-8",
            )

            history = router._association_history(path)

        self.assertEqual(
            history,
            [
                {
                    "epoch": 1,
                    "train_player_top1": 0.7,
                    "val_player_top1": 0.4,
                    "train_overall_exact": 0.6,
                    "val_overall_exact": 0.3,
                    "train_loss": 2.0,
                    "val_loss": 3.0,
                    "best": True,
                }
            ],
        )

    def test_predict_contract_no_longer_accepts_a_linear_checkpoint(self) -> None:
        adapter = TypeAdapter(router.PredictRequest)
        with self.assertRaises(ValueError):
            adapter.validate_python(
                {"videos": ["a.mp4"], "checkpoint": "linear-model"}
            )


class SpotActorInferenceContractTests(unittest.TestCase):
    @staticmethod
    def _declare_independent(package: Path) -> None:
        (package / "config.json").write_text(
            json.dumps(
                {
                    "task": "association",
                    "checkpoint_format": "yp-association-v1",
                }
            ),
            encoding="utf-8",
        )

    @staticmethod
    def _declare_legacy(package: Path) -> None:
        (package / "config.json").write_text(
            json.dumps({"predict_actor": True, "audio_backend": "logmel"}),
            encoding="utf-8",
        )
        (package / "manifest.json").write_text(
            json.dumps(
                {
                    "type": "actor-association-spot",
                    "holdout": "held-out-video",
                    "actor_targets": {"track": 12},
                    "holdout_metrics": {"all_top1": 0.84},
                }
            ),
            encoding="utf-8",
        )

    def test_inference_uses_the_independent_event_model_contract(self) -> None:
        with tempfile.TemporaryDirectory() as raw_dir:
            root = Path(raw_dir)
            package = root / "checkpoints" / "yp_actor"
            package.mkdir(parents=True)
            checkpoint = package / "checkpoint_best.pt"
            checkpoint.touch()
            self._declare_independent(package)
            label_file = root / "video_actions.jsonl"
            label_file.touch()
            predictions = root / "video_predictions.json"
            captured: list[str] = []

            def run_subprocess(command, **_kwargs):
                captured.extend(command)
                output = Path(command[command.index("--out") + 1])
                output.write_text(
                    json.dumps(
                        {
                            "events": [
                                {
                                    "id": "event",
                                    "track": "1:1",
                                    "confidence": 0.9,
                                    "kind": "track",
                                }
                            ]
                        }
                    ),
                    encoding="utf-8",
                )
                return SimpleNamespace(returncode=0, stdout="", stderr="")

            with (
                patch.object(
                    router.spot_associate,
                    "action_label_path",
                    return_value=label_file,
                ),
                patch.object(
                    router.spot_associate,
                    "read_jsonl",
                    return_value=({}, [{"id": "event"}]),
                ),
                patch.object(
                    router.spot_associate.actor_labels,
                    "candidates_only",
                    return_value=[{"id": "event", "frame": 10}],
                ),
                patch.object(
                    router.spot_associate,
                    "ensure_action_frame_cache",
                ),
                patch.object(
                    router.spot_associate.subprocess,
                    "run",
                    side_effect=run_subprocess,
                ),
                patch.object(
                    router.spot_associate,
                    "ACTOR_PREDICTIONS_DIR",
                    predictions.parent,
                ),
                patch.object(
                    router.spot_associate,
                    "predictions_path",
                    return_value=predictions,
                ),
            ):
                answers = router.spot_associate.run(
                    root / "video.mp4",
                    checkpoint,
                )

        self.assertEqual(answers["event"].track.key, "1:1")
        self.assertIn("yp_spot.association.predict", captured)
        self.assertEqual(
            captured[captured.index("--checkpoint-path") + 1],
            str(checkpoint),
        )
        self.assertNotIn("--audio-dir", captured)

    def test_picker_lists_a_legacy_actor_head_with_its_family(self) -> None:
        with tempfile.TemporaryDirectory() as raw_dir:
            package = Path(raw_dir) / "yp_actor_only"
            package.mkdir()
            checkpoint = package / "checkpoint_best.pt"
            checkpoint.touch()
            self._declare_legacy(package)
            with patch.object(
                router.spot_associate.prelabel,
                "list_checkpoints",
                return_value=[
                    {
                        "path": str(checkpoint),
                        "epoch": 4,
                        "mtime": 123.0,
                    }
                ],
            ):
                listed = router.spot_associate.list_association_checkpoints()

        self.assertEqual(len(listed), 1)
        self.assertEqual(listed[0]["name"], "yp_actor_only")
        self.assertEqual(listed[0]["family"], "legacy-actor-head")
        self.assertEqual(listed[0]["metrics"]["all_top1"], 0.84)
        self.assertEqual(listed[0]["validation_videos"], ["held-out-video"])
        self.assertEqual(listed[0]["actor_targets"], {"track": 12})

    def test_picker_exposes_joint_actor_validation_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as raw_dir:
            package = Path(raw_dir) / "yp_fusion_joint"
            package.mkdir()
            checkpoint = package / "checkpoint_best.pt"
            checkpoint.touch()
            self._declare_legacy(package)
            manifest = json.loads(
                (package / "manifest.json").read_text(encoding="utf-8")
            )
            manifest["best"] = {
                "task_metrics": {
                    "actor": {
                        "validation": {
                            "metrics": {
                                "player_top1": 0.72,
                                "overall_top1": 0.68,
                                "occluded_recall": 0.5,
                                "untracked_recall": 0.25,
                            }
                        }
                    }
                }
            }
            (package / "manifest.json").write_text(
                json.dumps(manifest),
                encoding="utf-8",
            )
            with patch.object(
                router.spot_associate.prelabel,
                "list_checkpoints",
                return_value=[
                    {
                        "path": str(checkpoint),
                        "epoch": 4,
                        "mtime": 123.0,
                    }
                ],
            ):
                listed = router.spot_associate.list_association_checkpoints()

        self.assertEqual(listed[0]["metrics"]["player_top1"], 0.72)
        self.assertEqual(listed[0]["metrics"]["overall_exact"], 0.68)
        self.assertEqual(listed[0]["metrics"]["occluded_recall"], 0.5)
        self.assertEqual(listed[0]["metrics"]["untracked_recall"], 0.25)

    def test_submit_validation_accepts_legacy_actor_weights(self) -> None:
        with tempfile.TemporaryDirectory() as raw_dir:
            package = Path(raw_dir) / "yp_actor_only"
            package.mkdir()
            checkpoint = package / "checkpoint_best.pt"
            checkpoint.touch()
            self._declare_legacy(package)
            with patch(
                "torch.load",
                return_value={"model._pred_actor.weight": object()},
            ):
                reason = router.spot_associate.rejection(checkpoint)

        self.assertIsNone(reason)

    def test_legacy_actor_head_uses_the_original_inference_contract(self) -> None:
        with tempfile.TemporaryDirectory() as raw_dir:
            root = Path(raw_dir)
            package = root / "yp_actor_only"
            package.mkdir()
            checkpoint = package / "checkpoint_best.pt"
            checkpoint.touch()
            self._declare_legacy(package)
            label_file = root / "video_actions.jsonl"
            label_file.touch()
            predictions = root / "video_predictions.json"
            audio_dir = root / "audio"
            captured: list[str] = []

            def run_subprocess(command, **_kwargs):
                captured.extend(command)
                output = Path(command[command.index("--out") + 1])
                output.write_text(
                    json.dumps(
                        {
                            "events": [
                                {
                                    "id": "event",
                                    "track": "1:1",
                                    "confidence": 0.8,
                                    "kind": "track",
                                }
                            ]
                        }
                    ),
                    encoding="utf-8",
                )
                return SimpleNamespace(returncode=0, stdout="", stderr="")

            with (
                patch.object(
                    router.spot_associate,
                    "action_label_path",
                    return_value=label_file,
                ),
                patch.object(
                    router.spot_associate,
                    "read_jsonl",
                    return_value=({}, [{"id": "event"}]),
                ),
                patch.object(
                    router.spot_associate.actor_labels,
                    "candidates_only",
                    return_value=[{"id": "event", "frame": 10}],
                ),
                patch.object(
                    router.spot_associate,
                    "ensure_action_frame_cache",
                ),
                patch.object(
                    router.spot_associate,
                    "_ensure_legacy_audio",
                    return_value=audio_dir,
                ),
                patch.object(
                    router.spot_associate.subprocess,
                    "run",
                    side_effect=run_subprocess,
                ),
                patch.object(
                    router.spot_associate,
                    "ACTOR_PREDICTIONS_DIR",
                    predictions.parent,
                ),
                patch.object(
                    router.spot_associate,
                    "predictions_path",
                    return_value=predictions,
                ),
            ):
                answers = router.spot_associate.run(
                    root / "video.mp4",
                    checkpoint,
                )

        self.assertEqual(answers["event"].track.key, "1:1")
        self.assertIn("yp_spot.associate", captured)
        self.assertEqual(
            captured[captured.index("--checkpoint_path") + 1],
            str(checkpoint),
        )
        self.assertEqual(
            captured[captured.index("--audio_dir") + 1],
            str(audio_dir),
        )


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
    """A trained checkpoint is a file, and stays one until it is named."""

    def _track_model(self, name: str = "candidate") -> AssociationModel:
        n_candidate = len(TRACK_CANDIDATE_FEATURE_NAMES)
        n_context = len(TRACK_CONTEXT_FEATURE_NAMES)
        return AssociationModel(
            name=name,
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

    @contextmanager
    def _repository(self):
        with tempfile.TemporaryDirectory() as raw_dir:
            with (
                patch.object(
                    association_checkpoints,
                    "CHECKPOINTS_DIR",
                    Path(raw_dir) / "checkpoints",
                ),
                patch.object(
                    association_checkpoints, "_model_cache", StatCache()
                ),
            ):
                yield Path(raw_dir) / "checkpoints"

    def test_a_saved_candidate_survives_a_round_trip(self) -> None:
        model = self._track_model()
        restored = AssociationModel.from_payload(model.payload())

        self.assertEqual(restored.name, model.name)
        self.assertEqual(restored.feature_set, FEATURE_SET_TRACK)
        self.assertTrue(
            np.allclose(restored.candidate_weights, model.candidate_weights)
        )

    def test_saving_a_candidate_activates_nothing(self) -> None:
        """There is no "current model" setting to drift out of sync with what
        produced a record. A model decides only where it is named, and its
        name is written into the record it produced."""
        model = self._track_model()
        with self._repository():
            association_checkpoints.save_candidate(
                model,
                {
                    "name": model.name,
                    "metrics": {},
                    "training": {"examples": 1, "stems": []},
                },
            )
            listed = association_checkpoints.list_candidates()

            self.assertEqual([row["name"] for row in listed], [model.name])
            self.assertIsNone(
                association_checkpoints.usable_rejection(model.name)
            )
            self.assertFalse(
                any(key.startswith("active") for key in listed[0])
            )

    def test_a_retired_contract_is_listed_with_its_reason(self) -> None:
        """box-v3 checkpoints sit on disk from before the box ranker was
        removed. Hiding them would be a worse answer than showing them with
        the reason they cannot run — the page has to explain the file."""
        model = self._track_model("legacy")
        with self._repository() as root:
            association_checkpoints.save_candidate(
                model,
                {
                    "name": model.name,
                    "metrics": {},
                    "training": {"examples": 1, "stems": []},
                },
            )
            payload = json.loads(
                (root / "legacy" / "model.json").read_text(encoding="utf-8")
            )
            payload["feature_set"] = "box-v3"
            (root / "legacy" / "model.json").write_text(
                json.dumps(payload), encoding="utf-8"
            )

            self.assertEqual(
                [row["name"] for row in association_checkpoints.list_candidates()],
                ["legacy"],
            )
            reason = association_checkpoints.usable_rejection("legacy")

        self.assertIsNotNone(reason)
        self.assertIn("box-v3", reason or "")

    def test_extraction_associates_on_the_rule_alone(self) -> None:
        """Extraction associates from detection boxes, before tracking has
        necessarily run, so the geometric question is the only one answerable
        there. The learned path answers a tracklet question and is reached by
        naming it in Association Predict, not by activating anything here."""
        actor = _person(score=0.9, box=(30, 20, 70, 120))

        result = ActorAssociationService().associate([actor], 50, 20)

        self.assertIs(result.production.selected, actor)
        self.assertEqual(result.production_candidates, [actor])
        self.assertEqual(result.diagnostic()["version"], "rule-based")


if __name__ == "__main__":
    unittest.main()
