from __future__ import annotations

import json
import tempfile
import unittest
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from fastapi import HTTPException
from pydantic import TypeAdapter

from yp_video.actor import labels as actor_labels
from yp_video.actor.labels import ActorLabel, ActorVerdict
from yp_video.core.cache import StatCache
from yp_video.core.jsonl import write_jsonl
from yp_video.extraction import actor_fix, board, done, pipeline, store
from yp_video.web.routers import actor_association as router


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
                patch.object(actor_labels._store, "_cache", StatCache()),
            ):
                yield path

    def test_verdict_survives_a_round_trip_and_is_never_inferred(
        self,
    ) -> None:
        with self._store() as path:
            actor_labels.save(
                "match",
                "manual-event",
                ActorLabel(ActorVerdict.MANUAL, frame=812, box=(1, 2, 3, 4)),
            )
            actor_labels.save(
                "match", "occluded-event", ActorLabel(ActorVerdict.OCCLUDED)
            )

            labels = actor_labels.load("match")
            self.assertEqual(
                labels["manual-event"],
                ActorLabel(ActorVerdict.MANUAL, frame=812, box=(1.0, 2.0, 3.0, 4.0)),
            )
            self.assertEqual(
                labels["occluded-event"].verdict, ActorVerdict.OCCLUDED
            )
            self.assertTrue(labels["manual-event"].human)

            stored = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(stored["version"], actor_labels.SCHEMA_VERSION)
            self.assertEqual(
                stored["actors"]["manual-event"],
                {"verdict": "manual", "frame": 812, "box": [1.0, 2.0, 3.0, 4.0]},
            )
            self.assertEqual(
                stored["actors"]["occluded-event"], {"verdict": "occluded"}
            )

    def test_a_label_must_carry_its_box_exactly_when_it_names_someone(self) -> None:
        for bad in (
            lambda: ActorLabel(ActorVerdict.MANUAL),
            lambda: ActorLabel(ActorVerdict.MANUAL, box=(1, 2, 3, 4)),
            lambda: ActorLabel(ActorVerdict.OCCLUDED, 3, (1, 2, 3, 4)),
            lambda: ActorLabel.from_payload({"verdict": "manual", "frame": 3, "box": [1, 2, 3]}),
            lambda: ActorLabel.from_payload({"verdict": "manual", "frame": 3, "box": [1, 2, 3, 4], "snap": False}),
        ):
            with self.assertRaises(ValueError):
                bad()

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

            added = actor_labels.confirm(
                "match",
                {
                    "fixed": ActorLabel(
                        ActorVerdict.CONFIRMED, 1, (1, 2, 3, 4)
                    ),
                    "untouched": ActorLabel(
                        ActorVerdict.CONFIRMED, 2, (5, 6, 7, 8)
                    ),
                },
            )

            labels = actor_labels.load("match")
            self.assertEqual(added, ['untouched'])
            self.assertEqual(labels["fixed"].verdict, ActorVerdict.OCCLUDED)
            self.assertEqual(
                labels["untouched"].verdict, ActorVerdict.CONFIRMED
            )

    def test_confirmation_replaces_an_agent_proposal(self) -> None:
        with self._store():
            actor_labels.save("match", "picked", ActorLabel(ActorVerdict.AGENT, 1, (1, 2, 3, 4)))
            actor_labels.save("match", "hidden", ActorLabel(ActorVerdict.AGENT_OCCLUDED))
            labels = actor_labels.load("match")
            self.assertFalse(labels["picked"].human)

            added = actor_labels.confirm(
                "match", {k: labels[k].confirmed() for k in ("picked", "hidden")}
            )

            labels = actor_labels.load("match")
            self.assertEqual(added, ["hidden", "picked"])
            self.assertEqual(labels["picked"], ActorLabel(ActorVerdict.CONFIRMED, 1, (1, 2, 3, 4)))
            self.assertEqual(labels["hidden"], ActorLabel(ActorVerdict.OCCLUDED))
            self.assertTrue(labels["hidden"].human)


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
        ), patch("yp_video.actor.box_style.dense_pass", return_value=None):
            confirmable = done.confirmable_actors("match", records)

        self.assertEqual(list(confirmable), ["auto-assigned"])
        self.assertEqual(
            confirmable["auto-assigned"],
            ActorLabel(ActorVerdict.CONFIRMED, 10, (1.0, 2.0, 3.0, 4.0)),
        )


@contextmanager
def _association_video(records: list[dict] | None, actions: list[dict]):
    """One scratch video as the Association endpoints read it: its action
    annotation, its extraction records (None = Player Detection never ran),
    an empty label store and a 30 fps dense pass."""
    with tempfile.TemporaryDirectory() as raw_dir:
        root = Path(raw_dir)
        record_file = root / "match.jsonl"
        action = root / "match_actions.jsonl"
        if records is not None:
            write_jsonl(record_file, {"video": "match"}, records)
        write_jsonl(action, {"video": "match"}, actions)
        with (
            patch.object(board, "records_path", return_value=record_file),
            patch.object(board, "action_annotation_path", return_value=action),
            patch.object(board, "_rows_cache", StatCache()),
            patch.object(store, "action_annotation_path", return_value=action),
            patch.object(store, "load_rallies", return_value=[]),
            patch.object(pipeline, "action_annotation_path", return_value=action),
            patch.object(actor_labels, "actors_path", return_value=root / "match_actors.json"),
            patch.object(actor_labels._store, "_cache", StatCache()),
            patch.object(router, "_dense", return_value=SimpleNamespace(fps=30.0)),
            patch("yp_video.actor.box_style.dense_pass", return_value=None),
            patch.object(router, "sync_to_r2"),
        ):
            yield root


class FixEndpointTests(unittest.TestCase):
    """The Association Label page's one write, wired end to end.

    The service is stubbed on purpose: this asserts the transport contract
    (mode → command, response shape, deferred refresh scheduled) without
    letting a test touch a real video's annotations.
    """

    def _fix(self, payload: dict, *, revision: int | None = 3) -> tuple[dict, tuple, list]:
        adapter = TypeAdapter(router.ActorFixRequest)
        applied: list[tuple] = []

        class _Tasks:
            def __init__(self) -> None:
                self.scheduled: list[tuple] = []

            def add_task(self, fn, *args, **kwargs) -> None:
                self.scheduled.append((fn, args, kwargs))

        tasks = _Tasks()

        def fake_apply(stem, frame_source, command):
            applied.append((command, frame_source))
            label = command.label_on(7)
            actor_labels.save(stem, command.event_id, label)
            return actor_fix.ActorFixResult(
                label=label,
                refreshing_models=("clip-reid",) if revision is not None else (),
                actor_revision=revision,
            )

        with _association_video(None, [{"id": "e1", "frame": 7, "label": "set"}]) as root:
            with (
                patch.object(router, "resolve_cut", return_value=root / "match.mp4"),
                patch.object(router, "cut_frame_source", return_value="https://r2.test/match.mp4"),
                patch.object(router.actor_fix, "apply", side_effect=fake_apply),
            ):
                response = router.fix(
                    "match.mp4", adapter.validate_python(payload), tasks  # type: ignore[arg-type]
                )
        return response, applied[0], tasks.scheduled

    def test_pick_reaches_the_service_as_a_manual_label(self) -> None:
        response, (command, frame_source), scheduled = self._fix(
            {"mode": "pick", "event_id": "e1", "box": [1, 2, 3, 4]}
        )

        self.assertEqual(
            command, actor_fix.PickActor(mode="pick", event_id="e1", box=(1, 2, 3, 4))
        )
        # An R2-only cut is read over its URL, not required on disk.
        self.assertEqual(frame_source, "https://r2.test/match.mp4")
        self.assertEqual(response["record"]["id"], "e1")
        self.assertEqual(response["record"]["actor_review"], "manual")
        self.assertEqual(response["refreshing_models"], ("clip-reid",))
        # Every matrix is refreshed after the response; unscheduled, they'd
        # stay silently stale.
        self.assertEqual(len(scheduled), 1)
        self.assertEqual(scheduled[0][2]["expected_revision"], 3)

    def test_without_records_nothing_is_left_to_refresh(self) -> None:
        response, _applied, scheduled = self._fix(
            {"mode": "occluded", "event_id": "e1"}, revision=None
        )

        self.assertEqual(response["record"]["actor_review"], "occluded")
        self.assertEqual(response["refreshing_models"], ())
        self.assertEqual(scheduled, [])

    def test_revert_reports_the_event_as_unreviewed_again(self) -> None:
        response, (command, _frame_source), _scheduled = self._fix(
            {"mode": "auto", "event_id": "e1"}
        )

        self.assertIsNone(command.label_on(7))
        self.assertEqual(response["record"]["actor_review"], "unreviewed")

    def test_a_pick_names_only_its_box(self) -> None:
        """The event frame is the server's to stamp; tracklets and snapping
        are gone from the label altogether."""
        adapter = TypeAdapter(router.ActorFixRequest)
        for extra in ({"frame": 7}, {"track": "1:2"}, {"snap": False}):
            with self.subTest(extra=extra), self.assertRaises(ValueError):
                adapter.validate_python({"mode": "pick", "event_id": "e1", "box": [1, 2, 3, 4], **extra})

    def test_a_video_without_a_dense_pass_is_a_404(self) -> None:
        adapter = TypeAdapter(router.ActorFixRequest)
        with tempfile.TemporaryDirectory() as raw_dir:
            with (
                patch.object(router, "resolve_cut", return_value=Path(raw_dir) / "m.mp4"),
                patch.object(router, "dense_pass", return_value=None),
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


class EventsEndpointTests(unittest.TestCase):
    def test_a_video_detection_never_ran_on_lists_every_event_unassociated(self) -> None:
        actions = [
            {"id": "a", "frame": 30, "label": "serve"},
            {"id": "b", "frame": 60, "label": "receive"},
            {"id": "c", "frame": 90, "label": "score"},
        ]
        with _association_video(None, actions):
            actor_labels.save("match", "b", ActorLabel(ActorVerdict.AGENT_OCCLUDED))
            dense = SimpleNamespace(fps=30.0, meta={"frame_size": [1920, 1080]})
            with patch.object(router, "_dense", return_value=dense):
                response = router.get_events("match.mp4")

        self.assertEqual(response["meta"]["fps"], 30.0)
        self.assertEqual(response["meta"]["frame_size"], (1920, 1080))
        self.assertEqual([e["id"] for e in response["meta"]["action_events"]], ["a", "b", "c"])
        # A score has nobody to identify.
        rows = {r["id"]: r for r in response["records"]}
        self.assertEqual(list(rows), ["a", "b"])
        self.assertEqual(rows["a"]["resolution"], "unresolved")
        self.assertEqual(rows["a"]["label"], "serve")
        self.assertEqual(rows["b"]["actor_review"], "agent_occluded")


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

    def _video(self, records=RECORDS):
        return _association_video(records, [{"id": r["id"], "frame": r["frame"]} for r in self.RECORDS])

    def test_confirms_automatic_picks_and_leaves_a_human_fix_alone(self) -> None:
        with self._video():
            # A verdict the user already gave: bulk confirmation must not
            # quietly overwrite it with "the machine was right".
            actor_labels.save("match", "auto-b", ActorLabel(ActorVerdict.OCCLUDED))

            response = router.confirm("match.mp4", router.ConfirmRequest(event_ids=["auto-a", "auto-b"]))
            labels = actor_labels.load("match")

        self.assertEqual(response["confirmed"], {"auto-a": "confirmed"})
        self.assertEqual(labels["auto-a"].verdict, ActorVerdict.CONFIRMED)
        self.assertEqual(labels["auto-a"].box, (1.0, 2.0, 3.0, 4.0))
        self.assertEqual(labels["auto-b"].verdict, ActorVerdict.OCCLUDED)
        # A manual fix already had a label; an unresolved event has no box to
        # agree with, whatever its diagnostic says.
        self.assertNotIn("miss", labels)
        self.assertNotIn("model-occluded", labels)

    def test_confirming_twice_is_a_no_op(self) -> None:
        with self._video():
            request = router.ConfirmRequest(event_ids=["auto-a", "auto-b"])
            first = router.confirm("match.mp4", request)
            second = router.confirm("match.mp4", request)

        self.assertEqual(
            first["confirmed"],
            {"auto-a": "confirmed", "auto-b": "confirmed"},
        )
        self.assertEqual(second["confirmed"], {})

    def test_a_model_occluded_diagnostic_is_not_confirmable(self) -> None:
        """Only a pick can be endorsed; an occlusion needs a human verdict."""
        with self._video():
            with self.assertRaises(HTTPException) as caught:
                router.confirm(
                    "match.mp4",
                    router.ConfirmRequest(event_ids=["model-occluded"]),
                )
            self.assertEqual(actor_labels.load("match"), {})

        self.assertEqual(caught.exception.status_code, 400)
        self.assertIn("model-occluded", str(caught.exception.detail))

    def test_confirms_an_agent_proposal_as_its_human_twin(self) -> None:
        with self._video():
            # The agent picked for the manual-resolution event, and called
            # the miss occluded.
            actor_labels.save("match", "fixed", ActorLabel(ActorVerdict.AGENT, 3, (9, 10, 11, 12)))
            actor_labels.save("match", "miss", ActorLabel(ActorVerdict.AGENT_OCCLUDED))

            response = router.confirm("match.mp4", router.ConfirmRequest(event_ids=["fixed", "miss"]))
            labels = actor_labels.load("match")

        self.assertEqual(response["confirmed"], {"fixed": "confirmed", "miss": "occluded"})
        self.assertEqual(labels["fixed"], ActorLabel(ActorVerdict.CONFIRMED, 3, (9.0, 10.0, 11.0, 12.0)))
        self.assertEqual(labels["miss"].verdict, ActorVerdict.OCCLUDED)

    def test_without_records_only_agent_proposals_are_confirmable(self) -> None:
        with self._video(records=None):
            actor_labels.save("match", "auto-a", ActorLabel(ActorVerdict.AGENT, 1, (1, 2, 3, 4)))
            with self.assertRaises(HTTPException) as caught:
                router.confirm("match.mp4", router.ConfirmRequest(event_ids=["auto-b"]))
            response = router.confirm("match.mp4", router.ConfirmRequest(event_ids=["auto-a"]))

        self.assertEqual(caught.exception.status_code, 400)
        self.assertEqual(response["confirmed"], {"auto-a": "confirmed"})

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


if __name__ == "__main__":
    unittest.main()
