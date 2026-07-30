from __future__ import annotations

import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from yp_video.actor import labels as actor_labels
from yp_video.actor.labels import ActorLabel, ActorVerdict
from yp_video.core.cache import StatCache
from yp_video.core.jsonl import write_jsonl
from yp_video.extraction import store
from yp_video.extraction.prerequisites import Prerequisites
from yp_video.web.routers import actor_association


class CurrentActionJoinTests(unittest.TestCase):
    """Extraction is detector output; Action owns event meaning."""

    def test_current_action_replaces_stale_event_fields(self) -> None:
        with tempfile.TemporaryDirectory() as raw_dir:
            action = Path(raw_dir) / "match_actions.jsonl"
            write_jsonl(
                action,
                {"fps": 30},
                [
                    {
                        "id": "keep",
                        "frame": 30,
                        "time": 1.0,
                        "label": "set",
                        "xy": [0.2, 0.3],
                        "visible": False,
                    },
                    {
                        "id": "now-score",
                        "frame": 60,
                        "time": 2.0,
                        "label": "score",
                        "xy": [0.4, 0.5],
                        "visible": False,
                    },
                ],
            )
            stale = [
                {
                    "id": "keep",
                    "frame": 30,
                    "time": 0.9,
                    "label": "spike",
                    "xy": [0.1, 0.1],
                    "visible": True,
                    "detections": [{"box": [1, 2, 3, 4]}],
                },
                {
                    "id": "now-score",
                    "frame": 60,
                    "label": "spike",
                },
                {"id": "deleted", "frame": 90, "label": "serve"},
            ]

            with (
                patch.object(store, "action_annotation_path", return_value=action),
                patch.object(
                    store,
                    "load_rallies",
                    return_value=[{"start": 0.0, "end": 10.0}],
                ),
            ):
                records = store.labelable(stale, "match", 30.0)

        self.assertEqual([record["id"] for record in records], ["keep"])
        self.assertEqual(records[0]["label"], "set")
        self.assertEqual(records[0]["time"], 1.0)
        self.assertEqual(records[0]["xy"], [0.2, 0.3])
        self.assertFalse(records[0]["visible"])
        self.assertEqual(
            records[0]["detections"], [{"box": [1, 2, 3, 4]}]
        )

    def test_relabel_is_visible_without_rewriting_extraction(self) -> None:
        with tempfile.TemporaryDirectory() as raw_dir:
            action = Path(raw_dir) / "match_actions.jsonl"
            stored = [{"id": "event", "frame": 30, "label": "spike"}]
            with (
                patch.object(store, "action_annotation_path", return_value=action),
                patch.object(
                    store,
                    "load_rallies",
                    return_value=[{"start": 0.0, "end": 10.0}],
                ),
            ):
                write_jsonl(
                    action,
                    {"fps": 30},
                    [{"id": "event", "frame": 30, "label": "spike"}],
                )
                self.assertEqual(
                    [record["id"] for record in store.labelable(stored, "match", 30)],
                    ["event"],
                )

                # Only Action is rewritten. The derived extraction row remains
                # byte-for-byte stale and must immediately stop being work.
                write_jsonl(
                    action,
                    {"fps": 30},
                    [{"id": "event", "frame": 30, "label": "score"}],
                )
                self.assertEqual(store.labelable(stored, "match", 30), [])


class AssociationProgressTests(unittest.TestCase):
    def test_progress_counts_only_current_labelable_action_ids(self) -> None:
        with tempfile.TemporaryDirectory() as raw_dir:
            root = Path(raw_dir)
            video = root / "match.mp4"
            video.touch()
            records = root / "match.jsonl"
            action = root / "match_actions.jsonl"
            actors = root / "match_actors.json"
            write_jsonl(
                records,
                {"fps": 30, "events": 4, "ok": 3, "miss": 1},
                [
                    {"id": "reviewed", "frame": 30, "label": "set", "status": "ok"},
                    {"id": "pending", "frame": 60, "label": "set", "status": "miss"},
                    # Both are stale snapshots and must not block Done.
                    {"id": "now-score", "frame": 90, "label": "spike", "status": "ok"},
                    {"id": "outside", "frame": 900, "label": "serve", "status": "ok"},
                ],
            )
            write_jsonl(
                action,
                {"fps": 30},
                [
                    {"id": "reviewed", "frame": 30, "time": 1.0, "label": "set"},
                    {"id": "pending", "frame": 60, "time": 2.0, "label": "set"},
                    {"id": "now-score", "frame": 90, "time": 3.0, "label": "score"},
                    {"id": "outside", "frame": 900, "time": 30.0, "label": "serve"},
                ],
            )

            with (
                patch.object(actor_association, "iter_all_cuts", return_value=[video]),
                patch.object(store, "records_path", return_value=records),
                patch.object(store, "action_annotation_path", return_value=action),
                patch.object(
                    store,
                    "load_rallies",
                    return_value=[{"start": 0.0, "end": 10.0}],
                ),
                patch.object(actor_labels, "actors_path", return_value=actors),
                patch.object(actor_labels, "_cache", StatCache()),
                patch.object(
                    actor_association,
                    "prerequisites",
                    return_value=Prerequisites(
                        rally_sources=["annotation"],
                        has_action=True,
                        has_tracks=True,
                        has_masks=True,
                        tracks_stale=False,
                        has_records=True,
                    ),
                ),
            ):
                actor_labels.save(
                    "match",
                    "reviewed",
                    ActorLabel(ActorVerdict.CONFIRMED_AUTO),
                )
                # Old verdicts are durable, but do not count toward current
                # progress or its verdict breakdown.
                actor_labels.save(
                    "match",
                    "now-score",
                    ActorLabel(ActorVerdict.OCCLUDED),
                )
                result = actor_association.list_videos()[0]

        self.assertEqual(result["event_count"], 2)
        self.assertEqual(result["reviewed"], 1)
        self.assertEqual(result["unreviewed"], 1)
        self.assertEqual(result["verdicts"], {"confirmed_auto": 1})
        self.assertEqual(
            result["auto_counts"], {"ok": 3, "multi": 0, "miss": 1}
        )

    def test_a_video_missing_an_association_input_is_not_listed(self) -> None:
        """Extraction alone does not put a video on this list. Without actions
        there are no events, and without rallies there are no tracklet keys for
        an answer to name — a row would offer work that cannot be done here.
        """
        with tempfile.TemporaryDirectory() as raw_dir:
            root = Path(raw_dir)
            video = root / "match.mp4"
            video.touch()
            records = root / "match.jsonl"
            action = root / "match_actions.jsonl"
            write_jsonl(records, {"fps": 30}, [{"id": "a", "frame": 30, "label": "set"}])
            write_jsonl(action, {"fps": 30}, [{"id": "a", "frame": 30, "label": "set"}])
            complete = Prerequisites(
                rally_sources=["annotation"],
                has_action=True,
                has_tracks=True,
                has_masks=True,
                tracks_stale=False,
                has_records=True,
            )

            def listed(pipeline: Prerequisites, record_file: Path) -> list[dict]:
                with (
                    patch.object(
                        actor_association, "iter_all_cuts", return_value=[video]
                    ),
                    patch.object(
                        actor_association, "prerequisites", return_value=pipeline
                    ),
                    patch.object(store, "records_path", return_value=record_file),
                    patch.object(store, "action_annotation_path", return_value=action),
                    patch.object(
                        store, "load_rallies", return_value=[{"start": 0.0, "end": 10.0}]
                    ),
                    patch.object(actor_labels, "actors_path", return_value=root / "a.json"),
                    patch.object(actor_labels, "_cache", StatCache()),
                ):
                    return actor_association.list_videos()

            # The fixture is listable, so each removal below is the only cause.
            self.assertEqual(len(listed(complete, records)), 1)

            for missing, value in (("rally_sources", []), ("has_action", False)):
                with self.subTest(missing=missing):
                    gap = replace(complete, **{missing: value})
                    self.assertEqual(listed(gap, records), [])

            with self.subTest(missing="records"):
                self.assertEqual(listed(complete, root / "absent.jsonl"), [])
