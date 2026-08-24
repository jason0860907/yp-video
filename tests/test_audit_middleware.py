"""Every state-changing call leaves a row, and it names the right thing.

The action is derived from the FastAPI route template rather than a hand-kept
table, so a new endpoint is audited the day it is written. The tests below pin
the parts of that derivation that are easy to get subtly wrong — and one that
already was: an unrouted POST partial-matches the SPA catch-all, and FastAPI
stamps `route` onto partial matches too.
"""

from __future__ import annotations

import unittest

from fastapi import FastAPI
from fastapi.testclient import TestClient

from yp_video.web import audit


class _Collector:
    """Stands in for the Postgres writer."""

    def __init__(self) -> None:
        self.events: list[audit.AuditEvent] = []

    def __call__(self, event: audit.AuditEvent) -> None:
        self.events.append(event)


def _app() -> FastAPI:
    app = FastAPI()

    @app.get("/api/things")
    def read() -> dict:
        return {"ok": True}

    @app.post("/api/things/{name}")
    def create(name: str) -> dict:  # noqa: ARG001 — target comes from the path
        return {"ok": True}

    @app.post("/api/annotate/annotations")
    async def save(body: dict) -> dict:
        audit.detail(target=body["video"], rallies=len(body["rallies"]))
        return {"ok": True}

    @app.post("/api/sync-detail")
    def sync_detail() -> dict:
        """A `def` endpoint: FastAPI runs it in a worker thread."""
        audit.detail(target="from-thread", n=7)
        return {"ok": True}

    @app.post("/api/unchanged")
    def unchanged() -> dict:
        audit.skip()
        return {"ok": True}

    @app.post("/api/unchanged-but-broken")
    def unchanged_but_broken() -> dict:
        audit.skip()
        raise ValueError("kaboom")

    @app.post("/api/system/presence")
    def presence() -> dict:
        return {"online": 1}

    @app.post("/api/boom")
    def boom() -> dict:
        raise ValueError("kaboom")

    @app.delete("/api/things/{name}")
    def remove(name: str) -> dict:  # noqa: ARG001
        return {"ok": True}

    # The real app's SPA catch-all, which an unrouted POST partial-matches.
    @app.get("/{full_path:path}")
    def spa(full_path: str) -> dict:  # noqa: ARG001
        return {"ok": True}

    app.add_middleware(audit.AuditTrail)
    return app


class AuditMiddlewareTests(unittest.TestCase):
    def setUp(self) -> None:
        self.collector = _Collector()
        self._saved = audit.record
        audit.record = self.collector
        self.client = TestClient(_app(), raise_server_exceptions=False)

    def tearDown(self) -> None:
        audit.record = self._saved

    @property
    def one(self) -> audit.AuditEvent:
        self.assertEqual(len(self.collector.events), 1, self.collector.events)
        return self.collector.events[0]

    def test_reads_are_not_recorded(self) -> None:
        self.client.get("/api/things")
        self.assertEqual(self.collector.events, [])

    def test_action_is_the_route_template_and_target_the_path_param(self) -> None:
        self.client.post("/api/things/set1.mp4")
        self.assertEqual(self.one.action, "POST /api/things/{name}")
        self.assertEqual(self.one.target, "set1.mp4")

    def test_two_videos_share_one_action(self) -> None:
        """Otherwise the filter list grows one entry per video, forever."""
        self.client.post("/api/things/a.mp4")
        self.client.post("/api/things/b.mp4")
        actions = {e.action for e in self.collector.events}
        self.assertEqual(actions, {"POST /api/things/{name}"})

    def test_delete_is_recorded(self) -> None:
        self.client.delete("/api/things/a.mp4")
        self.assertEqual(self.one.action, "DELETE /api/things/{name}")

    def test_detail_supplies_target_and_summary_from_the_body(self) -> None:
        self.client.post(
            "/api/annotate/annotations", json={"video": "set2", "rallies": [1, 2, 3]}
        )
        self.assertEqual(self.one.target, "set2")
        self.assertEqual(self.one.summary, {"rallies": 3})

    def test_detail_works_from_a_sync_endpoint(self) -> None:
        """The whole ContextVar design rests on this.

        FastAPI runs `def` endpoints in a worker thread; anyio copies the
        context into it, so mutating the dict the ContextVar points at
        propagates back. A `.set()` from inside the thread would not.
        """
        self.client.post("/api/sync-detail")
        self.assertEqual(self.one.target, "from-thread")
        self.assertEqual(self.one.summary, {"n": 7})

    def test_presence_heartbeat_is_excluded(self) -> None:
        """~2,900 rows per browser per day, and not an action anyone took."""
        self.client.post("/api/system/presence")
        self.assertEqual(self.collector.events, [])

    def test_unrouted_post_records_the_real_path_not_the_spa_catch_all(self) -> None:
        self.client.post("/api/nope")
        self.assertEqual(self.one.action, "POST /api/nope")
        self.assertNotIn("full_path", self.one.action)

    def test_a_write_that_changed_nothing_leaves_no_row(self) -> None:
        """The autosave case — see tests/test_audit_change_detection.py."""
        self.client.post("/api/unchanged")
        self.assertEqual(self.collector.events, [])

    def test_skip_does_not_hide_a_failure(self) -> None:
        """A refused write is an event whatever it did or did not touch."""
        self.client.post("/api/unchanged-but-broken")
        self.assertEqual(self.one.outcome, "error")

    def test_failed_request_is_recorded_as_an_error(self) -> None:
        self.client.post("/api/boom")
        self.assertEqual(self.one.outcome, "error")
        self.assertEqual(self.one.status, 500)

    def test_exception_still_propagates(self) -> None:
        """Auditing observes the failure; it must not swallow it."""
        client = TestClient(_app(), raise_server_exceptions=True)
        with self.assertRaises(ValueError):
            client.post("/api/boom")

    def test_rejected_request_is_recorded_as_an_error(self) -> None:
        """A 422 never reaches the handler, so the row is all the middleware's."""
        self.client.post("/api/annotate/annotations", json=[1, 2, 3])
        self.assertEqual(self.one.outcome, "error")
        self.assertEqual(self.one.status, 422)
        self.assertEqual(self.one.action, "POST /api/annotate/annotations")

    def test_timing_and_time_are_populated(self) -> None:
        self.client.post("/api/things/a.mp4")
        self.assertIsNotNone(self.one.duration_ms)
        self.assertGreaterEqual(self.one.duration_ms, 0)
        self.assertIsNotNone(self.one.at.tzinfo)

    def test_detail_outside_a_request_is_a_no_op(self) -> None:
        audit.detail(target="nowhere")  # must not raise


class CoalescingSetTests(unittest.TestCase):
    def test_autosaving_editors_are_the_coalesced_actions(self) -> None:
        """The rally and action editors autosave 2 s after editing stops.

        Left alone they write hundreds of near-identical rows per video and
        bury everything else, so those actions fold into one row instead.
        """
        self.assertIn("POST /api/annotate/annotations", audit._COALESCING)
        self.assertIn("POST /api/action-annotate/annotations", audit._COALESCING)


if __name__ == "__main__":
    unittest.main()
