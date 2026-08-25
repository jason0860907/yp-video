"""Background jobs are attributed to whoever started them.

A job outlives the request that created it, and its terminal update arrives
from an executor thread via `loop.call_soon_threadsafe` — the loop's root
context, where the request's ContextVar is long gone. So the actor is stamped
onto the Job at creation and read back from there.

The cancellation test is a regression guard: `cancel_job` used to set the
status by hand and duplicate `update_job`'s broadcast, which made `update_job`
only *look* like the single funnel for status changes and left the single most
audit-worthy transition unrecorded.
"""

from __future__ import annotations

import unittest

from yp_video.web import access, audit
from yp_video.web.jobs import JobManager, JobStatus, JobType


class _Collector:
    def __init__(self) -> None:
        self.events: list[audit.AuditEvent] = []

    def __call__(self, event: audit.AuditEvent) -> None:
        self.events.append(event)


class JobAuditTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self.collector = _Collector()
        self._saved = audit.record
        audit.record = self.collector
        self.manager = JobManager()
        self._token = access._actor.set("labeler@example.com")

    def tearDown(self) -> None:
        audit.record = self._saved
        access._actor.reset(self._token)

    @property
    def actions(self) -> list[str]:
        return [e.action for e in self.collector.events]

    async def test_create_job_captures_the_current_actor(self) -> None:
        job = self.manager.create_job(JobType.SPOT_TRAIN, name="set1")
        self.assertEqual(job.actor, "labeler@example.com")

    async def test_running_then_completed_is_two_rows(self) -> None:
        job = self.manager.create_job(JobType.SPOT_TRAIN, name="set1")
        await self.manager.update_job(job.id, status=JobStatus.RUNNING)
        await self.manager.update_job(job.id, status=JobStatus.COMPLETED)
        self.assertEqual(self.actions, ["job.running", "job.completed"])
        self.assertTrue(all(e.actor == "labeler@example.com" for e in self.collector.events))
        self.assertEqual(self.collector.events[0].target, "set1")

    async def test_progress_ticks_are_not_recorded(self) -> None:
        """A training run reports progress hundreds of times."""
        job = self.manager.create_job(JobType.SPOT_TRAIN, name="set1")
        await self.manager.update_job(job.id, status=JobStatus.RUNNING)
        for pct in range(0, 100, 10):
            await self.manager.update_job(job.id, progress=pct / 100, message=f"{pct}%")
        self.assertEqual(self.actions, ["job.running"])

    async def test_repeated_running_records_once(self) -> None:
        job = self.manager.create_job(JobType.SPOT_TRAIN, name="set1")
        await self.manager.update_job(job.id, status="running")
        await self.manager.update_job(job.id, status="running", message="still going")
        self.assertEqual(self.actions, ["job.running"])

    async def test_cancel_job_records_exactly_one_cancellation(self) -> None:
        job = self.manager.create_job(JobType.SPOT_TRAIN, name="set1")
        await self.manager.update_job(job.id, status=JobStatus.RUNNING)
        self.assertTrue(await self.manager.cancel_job(job.id))
        self.assertEqual(self.actions, ["job.running", "job.cancelled"])
        self.assertIs(self.manager.get_job(job.id).status, JobStatus.CANCELLED)

    async def test_cancel_still_notifies_sse_subscribers(self) -> None:
        """Routing through update_job must not cost the UI its live update."""
        job = self.manager.create_job(JobType.SPOT_TRAIN, name="set1")
        await self.manager.update_job(job.id, status=JobStatus.RUNNING)
        queue = self.manager.subscribe(job.id)
        assert queue is not None
        while not queue.empty():
            queue.get_nowait()
        await self.manager.cancel_job(job.id)
        self.assertEqual(queue.get_nowait()["status"], "cancelled")

    async def test_cancelling_a_finished_job_records_nothing(self) -> None:
        job = self.manager.create_job(JobType.SPOT_TRAIN, name="set1")
        await self.manager.update_job(job.id, status=JobStatus.COMPLETED)
        self.assertFalse(await self.manager.cancel_job(job.id))
        self.assertEqual(self.actions, ["job.completed"])

    async def test_failure_carries_a_bounded_error_and_error_outcome(self) -> None:
        job = self.manager.create_job(JobType.SPOT_TRAIN, name="set1")
        await self.manager.update_job(
            job.id, status=JobStatus.FAILED, error="x" * 5000
        )
        event = self.collector.events[-1]
        self.assertEqual(event.outcome, "error")
        self.assertEqual(len(event.summary["error"]), 200)

    async def test_summary_never_carries_the_job_params(self) -> None:
        """params holds per-file item lists that would dwarf the row."""
        items = [{"name": f"f{i}.mp4", "status": "pending"} for i in range(500)]
        job = self.manager.create_job(
            JobType.R2_UPLOAD, params={"items": items}, name="upload"
        )
        await self.manager.update_job(job.id, status=JobStatus.RUNNING)
        self.assertEqual(
            set(self.collector.events[-1].summary), {"job", "type"}
        )


if __name__ == "__main__":
    unittest.main()
