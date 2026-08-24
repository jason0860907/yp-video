"""The parts of the audit trail that need a real Postgres.

These tests TRUNCATE audit_events, so they refuse to touch the database the
app actually uses. They read their own variable, YP_AUDIT_TEST_DB_URL, and
abort if it names the same database as YP_DB_URL in the workspace .env:

    YP_AUDIT_TEST_DB_URL=postgresql://ypvideo:...@127.0.0.1:5433/ypvideo_test \
        uv run pytest tests/test_audit_db.py

Reusing YP_DB_URL here once wiped a real trail. A comment saying "point this
at a scratch database" was not enough, because the variable it read was the
one already set for the real one.
"""

from __future__ import annotations

import os
import unittest
from datetime import UTC, datetime, timedelta

from yp_video.config import load_env
from yp_video.web import audit, db


def _scratch_url() -> str | None:
    """The database these tests may destroy, or None to skip."""
    url = (os.environ.get("YP_AUDIT_TEST_DB_URL") or "").strip()
    if not url:
        return None
    live = load_env().get("YP_DB_URL", "").strip()
    if live and url == live:
        raise RuntimeError(
            "YP_AUDIT_TEST_DB_URL points at the live audit database (the "
            "YP_DB_URL in .env). These tests TRUNCATE audit_events — create a "
            "separate scratch database and point them at that."
        )
    return url


_URL = _scratch_url()


@unittest.skipUnless(_URL, "set YP_AUDIT_TEST_DB_URL to a scratch database")
class AuditDatabaseTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        # Never a bare open_pool(): that reads .env and opens the live one.
        await db.open_pool(conninfo=_URL)
        async with db.pool().connection() as conn:
            await conn.execute("TRUNCATE audit_events RESTART IDENTITY")
            await conn.commit()

    async def asyncTearDown(self) -> None:
        await db.close_pool()

    async def _rows(self) -> list[tuple]:
        async with db.pool().connection() as conn:
            cur = await conn.execute(
                "SELECT actor, action, target, summary, outcome, repeats "
                "FROM audit_events ORDER BY id"
            )
            return await cur.fetchall()

    async def test_migrations_are_idempotent(self) -> None:
        """open_pool runs them on every boot; a restart must be a no-op."""
        before = await self._applied()
        await db._migrate(db.pool())
        self.assertEqual(await self._applied(), before)

    async def _applied(self) -> set[str]:
        async with db.pool().connection() as conn:
            cur = await conn.execute("SELECT version FROM schema_migrations")
            return {r[0] for r in await cur.fetchall()}

    async def test_a_row_round_trips(self) -> None:
        await audit._write(audit.AuditEvent(
            actor="labeler@example.com",
            action="POST /api/upload/delete-r2",
            target="cuts-broadcast",
            summary={"deleted": 4},
            status=200,
            duration_ms=12,
        ))
        rows = await self._rows()
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0][:3], ("labeler@example.com", "POST /api/upload/delete-r2", "cuts-broadcast"))
        self.assertEqual(rows[0][3], {"deleted": 4})
        self.assertEqual(rows[0][5], 1)

    async def test_autosaves_fold_into_one_row(self) -> None:
        """Otherwise a single labeling session buries the rest of the trail."""
        for n in (12, 13, 14):
            await audit._write(audit.AuditEvent(
                actor="labeler@example.com",
                action="POST /api/annotate/annotations",
                target="set1",
                summary={"rallies": n},
            ))
        rows = await self._rows()
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0][5], 3)
        # The surviving row shows the latest state, not the first.
        self.assertEqual(rows[0][3], {"rallies": 14})

    async def test_a_different_video_is_a_different_row(self) -> None:
        for target in ("set1", "set2"):
            await audit._write(audit.AuditEvent(
                actor="labeler@example.com",
                action="POST /api/annotate/annotations",
                target=target,
                summary={"rallies": 5},
            ))
        self.assertEqual(len(await self._rows()), 2)

    async def test_a_different_actor_is_a_different_row(self) -> None:
        for actor in ("a@example.com", "b@example.com"):
            await audit._write(audit.AuditEvent(
                actor=actor,
                action="POST /api/annotate/annotations",
                target="set1",
                summary={"rallies": 5},
            ))
        self.assertEqual(len(await self._rows()), 2)

    async def test_an_old_save_does_not_absorb_a_new_one(self) -> None:
        """Coalescing is for one editing burst, not for all of history."""
        stale = datetime.now(UTC) - timedelta(hours=3)
        await audit._write(audit.AuditEvent(
            actor="labeler@example.com",
            action="POST /api/annotate/annotations",
            target="set1", summary={"rallies": 5}, at=stale,
        ))
        await audit._write(audit.AuditEvent(
            actor="labeler@example.com",
            action="POST /api/annotate/annotations",
            target="set1", summary={"rallies": 6},
        ))
        self.assertEqual(len(await self._rows()), 2)

    async def test_a_new_row_starts_and_ends_at_the_same_instant(self) -> None:
        await audit._write(audit.AuditEvent(
            actor="a@example.com", action="POST /api/annotate/annotations",
            target="set1", summary={"edited": 1},
        ))
        async with db.pool().connection() as conn:
            row = await (await conn.execute(
                "SELECT first_at, at FROM audit_events")).fetchone()
        assert row is not None
        self.assertEqual(row[0], row[1])

    async def test_folding_moves_the_end_and_keeps_the_start(self) -> None:
        """first_at is the session start — this is the whole point of it."""
        start = datetime.now(UTC) - timedelta(minutes=4)
        await audit._write(audit.AuditEvent(
            actor="a@example.com", action="POST /api/annotate/annotations",
            target="set1", summary={"edited": 1}, at=start,
        ))
        later = start + timedelta(minutes=3)
        await audit._write(audit.AuditEvent(
            actor="a@example.com", action="POST /api/annotate/annotations",
            target="set1", summary={"edited": 2}, at=later,
        ))
        async with db.pool().connection() as conn:
            rows = await (await conn.execute(
                "SELECT first_at, at, repeats FROM audit_events")).fetchall()
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0][0], start)   # start held
        self.assertEqual(rows[0][1], later)   # end advanced
        self.assertEqual(rows[0][2], 2)

    async def test_a_gap_longer_than_the_idle_limit_starts_a_new_session(self) -> None:
        """Five quiet minutes end the session, so a break is not billed."""
        first = datetime.now(UTC) - timedelta(minutes=20)
        await audit._write(audit.AuditEvent(
            actor="a@example.com", action="POST /api/annotate/annotations",
            target="set1", summary={"edited": 1}, at=first,
        ))
        after_break = first + timedelta(minutes=6)
        await audit._write(audit.AuditEvent(
            actor="a@example.com", action="POST /api/annotate/annotations",
            target="set1", summary={"edited": 1}, at=after_break,
        ))
        async with db.pool().connection() as conn:
            rows = await (await conn.execute(
                "SELECT first_at, at FROM audit_events ORDER BY id")).fetchall()
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0][0], first)
        self.assertEqual(rows[1][0], after_break)   # its own start

    async def test_saves_just_inside_the_limit_stay_one_session(self) -> None:
        """An afternoon of steady work must not fragment into dozens of rows."""
        t = datetime.now(UTC) - timedelta(minutes=30)
        for _ in range(6):
            await audit._write(audit.AuditEvent(
                actor="a@example.com", action="POST /api/annotate/annotations",
                target="set1", summary={"edited": 1}, at=t,
            ))
            t += timedelta(minutes=4)   # under the 5-minute gap
        async with db.pool().connection() as conn:
            rows = await (await conn.execute(
                "SELECT first_at, at, repeats FROM audit_events")).fetchall()
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0][2], 6)
        self.assertEqual((rows[0][1] - rows[0][0]).total_seconds(), 20 * 60)

    async def test_deletions_are_never_folded(self) -> None:
        """Two deletes are two events, however alike they look."""
        for _ in range(3):
            await audit._write(audit.AuditEvent(
                actor="labeler@example.com",
                action="POST /api/upload/delete-r2",
                target="cuts-broadcast",
                summary={"deleted": 1},
            ))
        self.assertEqual(len(await self._rows()), 3)

    async def test_a_failed_save_is_not_folded_into_a_successful_one(self) -> None:
        await audit._write(audit.AuditEvent(
            actor="labeler@example.com", action="POST /api/annotate/annotations",
            target="set1", summary={"rallies": 5},
        ))
        await audit._write(audit.AuditEvent(
            actor="labeler@example.com", action="POST /api/annotate/annotations",
            target="set1", summary={}, outcome="error", status=500,
        ))
        self.assertEqual(len(await self._rows()), 2)

    async def test_every_folded_save_is_kept_with_what_it_changed(self) -> None:
        """The ×N badge expands to these, so each save must carry its edit."""
        t = datetime.now(UTC) - timedelta(minutes=20)
        stamps = []
        for i in range(5):
            stamps.append(t)
            await audit._write(audit.AuditEvent(
                actor="a@example.com", action="POST /api/annotate/annotations",
                target="set1", summary={"edited": 1}, at=t,
                changes=[{"op": "edited", "id": 3, "fields": {"end": [40.0 + i, 41.0 + i]}}],
            ))
            t += timedelta(minutes=3)
        async with db.pool().connection() as conn:
            row = await (await conn.execute(
                "SELECT repeats, saves FROM audit_events")).fetchone()
        assert row is not None
        self.assertEqual(row[0], 5)
        saves = row[1]
        self.assertEqual(len(saves), 5)
        self.assertEqual(
            [datetime.fromisoformat(sv["at"]) for sv in saves], stamps
        )
        # Each save kept its OWN edit, not just the last one.
        self.assertEqual(
            [sv["changes"][0]["fields"]["end"] for sv in saves],
            [[40.0 + i, 41.0 + i] for i in range(5)],
        )

    async def test_a_new_session_starts_its_own_save_list(self) -> None:
        first = datetime.now(UTC) - timedelta(minutes=30)
        await audit._write(audit.AuditEvent(
            actor="a@example.com", action="POST /api/annotate/annotations",
            target="set1", summary={"edited": 1}, at=first,
            changes=[{"op": "added", "id": 1, "item": {"start": 1.0}}],
        ))
        await audit._write(audit.AuditEvent(
            actor="a@example.com", action="POST /api/annotate/annotations",
            target="set1", summary={"edited": 1}, at=first + timedelta(minutes=9),
            changes=[{"op": "removed", "id": 1, "item": {"start": 1.0}}],
        ))
        async with db.pool().connection() as conn:
            rows = await (await conn.execute(
                "SELECT saves FROM audit_events ORDER BY id")).fetchall()
        self.assertEqual([len(r[0]) for r in rows], [1, 1])
        self.assertEqual(
            [r[0][0]["changes"][0]["op"] for r in rows], ["added", "removed"]
        )

    async def _worklog(self, since, until):
        from yp_video.web.routers import audit_log

        return await audit_log.worklog(since=since, until=until)

    async def test_worklog_sums_each_persons_session_spans(self) -> None:
        """This is the number a week gets settled on."""
        base = datetime.now(UTC) - timedelta(hours=3)
        # Ann: one 20-minute session (six saves, four minutes apart).
        t = base
        for _ in range(6):
            await audit._write(audit.AuditEvent(
                actor="ann@example.com", action="POST /api/annotate/annotations",
                target="set1", summary={"edited": 1}, at=t,
            ))
            t += timedelta(minutes=4)
        # Bob: two sessions of 8 minutes, split by a long break.
        for offset in (0, 60):
            t = base + timedelta(minutes=offset)
            for _ in range(3):
                await audit._write(audit.AuditEvent(
                    actor="bob@example.com",
                    action="POST /api/action-annotate/annotations",
                    target="set2", summary={"edited": 1}, at=t,
                ))
                t += timedelta(minutes=4)

        log = await self._worklog(base - timedelta(hours=1), datetime.now(UTC))
        by_actor = {p["actor"]: p for p in log["people"]}
        self.assertEqual(by_actor["ann@example.com"]["seconds"], 20 * 60)
        self.assertEqual(by_actor["ann@example.com"]["sessions"], 1)
        self.assertEqual(by_actor["ann@example.com"]["saves"], 6)
        self.assertEqual(by_actor["bob@example.com"]["seconds"], 16 * 60)
        self.assertEqual(by_actor["bob@example.com"]["sessions"], 2)
        # Longest first, so the settlement reads top-down.
        self.assertEqual([p["actor"] for p in log["people"]],
                         ["ann@example.com", "bob@example.com"])

    async def test_worklog_ignores_instantaneous_actions(self) -> None:
        """A publish or a delete spans nothing and must not inflate a total."""
        now = datetime.now(UTC)
        for action in ("POST /api/annotate/publish",
                       "POST /api/upload/delete-r2",
                       "job.completed"):
            await audit._write(audit.AuditEvent(
                actor="ann@example.com", action=action, target="x", at=now,
            ))
        log = await self._worklog(now - timedelta(hours=1), now + timedelta(hours=1))
        self.assertEqual(log["people"], [])

    async def test_worklog_excludes_sessions_outside_the_range(self) -> None:
        last_week = datetime.now(UTC) - timedelta(days=9)
        await audit._write(audit.AuditEvent(
            actor="ann@example.com", action="POST /api/annotate/annotations",
            target="set1", summary={"edited": 1}, at=last_week,
        ))
        this_week = datetime.now(UTC) - timedelta(hours=1)
        await audit._write(audit.AuditEvent(
            actor="ann@example.com", action="POST /api/annotate/annotations",
            target="set1", summary={"edited": 1}, at=this_week,
        ))
        log = await self._worklog(this_week - timedelta(hours=2), datetime.now(UTC))
        self.assertEqual(log["people"][0]["sessions"], 1)

    async def test_a_write_survives_the_database_going_away(self) -> None:
        """The labeling path must not learn about audit failures."""
        await db.close_pool()
        await audit._write(audit.AuditEvent(actor="x", action="POST /api/x"))
        await db.open_pool(conninfo=_URL)  # asyncTearDown closes it again


if __name__ == "__main__":
    unittest.main()
