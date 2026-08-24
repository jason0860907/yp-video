"""The audit trail: who did what, when, to which video.

Two producers feed one queue. The ASGI middleware records every state-changing
API call; JobManager records background job transitions. A single writer task
drains the queue into Postgres.

Nothing on the labeling path may wait on this. A save that succeeded on disk
must not report failure because the audit database hiccuped, and a slow INSERT
must not sit inside the request. So `record()` only enqueues, and the writer
absorbs every failure: a full queue drops the event, a broken connection drops
the row, and both leave a warning behind. Dropping audit rows is the lesser
harm, but it is still harm — the counter is logged so an unhealthy database is
visible rather than silent.
"""

import asyncio
import logging
import time
from collections.abc import Callable, Iterable, Mapping
from contextvars import ContextVar
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from yp_video.web import db
from yp_video.web.access import current_actor

log = logging.getLogger(__name__)

#: Heartbeat, not an action: usePresence beats every 30 s per browser, which
#: is ~2,900 rows a day each. app.py's _QuietPollFilter drops it from the
#: access log for the same reason.
_EXCLUDED_PATHS = frozenset({"/api/system/presence"})

#: HEAD and OPTIONS change nothing; "not GET" would sweep them in.
_MUTATING = frozenset({"POST", "PUT", "PATCH", "DELETE"})

#: The SPA catch-all. FastAPI attaches `route` to PARTIAL matches too, so an
#: unrouted POST /api/... lands on this template instead of a real endpoint.
_SPA_FALLBACK = "/{full_path:path}"

#: Path params to read a target from, best first.
_TARGET_KEYS = ("name", "job_id", "session_id", "video")

#: Actions the editors fire on a 2 s autosave timer. Consecutive saves of the
#: same thing fold into ONE row spanning first_at → at, so a row is a work
#: session rather than a keystroke. That is what makes the trail answer "how
#: long did this person label for", which is the point: the weekly settlement
#: reads these spans.
#:
#: Named explicitly so this stays a decision rather than a heuristic that
#: quietly swallows real edits — a deletion is never folded.
#: One entry per panel of the Label page — Rally, Action, Association, ReID.
#: Rally/Action/ReID autosave on a timer; Association fires once per event the
#: reviewer re-points. Either way it is a stretch of labeling, and folding it
#: is what makes an afternoon of work read as hours instead of as hundreds of
#: instantaneous rows totalling nothing.
_COALESCING = frozenset({
    "POST /api/annotate/annotations",
    "POST /api/action-annotate/annotations",
    "POST /api/actor-association/fix/{name}",
    "PUT /api/reid/players/{name}",
})

#: The same set, as the answer to "which rows are labeling work". These are the
#: only actions whose rows span time; everything else (a publish, a training
#: start, a delete) is instantaneous and would contribute nothing but noise to
#: a worked-hours total. The worklog query reads this rather than repeating the
#: list, so adding an autosaving editor extends both at once.
LABELING_ACTIONS = tuple(sorted(_COALESCING))

#: How long without a save ends a session. The next save after this much quiet
#: starts a new row, so a lunch break splits the day instead of being billed.
#: Measured from the last save in the row, not from its start — an afternoon of
#: continuous work stays one session.
_SESSION_IDLE_GAP = "5 minutes"

_QUEUE_MAX = 2000

_INSERT = """
INSERT INTO audit_events
  (at, first_at, actor, action, target, summary, outcome, status, duration_ms,
   saves)
VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, jsonb_build_array(%s))
"""

#: Extend the open session: move its end, keep its start, and remember when
#: this save happened. `first_at` is deliberately untouched — that is the whole
#: point of the column — and `saves` is what lets the UI expand a folded row
#: into the individual edits instead of only showing a count.
_COALESCE = f"""
UPDATE audit_events
   SET at = %s, summary = %s, duration_ms = %s, repeats = repeats + 1,
       saves = saves || %s
 WHERE id = (
       SELECT id FROM audit_events
        WHERE actor = %s AND action = %s
          AND target IS NOT DISTINCT FROM %s
          AND outcome = 'ok'
          AND at > %s - interval '{_SESSION_IDLE_GAP}'
        ORDER BY id DESC LIMIT 1)
"""


@dataclass(slots=True)
class AuditEvent:
    actor: str
    action: str
    at: datetime = field(default_factory=lambda: datetime.now(UTC))
    target: str | None = None
    summary: dict[str, Any] = field(default_factory=dict)
    outcome: str = "ok"
    status: int | None = None
    duration_ms: int | None = None
    #: What this one save changed (see Delta.changes). Appended to the row's
    #: `saves` array so a folded row can be expanded item by item.
    changes: list[dict[str, Any]] = field(default_factory=list)


# ── Enriching the current request's row ───────────────────────────
#
# A mutable dict held in a ContextVar, not a `request: Request` parameter on
# ten endpoints. uvicorn gives each request a fresh context and anyio copies
# that context into the worker thread it runs `def` endpoints in, so mutating
# the dict works from sync and async handlers alike — and a `.set()` from
# inside the thread, which would NOT propagate back, is never needed.

_detail: ContextVar[dict | None] = ContextVar("audit_detail", default=None)


def changed(items: list[dict[str, Any]]) -> None:
    """Attach this request's item-level changes (see `diff`).

    Kept apart from `detail`, whose contents become the row summary shown in
    every listing; these are the expandable detail and are fetched separately.
    """
    d = _detail.get()
    if d is not None:
        d["changes"] = items


def detail(*, target: str | None = None, **summary: Any) -> None:
    """Add what only this endpoint knows to its audit row.

    The middleware already records the actor, route, outcome and timing; this
    is for the target and counts that live in the request body. Outside a
    request (tests, CLI scripts) it does nothing.
    """
    d = _detail.get()
    if d is None:
        return
    if target is not None:
        d["target"] = target
    d["summary"].update(summary)


def skip() -> None:
    """Drop this request's audit row.

    For writes that turn out to change nothing. The editors autosave on a
    timer and flush again when the tab closes, so an unchanged save is a
    routine event — and a trail that logs "edited set1" when nobody edited
    anything is worse than no trail, because it cannot be trusted.

    Only for no-op writes. A refused or failed request is a real event and
    stays in the trail.
    """
    d = _detail.get()
    if d is not None:
        d["skip"] = True


#: A bulk edit could in principle change hundreds of items in one save. The
#: list is for reading, so it is bounded — but the overflow is COUNTED and
#: reported, never silently dropped.
_MAX_CHANGES_PER_SAVE = 200


@dataclass(slots=True)
class Delta:
    """What one save changed, item by item.

    Falsy when nothing changed, which is how the routers decide between
    recording a row and calling `skip()`.
    """

    counts: dict[str, int] = field(default_factory=dict)
    changes: list[dict[str, Any]] = field(default_factory=list)

    def __bool__(self) -> bool:
        return bool(self.counts)


def record_diff(
    *,
    target: str,
    before: Iterable[Mapping[str, Any]],
    after: Iterable[Mapping[str, Any]],
    key: Callable[[Mapping[str, Any]], Any],
    **summary: Any,
) -> Delta:
    """Diff two revisions and file the result — the whole path in one call.

    Every autosaving editor wants the same four things: the tally in the row
    summary, the item-level detail behind the ×N badge, and no row at all when
    the save changed nothing. Spelling that out per router meant a new editor
    could arrive with three of the four, which is exactly how ReID ended up
    counting unchanged saves as work.

    Returns the Delta so a caller can still branch on it.
    """
    delta = diff(before, after, key=key)
    if delta:
        detail(target=target, **delta.counts, **summary)
        changed(delta.changes)
    else:
        skip()
    return delta


def diff(
    before: Iterable[Mapping[str, Any]],
    after: Iterable[Mapping[str, Any]],
    *,
    key: Callable[[Mapping[str, Any]], Any],
) -> Delta:
    """How two revisions of a labeling file differ, keyed by *key*.

    Reports both the tally ("+2 -1 ~3", which the row summary shows) and the
    per-item detail, because "somebody edited this video" is not an answer to
    "what did they change". An edit records only the fields that actually
    moved, with their before and after values; an addition or removal records
    the item itself.
    """
    old = {key(row): row for row in before}
    new = {key(row): row for row in after}

    counts = {
        "added": len(new.keys() - old.keys()),
        "removed": len(old.keys() - new.keys()),
        "edited": sum(1 for k in old.keys() & new.keys() if old[k] != new[k]),
    }
    delta = Delta(counts={name: n for name, n in counts.items() if n})
    if not delta:
        return delta

    changes: list[dict[str, Any]] = []
    for k in new.keys() - old.keys():
        changes.append({"op": "added", "id": k, "item": dict(new[k])})
    for k in old.keys() - new.keys():
        changes.append({"op": "removed", "id": k, "item": dict(old[k])})
    for k in old.keys() & new.keys():
        was, now = old[k], new[k]
        if was == now:
            continue
        fields = {
            f: [was.get(f), now.get(f)]
            for f in set(was) | set(now)
            if was.get(f) != now.get(f)
        }
        changes.append({"op": "edited", "id": k, "fields": fields})

    # Stable order so a reader sees the same list twice.
    changes.sort(key=lambda c: (c["op"], str(c["id"])))
    if len(changes) > _MAX_CHANGES_PER_SAVE:
        dropped = len(changes) - _MAX_CHANGES_PER_SAVE
        changes = changes[:_MAX_CHANGES_PER_SAVE]
        changes.append({"op": "truncated", "count": dropped})
    delta.changes = changes
    return delta


# ── The queue ─────────────────────────────────────────────────────

_queue: "asyncio.Queue[AuditEvent] | None" = None
_task: asyncio.Task | None = None
_dropped = 0


def record(event: AuditEvent) -> None:
    """Enqueue one row. Never raises, never blocks, never waits on the DB."""
    global _dropped
    if _queue is None:
        return
    try:
        _queue.put_nowait(event)
    except asyncio.QueueFull:
        _dropped += 1
        log.warning(
            "audit queue full, dropped %s (%d dropped in total)", event.action, _dropped
        )


def start_writer() -> None:
    global _queue, _task
    if _task is not None:
        return
    _queue = asyncio.Queue(maxsize=_QUEUE_MAX)
    _task = asyncio.create_task(_drain(), name="audit-writer")


async def stop_writer() -> None:
    """Stop draining, then flush whatever is still queued."""
    global _queue, _task
    if _task is None:
        return
    task, _task = _task, None
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass
    queue, _queue = _queue, None
    while queue is not None and not queue.empty():
        await _write(queue.get_nowait())


async def _drain() -> None:
    assert _queue is not None
    while True:
        await _write(await _queue.get())


async def _write(event: AuditEvent) -> None:
    from psycopg.types.json import Jsonb

    summary = Jsonb(event.summary)
    # One entry per save: when it happened and what it changed.
    save = Jsonb({"at": event.at.isoformat(), "changes": event.changes})
    for attempt in (1, 2):
        try:
            async with db.pool().connection() as conn:
                if event.action in _COALESCING and event.outcome == "ok":
                    cur = await conn.execute(_COALESCE, (
                        event.at, summary, event.duration_ms, save,
                        event.actor, event.action, event.target, event.at,
                    ))
                    if cur.rowcount:
                        await conn.commit()
                        return
                # A new row starts and ends at the same instant; a second
                # save within the idle gap is what gives it a span.
                await conn.execute(_INSERT, (
                    event.at, event.at, event.actor, event.action, event.target,
                    summary, event.outcome, event.status, event.duration_ms,
                    save,
                ))
                await conn.commit()
            return
        except Exception as e:  # noqa: BLE001 — auditing must not raise upward
            if attempt == 2:
                log.warning("audit write failed, dropped %s: %s", event.action, e)
            else:
                await asyncio.sleep(0.2)


# ── Producers ─────────────────────────────────────────────────────


def job_transition(job, status: str) -> None:
    """Record a background job entering *status*.

    The actor comes off the Job, not the context: terminal transitions arrive
    from `threadsafe_update`, which hops through `loop.call_soon_threadsafe`
    and so runs in the loop's root context, where the request's ContextVar is
    long gone.

    `job.params` is never dumped wholesale — it carries per-file item lists
    that would dwarf the row.
    """
    summary: dict[str, Any] = {"job": job.id, "type": job.type}
    if job.error:
        summary["error"] = job.error[:200]
    record(AuditEvent(
        actor=job.actor or "(unknown)",
        action=f"job.{status}",
        target=job.name or job.type,
        summary=summary,
        outcome="error" if status == "failed" else "ok",
    ))


class AuditTrail:
    """Pure ASGI: one row per state-changing API call.

    The action is the FastAPI route template, read off the scope after the app
    has run — Starlette's router does `scope.update(child_scope)` on the very
    dict this middleware holds, and FastAPI's APIRoute puts itself in there.
    Deriving it rather than maintaining a route→name table means a new endpoint
    is audited the day it is written, and path parameters cannot explode one
    action into hundreds.

    Request bodies are deliberately not buffered: SSE, streaming uploads and
    video range requests all pass through here. What only the body knows comes
    from `detail()` instead.
    """

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if (
            scope["type"] != "http"
            or scope["method"] not in _MUTATING
            or not scope["path"].startswith("/api/")
            or scope["path"] in _EXCLUDED_PATHS
        ):
            await self.app(scope, receive, send)
            return

        status = 500
        collected: dict[str, Any] = {
            "target": None, "summary": {}, "skip": False, "changes": [],
        }
        token = _detail.set(collected)
        started = time.perf_counter()

        async def send_status(message):
            nonlocal status
            if message["type"] == "http.response.start":
                status = message["status"]
            await send(message)

        try:
            await self.app(scope, receive, send_status)
        except BaseException:
            # ServerErrorMiddleware is mounted outside this one, so an
            # unhandled error never reaches the send wrapper. Record it, then
            # let it carry on to the handler that turns it into a 500.
            self._record(scope, collected, 500, started)
            raise
        else:
            self._record(scope, collected, status, started)
        finally:
            _detail.reset(token)

    @staticmethod
    def _record(scope, collected, status: int, started: float) -> None:
        # A write that changed nothing asked to be dropped (see skip()). Only
        # honoured on success: a failure is an event whatever it touched.
        if collected["skip"] and status < 400:
            return
        route = scope.get("route")
        template = getattr(route, "path", None)
        # A POST to an unrouted /api path PARTIAL-matches the SPA catch-all,
        # and FastAPI stamps `route` onto partial matches too. Recording that
        # template would file every 404 under one meaningless action.
        action = scope["path"] if not template or template == _SPA_FALLBACK else template

        params = scope.get("path_params") or {}
        target = collected["target"] or next(
            (str(params[k]) for k in _TARGET_KEYS if k in params), None
        )
        record(AuditEvent(
            actor=current_actor() or "(unknown)",
            action=f"{scope['method']} {action}",
            target=target,
            summary=collected["summary"],
            changes=collected["changes"],
            outcome="ok" if status < 400 else "error",
            status=status,
            duration_ms=int((time.perf_counter() - started) * 1000),
        ))
