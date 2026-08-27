"""Audit router — reading the trail.

Named audit_log, not audit, so that `from yp_video.web import audit` (the
machinery) and this module can both be imported into app.py without one name
shadowing the other.
"""

from datetime import datetime

from fastapi import APIRouter, HTTPException, Query

from yp_video.web import audit, db

router = APIRouter()

#: One page. The table is read newest-first and paged with a keyset cursor on
#: `id`, which is monotonic with `at` — so the primary key's btree, scanned
#: backwards, answers both without a second index.
_MAX_LIMIT = 200

_SELECT = """
SELECT id, first_at, at, actor, action, target, summary, outcome, status,
       duration_ms, repeats
  FROM audit_events
 WHERE (%(before)s::bigint IS NULL OR id < %(before)s)
   AND (%(actor)s::text   IS NULL OR actor = %(actor)s)
   AND (%(action)s::text  IS NULL OR action = %(action)s)
   AND (%(target)s::text  IS NULL OR target ILIKE '%%' || %(target)s || '%%')
   AND (%(since)s::timestamptz IS NULL OR at >= %(since)s)
   AND (%(until)s::timestamptz IS NULL OR first_at <= %(until)s)
 ORDER BY id DESC
 LIMIT %(limit)s
"""

#: first_at → at is the row's span; see SESSION_IDLE_GAP in web/audit.py.
_COLUMNS = (
    "id", "first_at", "at", "actor", "action", "target",
    "summary", "outcome", "status", "duration_ms", "repeats",
)


@router.get("/events")
async def list_events(
    actor: str | None = None,
    action: str | None = None,
    target: str | None = None,
    since: datetime | None = None,
    until: datetime | None = None,
    before: int | None = None,
    limit: int = Query(default=100, ge=1, le=_MAX_LIMIT),
) -> dict:
    """One page of events, newest first.

    ``before`` is the ``next_before`` of the previous page; omit it for the
    first. A full page comes back with a cursor, a short one without — that is
    how the caller knows it has reached the end.
    """
    params = {
        "actor": actor or None,
        "action": action or None,
        "target": target or None,
        "since": since,
        "until": until,
        "before": before,
        "limit": limit,
    }
    async with db.pool().connection() as conn:
        cur = await conn.execute(_SELECT, params)
        rows = await cur.fetchall()
    events = [dict(zip(_COLUMNS, row)) for row in rows]
    return {
        "events": events,
        "next_before": events[-1]["id"] if len(events) == limit else None,
    }


@router.get("/events/{event_id}/saves")
async def event_saves(event_id: int) -> dict:
    """Each save folded into one row: when it happened and what it changed.

    Fetched only when the reader expands a row. A session holds one entry per
    save with its item-level changes, which would dwarf the listing itself if
    it rode along with every page.
    """
    async with db.pool().connection() as conn:
        row = await (await conn.execute(
            "SELECT saves FROM audit_events WHERE id = %s", (event_id,)
        )).fetchone()
    if row is None:
        raise HTTPException(404, f"No audit event {event_id}")
    return {"saves": row[0]}


#: Gaps-and-islands over every save a person made, regardless of which video
#: or editor it went to: a new island starts wherever two consecutive saves are
#: more than SESSION_IDLE_GAP apart. Trail rows are folded per video, so
#: summing their spans would drop the minutes between videos — which is
#: exactly when someone is watching the next one.
_WORKLOG = f"""
WITH ticks AS (
  SELECT actor, (s->>'at')::timestamptz AS t
    FROM audit_events, jsonb_array_elements(saves) AS s
   WHERE action = ANY(%(actions)s)
     AND outcome = 'ok'
     AND at      >= %(since)s
     AND first_at <= %(until)s
), marked AS (
  SELECT actor, t,
         CASE WHEN t - lag(t) OVER w > interval '{audit.SESSION_IDLE_GAP}'
              THEN 1 ELSE 0 END AS starts
    FROM ticks
  WINDOW w AS (PARTITION BY actor ORDER BY t)
), islands AS (
  SELECT actor, t, sum(starts) OVER (PARTITION BY actor ORDER BY t) AS island
    FROM marked
), spans AS (
  SELECT actor, island, min(t) AS first_t, max(t) AS last_t, count(*) AS saves
    FROM islands
   GROUP BY actor, island
)
SELECT actor,
       count(*)   AS sessions,
       sum(saves) AS saves,
       COALESCE(sum(EXTRACT(EPOCH FROM (last_t - first_t))), 0)::bigint AS seconds
  FROM spans
 GROUP BY actor
 ORDER BY seconds DESC
"""


@router.get("/worklog")
async def worklog(since: datetime, until: datetime) -> dict:
    """Labeling time per person over a date range.

    A session is a run of one person's saves in which no two consecutive
    saves are more than SESSION_IDLE_GAP apart — across videos and across
    editors, since the quiet between two videos is usually spent watching the
    next one. The time worked is the sum of each session's first-to-last
    save. Only the labeling actions count (see audit.LABELING_ACTIONS) —
    everything else is instantaneous and would add zero while implying it was
    measured.

    Two caveats worth knowing before this settles anybody's week:

    - A session's clock starts at its FIRST save, not the first edit, so the
      couple of seconds before that autosave are not counted.
    - A session with a single save spans zero. Real work that produced exactly
      one save inside the idle gap therefore reads as 0, not as its true
      length; `saves` and `sessions` are there to make that visible.
    """
    params = {
        "actions": list(audit.LABELING_ACTIONS),
        "since": since,
        "until": until,
    }
    async with db.pool().connection() as conn:
        rows = await (await conn.execute(_WORKLOG, params)).fetchall()
    return {
        "since": since,
        "until": until,
        "people": [
            {"actor": a, "sessions": s, "saves": sv, "seconds": sec}
            for a, s, sv, sec in rows
        ],
    }


@router.get("/filters")
async def list_filters() -> dict:
    """What the trail actually contains, for the filter controls.

    One round trip rather than three endpoints the page would always call
    together. DISTINCT over the whole table is honest at this size and needs
    no index; revisit if it ever stops being instant.

    Targets are included so the page can offer the real video names instead of
    asking the reader to guess what to type.
    """
    async with db.pool().connection() as conn:
        actors = await (await conn.execute(
            "SELECT DISTINCT actor FROM audit_events ORDER BY 1"
        )).fetchall()
        actions = await (await conn.execute(
            "SELECT DISTINCT action FROM audit_events ORDER BY 1"
        )).fetchall()
        targets = await (await conn.execute(
            "SELECT DISTINCT target FROM audit_events "
            "WHERE target IS NOT NULL ORDER BY 1"
        )).fetchall()
    return {
        "actors": [r[0] for r in actors],
        "actions": [r[0] for r in actions],
        "targets": [r[0] for r in targets],
    }
