-- Who did what, when, to which video.
--
-- One row per state-changing API call (web/audit.py) and per background job
-- status transition (web/jobs.py). `action` is the FastAPI route template, not
-- the concrete path, so the same operation on two videos groups together and
-- only a renamed endpoint shifts it.
--
-- `summary` holds a handful of numbers the route chose to record (rally
-- counts, files deleted, the job id it started); request payloads are
-- deliberately NOT stored.
--
-- `repeats` exists because the rally and action editors autosave every 2 s of
-- idle. Without it one labeling session writes hundreds of near-identical
-- rows and buries the events this table exists to surface. Consecutive
-- same-actor/same-action/same-target saves fold into one row instead.
--
-- No secondary indexes: `id` is monotonic with `at`, so the primary key's
-- btree scanned backwards already serves both the newest-first listing and its
-- keyset cursor. The filters scan, which is the right trade for a table this
-- size — add an index when EXPLAIN asks for one.
CREATE TABLE audit_events (
    id          BIGSERIAL   PRIMARY KEY,
    -- Set when the event happened, not when the writer drained it: now()
    -- would be the drain transaction's clock, which lags the request.
    at          TIMESTAMPTZ NOT NULL,
    actor       TEXT        NOT NULL,
    action      TEXT        NOT NULL,
    target      TEXT,
    summary     JSONB       NOT NULL DEFAULT '{}'::jsonb,
    outcome     TEXT        NOT NULL,   -- ok | error
    status      INT,                    -- HTTP status; NULL on job rows
    duration_ms INT,
    repeats     INT         NOT NULL DEFAULT 1
);
