"""Postgres connection pool for the audit trail.

Lives under `web/` deliberately. tests/test_layering.py already forbids every
domain package (person, tracklets, actor, reid, action) from importing
`yp_video.web`, so putting the database here means no domain package can grow
a dependency on it by accident. A top-level `yp_video.db` would have needed
that rule restated in five places.

Nothing is read from disk at import time: tests import `web.app` (see
RouterSurfaceTests) on machines without a .env, and a module that raises on
import would take them all down. Configuration is validated in the app
lifespan instead — the same shape r2_client uses.
"""

import logging
from typing import Any, LiteralString, cast

from psycopg import AsyncConnection
from psycopg_pool import AsyncConnectionPool

from yp_video.config import ENV_PATH, MIGRATIONS_DIR, load_env

log = logging.getLogger(__name__)

#: The pool's connection type. Spelled out because psycopg_pool's default is a
#: cast, which leaves the type parameter unsolved for a type checker.
Pool = AsyncConnectionPool[AsyncConnection[tuple[Any, ...]]]

_pool: Pool | None = None

_LEDGER = """
CREATE TABLE IF NOT EXISTS schema_migrations (
  version    TEXT PRIMARY KEY,
  applied_at TIMESTAMPTZ NOT NULL DEFAULT now()
)
"""

#: Arbitrary constant ("YPVI"), just has to be the same in every process.
_MIGRATION_LOCK = 0x59505649


class DatabaseNotConfigured(RuntimeError):
    """The workspace .env is missing YP_DB_URL."""


def _conninfo() -> str:
    url = load_env().get("YP_DB_URL", "").strip()
    if not url:
        raise DatabaseNotConfigured(
            f"YP_DB_URL is not set in {ENV_PATH}. Fill it in (see .env.example) "
            "and start the database with `make db-up`."
        )
    return url


async def open_pool(conninfo: str | None = None) -> None:
    """Open the pool and bring the schema up to date.

    Called from the app lifespan with no argument, which reads .env. Failing
    here stops the server on purpose: serving without an audit trail is the
    thing this feature exists to prevent.

    ``conninfo`` exists so the destructive database tests must name their
    scratch database explicitly. Defaulting them to .env once truncated a real
    trail.
    """
    global _pool
    if _pool is not None:
        return
    p: Pool = AsyncConnectionPool(
        conninfo or _conninfo(),
        connection_class=AsyncConnection,
        min_size=1,
        max_size=4,
        open=False,
    )
    await p.open(wait=True, timeout=15)
    _pool = p
    await _migrate(p)
    log.info("audit database ready")


async def close_pool() -> None:
    global _pool
    if _pool is None:
        return
    await _pool.close()
    _pool = None


def pool() -> Pool:
    """The open pool. Raises if the lifespan never opened it."""
    if _pool is None:
        raise DatabaseNotConfigured("database pool is not open")
    return _pool


async def _migrate(p: Pool) -> None:
    """Apply migrations/NNNN_*.sql once each, in filename order.

    Thirty lines instead of Alembic: there is one table and the only state
    worth keeping is which files have run. Everything happens in a single
    transaction — DDL is transactional in Postgres, so a bad boot leaves the
    schema exactly as it was. The advisory lock keeps two yp-app processes
    (`make serve` restarts on top of a lingering one) from racing.
    """
    async with p.connection() as conn, conn.transaction():
        await conn.execute("SELECT pg_advisory_xact_lock(%s)", (_MIGRATION_LOCK,))
        await conn.execute(_LEDGER)
        cur = await conn.execute("SELECT version FROM schema_migrations")
        done = {row[0] for row in await cur.fetchall()}
        for path in sorted(MIGRATIONS_DIR.glob("[0-9]*.sql")):
            if path.name in done:
                continue
            log.info("applying migration %s", path.name)
            # No parameters, so psycopg uses the simple protocol and accepts
            # several statements from one file.
            #
            # psycopg types queries as LiteralString to keep interpolated
            # values out of SQL. These come from migrations/ in this repo, not
            # from anything a request can reach, so the cast says so out loud
            # rather than routing checked-in DDL through a parameter.
            sql = cast(LiteralString, path.read_text(encoding="utf-8"))
            await conn.execute(sql)
            await conn.execute(
                "INSERT INTO schema_migrations (version) VALUES (%s)", (path.name,)
            )
