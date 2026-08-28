"""Unified web application for volleyball video analysis."""

import logging
import signal
import threading
from contextlib import asynccontextmanager
from logging.handlers import RotatingFileHandler

from fastapi import FastAPI, HTTPException
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import FileResponse, ORJSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.datastructures import MutableHeaders

from yp_video.config import APP_LOG_PATH, FRONTEND_DIST_DIR, LOGS_DIR
from yp_video.core import label_done
from yp_video.web import audit, db, worklists
from yp_video.web.access import AccessAuth, verifier
from yp_video.web.r2_client import mirror_file, r2_client
from yp_video.web.routers import (
    action_annotate,
    actor_association,
    annotate,
    audit_log,
    cut,
    detect,
    download,
    extraction,
    fusion_model,
    jobs,
    label_stats,
    reid,
    reid_train,
    spot_predict,
    system,
    tracklets,
    upload,
)
from yp_video.web.vllm_manager import vllm_manager


class _QuietPollFilter(logging.Filter):
    """Suppress uvicorn access logs for high-frequency polling endpoints."""

    _QUIET_PATHS = (
        "/api/jobs/active-count",
        "/api/system/vllm/status",
        "/api/system/presence",
        # The sidebar's LabelProgress polls the four label work lists.
        "/api/annotate/results",
        "/api/action-annotate/videos",
        "/api/actor-association/videos",
        "/api/reid/videos",
    )

    def filter(self, record: logging.LogRecord) -> bool:
        msg = record.getMessage()
        return not any(p in msg for p in self._QUIET_PATHS)


def _configure_logging() -> None:
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    target = str(APP_LOG_PATH.resolve())
    for handler in root.handlers:
        if isinstance(handler, logging.FileHandler) and getattr(handler, "baseFilename", None) == target:
            return
    handler = RotatingFileHandler(
        APP_LOG_PATH,
        maxBytes=5_000_000,
        backupCount=3,
        encoding="utf-8",
    )
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    root.addHandler(handler)


_configure_logging()
logging.getLogger("uvicorn.access").addFilter(_QuietPollFilter())

log = logging.getLogger(__name__)


def _warm_worklists() -> None:
    """Prime the four label work lists (web/worklists.py) at startup.

    Deriving them cold parses every annotation and extraction records file
    (hundreds of MB) — tens of seconds the first visitor after a restart
    would otherwise eat. Warm, the list endpoints and /label/stats are
    stat-checks only. One thread, heaviest first, each failure logged
    without stopping the rest.
    """
    import time

    started = time.perf_counter()
    for warm in (
        worklists.association_videos,
        worklists.reid_videos,
        worklists.action_videos,
        worklists.rally_results,
    ):
        try:
            warm()
        except Exception:
            log.warning("worklist warm-up failed: %s", warm.__name__, exc_info=True)
    log.info("worklist warm-up done in %.1fs", time.perf_counter() - started)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Handle startup and shutdown."""
    print("Starting YP Video Analysis...")

    # Identity and the audit trail come up before anything can be served.
    # Both failures are fatal on purpose: serving without knowing who is
    # acting is exactly the state this app just left behind.
    verifier.configure()
    await db.open_pool()
    audit.start_writer()
    print("Audit: Cloudflare Access identity + Postgres trail ready")

    # Let uvicorn own SIGINT/SIGTERM so Ctrl+C and `make dev` shutdown
    # follow uvicorn's normal graceful server-close path.
    # Survive controlling-tty close (tmux pane exit, SSH disconnect without
    # nohup). Without this the default SIGHUP action terminates the process
    # mid-job and we lose hours of feature extraction.
    signal.signal(signal.SIGHUP, signal.SIG_IGN)

    # Eager-load R2 config so a misconfigured .env surfaces at boot
    # rather than on the first upload deep inside a job handler.
    r2_client.reload()
    print(f"R2: {'configured' if r2_client.configured else 'not configured (uploads will be skipped)'}")
    # Every Done click lands in R2 too; the ledger is human work no rerun rebuilds.
    label_done.ledger.on_write = lambda path: mirror_file(path, f"label-done/{path.name}")

    # Detect existing vLLM server
    await vllm_manager.initial_check()

    threading.Thread(target=_warm_worklists, name="warm-worklists", daemon=True).start()

    yield

    # Cleanup
    print("Shutting down...")
    await audit.stop_writer()
    await db.close_pool()


# orjson serializes the big numeric payloads (reid tracks ships ~100k boxes)
# 5-10x faster than stdlib json; every dict-returning route benefits.
app = FastAPI(title="YP Video Analysis", lifespan=lifespan, default_response_class=ORJSONResponse)

# Numeric JSON payloads (reid tracks ships ~100k boxes) compress 4-5x;
# small responses and SSE pass through untouched. Video ranges do NOT on
# their own — Starlette only exempts text/event-stream — so the video routes
# stamp Content-Encoding: identity (see r2_client.serve_video_or_r2_redirect).
# Level 4, not the default 9: on the 3.6 MB tracks payload that's 27 ms vs
# 197 ms for a 10% larger body — the right trade for a LAN tool.
app.add_middleware(GZipMiddleware, minimum_size=1024, compresslevel=4)


class _CachePolicy:
    """Say what may be cached, everywhere. Silence is what bites.

    FileResponse sets only etag/last-modified, so an unlabelled response gets
    heuristically cached by the browser AND by Cloudflare — which is how a
    fresh deploy keeps serving the previous SPA shell.

    Three rules:

    - ``/api/`` → ``no-store``. Cloudflare's default cache keys on the URL
      extension and most API paths end in a video filename
      (…/annotations/<name>.mp4), so the edge would serve stale JSON for its
      2-hour default TTL.
    - ``/assets/`` → immutable, one year. Vite content-hashes these names, so
      a changed file is a different URL; caching them forever is free.
    - everything else (the SPA shell) → ``no-cache``. Not no-store: the etag
      still lets a revalidation come back 304, but the shell is never served
      from cache without asking, so a new build is picked up immediately.

    Pure ASGI, not BaseHTTPMiddleware — the job SSE streams must pass through
    untouched.
    """

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        path = scope["path"]
        if path.startswith("/api/"):
            policy = "no-store"
        elif path.startswith("/assets/"):
            policy = "public, max-age=31536000, immutable"
        else:
            policy = "no-cache"

        async def send_with_policy(message):
            if message["type"] == "http.response.start":
                MutableHeaders(scope=message)["Cache-Control"] = policy
            await send(message)

        await self.app(scope, receive, send_with_policy)


app.add_middleware(_CachePolicy)

# add_middleware inserts at the front and the stack is built in reverse, so
# the LAST one added is the outermost. AccessAuth therefore runs first and a
# rejected request never reaches the audit trail — there is no identity on it
# to record.
app.add_middleware(audit.AuditTrail)
app.add_middleware(AccessAuth)

# Mount API routers
app.include_router(download.router, prefix="/api/download", tags=["download"])
app.include_router(cut.router, prefix="/api/cut", tags=["cut"])
app.include_router(action_annotate.router, prefix="/api/action-annotate", tags=["action-annotate"])
app.include_router(annotate.router, prefix="/api/annotate", tags=["annotate"])
app.include_router(detect.router, prefix="/api/detect", tags=["detect"])
app.include_router(spot_predict.router, prefix="/api/spot-predict", tags=["spot-predict"])
app.include_router(tracklets.router, prefix="/api/tracklets", tags=["tracklets"])
app.include_router(extraction.router, prefix="/api/extraction", tags=["extraction"])
app.include_router(fusion_model.router, prefix="/api/fusion-model", tags=["fusion-model"])
app.include_router(label_stats.router, prefix="/api/label", tags=["label"])
app.include_router(reid.router, prefix="/api/reid", tags=["reid"])
app.include_router(reid_train.router, prefix="/api/reid-train", tags=["reid-train"])
app.include_router(
    actor_association.router,
    prefix="/api/actor-association",
    tags=["actor-association"],
)
app.include_router(jobs.router, prefix="/api/jobs", tags=["jobs"])
app.include_router(system.router, prefix="/api/system", tags=["system"])
app.include_router(upload.router, prefix="/api/upload", tags=["upload"])
app.include_router(audit_log.router, prefix="/api/audit", tags=["audit"])

# ── Built React SPA (frontend/dist) ──────────────────────────────
# Hashed JS/CSS live under /assets; every other non-API path returns the
# shell so client-side routing survives a hard refresh.
_INDEX_FILE = FRONTEND_DIST_DIR / "index.html"
_DIST_READY = _INDEX_FILE.is_file()
_NOT_BUILT_MSG = (
    "Frontend not built. Run: cd src/yp_video/web/frontend && npm install && npm run build"
)

if _DIST_READY:
    app.mount("/assets", StaticFiles(directory=FRONTEND_DIST_DIR / "assets"), name="assets")


@app.get("/")
async def index():
    """Serve the SPA shell."""
    if not _DIST_READY:
        raise HTTPException(status_code=503, detail=_NOT_BUILT_MSG)
    return FileResponse(_INDEX_FILE)


@app.get("/{full_path:path}")
async def spa_fallback(full_path: str):
    """Serve built files when they exist, else the SPA shell (client routing)."""
    if full_path.startswith("api/"):
        raise HTTPException(status_code=404)
    if not _DIST_READY:
        raise HTTPException(status_code=503, detail=_NOT_BUILT_MSG)
    dist = FRONTEND_DIST_DIR.resolve()
    candidate = (dist / full_path).resolve()
    if candidate.is_file() and candidate.is_relative_to(dist):
        return FileResponse(candidate)
    return FileResponse(_INDEX_FILE)


def run_server(host: str = "127.0.0.1", port: int = 8080):
    """Run the unified app server.

    Loopback only: cloudflared is the sole client. Reaching the port directly
    would be an unauthenticated path around Cloudflare Access, which is the
    one hole the audit trail cannot tolerate.
    """
    import uvicorn
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    run_server()
