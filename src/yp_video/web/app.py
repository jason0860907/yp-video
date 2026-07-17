"""Unified web application for volleyball video analysis."""

import logging
import signal
from contextlib import asynccontextmanager
from logging.handlers import RotatingFileHandler

from fastapi import FastAPI, HTTPException
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import FileResponse, ORJSONResponse
from fastapi.staticfiles import StaticFiles

from yp_video.config import APP_LOG_PATH, FRONTEND_DIST_DIR, LOGS_DIR
from yp_video.web.r2_client import r2_client
from yp_video.web.routers import (
    action_annotate,
    action_train,
    annotate,
    cut,
    detect,
    download,
    jobs,
    reid,
    spot_predict,
    spot_train,
    system,
    upload,
)
from yp_video.web.vllm_manager import vllm_manager


class _QuietPollFilter(logging.Filter):
    """Suppress uvicorn access logs for high-frequency polling endpoints."""

    _QUIET_PATHS = (
        "/api/system/stats",
        "/api/jobs/active-count",
        "/api/system/vllm/status",
        "/api/system/presence",
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


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Handle startup and shutdown."""
    print("Starting YP Video Analysis...")

    # Let uvicorn own SIGINT/SIGTERM so Ctrl+C and `make dev` shutdown
    # follow uvicorn's normal graceful server-close path.
    # Survive controlling-tty close (tmux pane exit, SSH disconnect without
    # nohup). Without this the default SIGHUP action terminates the process
    # mid-job and we lose hours of feature extraction.
    signal.signal(signal.SIGHUP, signal.SIG_IGN)

    # Eager-load R2 config so a misconfigured r2.env surfaces at boot
    # rather than on the first upload deep inside a job handler.
    r2_client.reload()
    print(f"R2: {'configured' if r2_client.configured else 'not configured (uploads will be skipped)'}")

    # Detect existing vLLM server
    await vllm_manager.initial_check()

    yield

    # Cleanup
    print("Shutting down...")


# orjson serializes the big numeric payloads (reid tracks ships ~100k boxes)
# 5-10x faster than stdlib json; every dict-returning route benefits.
app = FastAPI(title="YP Video Analysis", lifespan=lifespan, default_response_class=ORJSONResponse)

# Numeric JSON payloads (reid tracks ships ~100k boxes) compress 4-5x;
# small responses and streams (SSE, video ranges) pass through untouched.
# Level 4, not the default 9: on the 3.6 MB tracks payload that's 27 ms vs
# 197 ms for a 10% larger body — the right trade for a LAN tool.
app.add_middleware(GZipMiddleware, minimum_size=1024, compresslevel=4)

# Mount API routers
app.include_router(download.router, prefix="/api/download", tags=["download"])
app.include_router(cut.router, prefix="/api/cut", tags=["cut"])
app.include_router(action_annotate.router, prefix="/api/action-annotate", tags=["action-annotate"])
app.include_router(action_train.router, prefix="/api/action-train", tags=["action-train"])
app.include_router(annotate.router, prefix="/api/annotate", tags=["annotate"])
app.include_router(detect.router, prefix="/api/detect", tags=["detect"])
app.include_router(spot_train.router, prefix="/api/spot-train", tags=["spot-train"])
app.include_router(spot_predict.router, prefix="/api/spot-predict", tags=["spot-predict"])
app.include_router(reid.router, prefix="/api/reid", tags=["reid"])
app.include_router(jobs.router, prefix="/api/jobs", tags=["jobs"])
app.include_router(system.router, prefix="/api/system", tags=["system"])
app.include_router(upload.router, prefix="/api/upload", tags=["upload"])

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


def run_server(host: str = "0.0.0.0", port: int = 8080):
    """Run the unified app server."""
    import uvicorn
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    run_server()
