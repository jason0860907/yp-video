"""Unified web application for volleyball video analysis."""

import logging
import signal
import sys
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles


class _QuietPollFilter(logging.Filter):
    """Suppress uvicorn access logs for high-frequency polling endpoints."""

    _QUIET_PATHS = ("/api/system/stats", "/api/jobs/active-count", "/api/system/vllm/status")

    def filter(self, record: logging.LogRecord) -> bool:
        msg = record.getMessage()
        return not any(p in msg for p in self._QUIET_PATHS)


logging.getLogger("uvicorn.access").addFilter(_QuietPollFilter())

from yp_video.config import STATIC_DIR
from yp_video.web.routers import download, cut, annotate, detect, train, predict, jobs, system, upload
from yp_video.web.vllm_manager import vllm_manager


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Handle startup and shutdown."""
    print("Starting YP Video Analysis...")

    def handle_signal(signum, frame):
        print(f"\nReceived signal {signum}, shutting down...")
        sys.exit(0)

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    # Detect existing vLLM server
    await vllm_manager.initial_check()

    yield

    # Cleanup
    print("Shutting down...")


app = FastAPI(title="YP Video Analysis", lifespan=lifespan)

# Mount API routers
app.include_router(download.router, prefix="/api/download", tags=["download"])
app.include_router(cut.router, prefix="/api/cut", tags=["cut"])
app.include_router(annotate.router, prefix="/api/annotate", tags=["annotate"])
app.include_router(detect.router, prefix="/api/detect", tags=["detect"])
app.include_router(train.router, prefix="/api/train", tags=["train"])
app.include_router(predict.router, prefix="/api/predict", tags=["predict"])
app.include_router(jobs.router, prefix="/api/jobs", tags=["jobs"])
app.include_router(system.router, prefix="/api/system", tags=["system"])
app.include_router(upload.router, prefix="/api/upload", tags=["upload"])

# Mount static files
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/")
async def index():
    """Serve the SPA shell."""
    return FileResponse(STATIC_DIR / "index.html")


def run_server(host: str = "0.0.0.0", port: int = 8080):
    """Run the unified app server."""
    import uvicorn
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    run_server()
