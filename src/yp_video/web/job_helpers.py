"""Helpers shared across routers that spawn / orchestrate background jobs.

Three duplicated patterns are factored out here:

1. ``stream_subprocess`` — spawn a subprocess, stream stdout into job.logs,
   parse lines for progress, and throttle SSE pushes. Used by train + vlm.
2. ``finalize_batch_job`` — set the all-success / all-failed / partial
   final status for a job that processed N items with K failures. Used
   by predict + detect.
3. ``stop_vllm_for_job`` — async context manager that releases vLLM's GPU
   for the duration of a job and restarts it in the background after.
"""

from __future__ import annotations

import asyncio
import logging
import re
import time
from collections.abc import Callable
from contextlib import asynccontextmanager
from pathlib import Path

from yp_video.web.jobs import job_manager

log = logging.getLogger(__name__)


# ── Subprocess streaming ──────────────────────────────────────────────


class ProgressParser:
    """Regex + handler that turns a log line into job-state updates.

    The handler receives the match object and returns either ``None`` (no
    update) or a dict of fields to merge into the throttled state. Typical
    fields are ``progress`` (0.0–1.0) and/or ``message``. If the handler
    needs to remember state across lines (e.g. capture total epochs from one
    line and use it on later lines), close over it in the function.
    """

    def __init__(self, pattern: str | re.Pattern, handler: Callable[[re.Match], dict | None]):
        self.pattern = re.compile(pattern) if isinstance(pattern, str) else pattern
        self.handler = handler


async def stream_subprocess(
    job_id: str,
    cmd: list[str],
    cwd: Path | str,
    *,
    env: dict | None = None,
    parsers: list[ProgressParser] | None = None,
    is_key_line: Callable[[str], bool] | None = None,
    push_interval: float = 1.0,
) -> tuple[int, str]:
    """Spawn ``cmd``, stream its merged stdout/stderr, return ``(exit_code, last_line)``.

    Every line is appended to ``job.logs``. Parsers run on each line and may
    return updates that get merged into the throttled state. The job is
    updated via SSE at most once per ``push_interval`` seconds, with lines
    matching ``is_key_line`` pushed immediately so users see epoch/eval
    boundaries without waiting.

    On ``CancelledError`` the subprocess is terminated and the exception is
    re-raised; the caller's outer except block is responsible for setting
    the cancelled status.
    """
    process = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
        cwd=str(cwd),
        env=env,
    )
    state = {"progress": 0.0, "message": ""}
    last_msg = ""
    last_push = 0.0
    job_obj = job_manager.get_job(job_id)
    parsers = parsers or []
    is_key = is_key_line or (lambda _t: False)

    try:
        while True:
            line = await process.stdout.readline()
            if not line:
                break
            text = line.decode().rstrip()
            if not text:
                continue
            last_msg = text
            if job_obj is not None:
                job_obj.logs.append(text)
            state["message"] = text
            for p in parsers:
                m = p.pattern.search(text)
                if m:
                    update = p.handler(m)
                    if update:
                        state.update(update)
            now = time.monotonic()
            if is_key(text) or now - last_push >= push_interval:
                last_push = now
                await job_manager.update_job(
                    job_id, message=state["message"], progress=state["progress"],
                )
        rc = await process.wait()
        return rc, last_msg
    except asyncio.CancelledError:
        if process.returncode is None:
            process.terminate()
        raise


# ── Batch job finalization ────────────────────────────────────────────


async def finalize_batch_job(
    job_id: str,
    total: int,
    failed: int,
    *,
    name: str | None = None,
) -> None:
    """Set the terminal status / progress / message for a multi-item batch job.

    - failed == 0:        completed, "All {total} videos complete"
    - failed == total:    failed,    "All {total} videos failed — see logs"
    - otherwise:          completed, "{ok}/{total} completed, {failed} failed"
    """
    update: dict = {"progress": 1.0}
    if name is not None:
        update["name"] = name
    if failed == 0:
        update.update(status="completed", message=f"All {total} videos complete")
    elif failed == total:
        update.update(status="failed", message=f"All {total} videos failed — see logs")
    else:
        update.update(
            status="completed",
            message=f"{total - failed}/{total} completed, {failed} failed",
        )
    await job_manager.update_job(job_id, **update)


# ── vLLM GPU yielding ─────────────────────────────────────────────────


@asynccontextmanager
async def stop_vllm_for_job(job_id: str, *, when: bool):
    """Release vLLM's GPU for the duration of a job, restart on exit.

    No-op when ``when`` is False or vLLM isn't currently using the GPU. The
    restart is fire-and-forget so the job can return to the user immediately
    without waiting for vLLM to be ready again.
    """
    if not when or not job_manager.vllm_using_gpu:
        yield
        return
    from yp_video.web.vllm_manager import vllm_manager
    await job_manager.update_job(job_id, message="Stopping vLLM to free VRAM...")
    await vllm_manager.stop()
    try:
        yield
    finally:
        log.info("Auto-restarting vLLM after job %s", job_id)
        asyncio.create_task(vllm_manager.start())
