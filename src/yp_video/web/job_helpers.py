"""Helpers shared across routers that spawn / orchestrate background jobs.

Four duplicated patterns are factored out here:

1. ``stream_subprocess`` — spawn a subprocess, stream stdout into job.logs,
   parse lines for progress, and throttle SSE pushes. Used by TAD train +
   VLM train. Subprocess cancel is automatic via ``process.terminate()``,
   which kills the child and lets the OS reclaim its VRAM.
2. ``run_gpu_sync`` — run an in-process sync GPU function (V-JEPA / Action-
   Former) under the gpu_lock with a unified cancel + cache-flush flow.
   Used by feature extract + TAD predict. The two paths now share one
   shutdown protocol so cancelling either reliably frees the V-JEPA
   compiled module from VRAM before the lock releases.
3. ``finalize_batch_job`` — set the all-success / all-failed / partial
   final status for a job that processed N items with K failures. Used
   by predict + detect.
4. ``stop_vllm_for_job`` — async context manager that releases vLLM's GPU
   for the duration of a job and restarts it in the background after.

Cancel semantics across all GPU-using jobs:

  Subprocess work (TAD train / VLM train):
      stream_subprocess catches CancelledError, calls process.terminate().
      OS reclaims subprocess VRAM. gpu_lock.__aexit__ then runs gc.collect
      + torch.cuda.empty_cache() in the parent.

  In-process work (feature extract / TAD predict):
      run_gpu_sync sets a should_stop event the worker polls between
      batches/videos, waits for the worker thread to actually return
      (so gpu_lock isn't released while CUDA is mid-flight), runs
      on_cleanup (typically clear_model_cache to drop the cached
      V-JEPA module), then re-raises so gpu_lock.__aexit__ frees the
      torch caching allocator's blocks back to the driver.

  External vLLM detect:
      The vLLM server is a separate process and intentionally keeps
      its VRAM across job cancels — pass stop_vllm=True at the API
      layer if you actually want it stopped.
"""

from __future__ import annotations

import asyncio
import logging
import re
import threading
import time
from collections.abc import Callable
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

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


# ── Unified in-process GPU job lifecycle ──────────────────────────────


async def run_gpu_sync(
    job_id: str,
    sync_fn: Callable[[threading.Event], Any],
    *,
    stop_vllm: bool = False,
    on_cleanup: Callable[[], None] | None = None,
    cancel_message: str = "Cancelling — finishing current step...",
) -> Any:
    """Run a sync GPU function under gpu_lock with unified cancel + cleanup.

    Lifecycle in order:

      1. Yield vLLM if ``stop_vllm`` and vLLM was actually using the GPU.
      2. Acquire ``job_manager.gpu_lock``.
      3. Run ``sync_fn(should_stop)`` in a single executor thread. The
         callable must poll ``should_stop`` between batches/videos and
         return when it sees the flag set; nested ``run_in_executor``
         is unnecessary because everything inside ``sync_fn`` is already
         on a worker thread.
      4. On ``asyncio.CancelledError``: set ``should_stop``, surface
         ``cancel_message`` on the job, then **block on the worker
         thread until it actually returns** before letting the lock
         release. We use a fresh executor slot to wait on the
         ``worker_done`` event because the asyncio Future has already
         been cancelled at this point and isn't awaitable for the
         thread's true completion.
      5. Run ``on_cleanup`` (e.g. ``clear_model_cache``) so the cached
         V-JEPA compiled module drops its references *before*
         ``gpu_lock.__aexit__`` runs ``gc.collect()`` +
         ``torch.cuda.empty_cache()`` — without that ordering the
         allocator hangs onto V-JEPA's VRAM forever.
      6. ``gpu_lock`` releases (in ``finally``), restart vLLM if it was
         stopped.

    Returns whatever ``sync_fn`` returned. Re-raises ``CancelledError``
    after cleanup so the outer caller's except branch can flip the job
    to ``cancelled`` status.
    """
    should_stop = threading.Event()
    worker_done = threading.Event()

    def _wrapped() -> Any:
        try:
            return sync_fn(should_stop)
        finally:
            worker_done.set()

    async with stop_vllm_for_job(job_id, when=stop_vllm):
        async with job_manager.gpu_lock:
            loop = asyncio.get_event_loop()
            worker = loop.run_in_executor(None, _wrapped)
            try:
                result = await worker
            except asyncio.CancelledError:
                should_stop.set()
                try:
                    await job_manager.update_job(job_id, message=cancel_message)
                except Exception:
                    pass
                # Wait for the underlying thread to actually exit. Awaiting
                # `worker` again would just re-raise CancelledError because
                # the asyncio Future is now in cancelled state — block on a
                # fresh executor slot polling our own threading.Event.
                await loop.run_in_executor(None, worker_done.wait, 120.0)
                if on_cleanup:
                    try:
                        on_cleanup()
                    except Exception as e:  # noqa: BLE001
                        log.warning("on_cleanup failed for job %s: %s", job_id, e)
                raise
            else:
                # Success path: drop caches before gpu_lock's __aexit__ runs
                # gc.collect + empty_cache so the allocator can actually
                # reclaim V-JEPA's blocks.
                if on_cleanup:
                    try:
                        on_cleanup()
                    except Exception as e:  # noqa: BLE001
                        log.warning("on_cleanup failed for job %s: %s", job_id, e)
                return result


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
