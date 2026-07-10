"""Helpers shared across routers that spawn / orchestrate background jobs.

This module is the single source of truth for job display: message formats
(``batch_message``), terminal prefixes (``terminal_prefix``), and terminal
status epilogues (``finalize_batch_job``, ``fail_job_from_exc``) — routers
must not hand-roll their own variants.

The duplicated patterns factored out here:

1. ``stream_subprocess`` — spawn a subprocess, stream stdout into job.logs,
   parse lines for progress, and throttle SSE pushes. Used by SPOT train /
   predict jobs. Subprocess cancel is automatic via ``process.terminate()``,
   which kills the child and lets the OS reclaim its VRAM.
2. ``finalize_batch_job`` — set the all-success / all-failed / partial
   final status for a job that processed N items with K failures.
3. ``fail_job_from_exc`` — the standard failure epilogue (traceback into
   logs, ``error`` set, last progress message preserved).
4. ``stop_vllm_for_job`` — async context manager that releases vLLM's GPU
   for the duration of a job and restarts it in the background after.

Cancel semantics across all GPU-using jobs:

  Subprocess work (SPOT train / predict / pre-label):
      stream_subprocess catches CancelledError, calls process.terminate().
      OS reclaims subprocess VRAM. gpu_lock.__aexit__ then runs gc.collect
      + torch.cuda.empty_cache() in the parent.

  External vLLM detect:
      The vLLM server is a separate process and intentionally keeps
      its VRAM across job cancels — pass stop_vllm=True at the API
      layer if you actually want it stopped.
"""

from __future__ import annotations

import asyncio
import codecs
import logging
import re
import shlex
import time
import traceback
from collections.abc import Callable
from contextlib import asynccontextmanager
from pathlib import Path

from yp_video.web.jobs import job_manager

log = logging.getLogger(__name__)


# ── Message formatting ────────────────────────────────────────────────


def batch_message(index: int, total: int, item: str, detail: str = "") -> str:
    """Standard per-item progress message: ``(i/N) item: detail``.

    ``index`` is the 0-based loop variable (rendered 1-based). Telemetry like
    percent / speed / ETA goes in ``detail``, joined with `` · `` by the caller.
    """
    msg = f"({index + 1}/{total}) {item}"
    return f"{msg}: {detail}" if detail else msg


def terminal_prefix(job) -> str:
    """One prefix format for every job's terminal echo: ``[{type} {id}] ``."""
    return f"[{job.type} {job.id}] " if job is not None else ""


# ── Batch item tracking ───────────────────────────────────────────────
#
# Multi-video jobs publish per-item state through ``params["items"]`` so the
# frontend can render one expandable row per video. One shape for every job
# type: {video, status, progress, message, error?, started_at?, finished_at?}.

TERMINAL_ITEM_STATUSES = {"completed", "failed", "cancelled"}


def init_batch_items(videos: list[str]) -> list[dict]:
    """Fresh ``params["items"]`` for a batch job — one entry per video."""
    return [
        {"video": name, "status": "pending", "progress": 0.0, "message": "Pending"}
        for name in videos
    ]


def batch_counts(items: list[dict]) -> dict:
    return {
        "total": len(items),
        "completed": sum(1 for item in items if item.get("status") == "completed"),
        "failed": sum(1 for item in items if item.get("status") == "failed"),
        "cancelled": sum(1 for item in items if item.get("status") == "cancelled"),
    }


def batch_progress(index: int, item_progress: float, total: int) -> float:
    """Overall job fraction with item ``index`` at ``item_progress`` of its share."""
    return min(0.99, max(0.0, (index + item_progress) / max(1, total)))


def mark_batch_item(
    items: list[dict],
    index: int,
    *,
    status: str | None = None,
    progress: float | None = None,
    message: str | None = None,
    error: str | None = None,
    extra: dict | None = None,
) -> bool:
    """Mutate one batch item in place; the sync half of ``update_batch_item``.

    Stamps ``started_at`` on the first transition to running and
    ``finished_at`` on the first terminal status so the frontend can show
    when each video started and how long it took. Status-less updates on an
    already-terminal item are dropped (returns False) — late progress
    callbacks race the finalizer. Callers must republish via
    ``batch_items_params`` (or use ``update_batch_item``).
    """
    item = dict(items[index])
    if status is None and item.get("status") in TERMINAL_ITEM_STATUSES:
        return False
    if status is not None:
        item["status"] = status
        if status == "running":
            item.setdefault("started_at", time.time())
        elif status in TERMINAL_ITEM_STATUSES:
            item.setdefault("finished_at", time.time())
    if progress is not None:
        item["progress"] = max(0.0, min(float(progress), 1.0))
    if message is not None:
        item["message"] = message
    if error is not None:
        item["error"] = error
    if extra:
        item.update(extra)
    items[index] = item
    return True


def batch_items_params(items: list[dict]) -> dict:
    """The ``params`` fragment that publishes item state — merge over job.params."""
    return {"items": [dict(i) for i in items], **batch_counts(items)}


async def update_batch_item(
    job_id: str,
    items: list[dict],
    index: int,
    *,
    status: str | None = None,
    progress: float | None = None,
    message: str | None = None,
    error: str | None = None,
    overall_progress: float | None = None,
    overall_message: str | None = None,
    extra: dict | None = None,
) -> None:
    """Update one batch item and republish the job's ``params["items"]``."""
    if not mark_batch_item(
        items, index,
        status=status, progress=progress, message=message, error=error, extra=extra,
    ):
        return

    job = job_manager.get_job(job_id)
    if job is None:
        return
    update: dict = {"params": {**job.params, **batch_items_params(items)}}
    if overall_progress is not None:
        update["progress"] = max(float(job.progress), overall_progress)
    if overall_message is not None:
        update["message"] = overall_message
    await job_manager.update_job(job_id, **update)


# ── Subprocess streaming ──────────────────────────────────────────────


class ProgressParser:
    """Regex + handler that turns a log line into job-state updates.

    The handler receives the match object and returns either ``None`` (no
    update) or a dict of fields to merge into the throttled state. Typical
    fields are ``progress`` (0.0–1.0), ``message``, and/or ``params``. If the
    handler needs to remember state across lines (e.g. capture total epochs
    from one line and use it on later lines), close over it in the function.
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
    tee_to_terminal: bool = False,
    log_path: Path | str | None = None,
    log_line: Callable[[str], str | None] | None = None,
    log_command: str | None = None,
    update_job: bool = True,
) -> tuple[int, str]:
    """Spawn ``cmd``, stream its merged stdout/stderr, return ``(exit_code, last_line)``.

    Every line is appended to ``job.logs``. Parsers run on each line and may
    return updates that get merged into the throttled state. The job is
    updated via SSE at most once per ``push_interval`` seconds, with lines
    matching ``is_key_line`` pushed immediately so users see epoch/eval
    boundaries without waiting.

    When ``tee_to_terminal`` is set, every line is echoed to the terminal
    prefixed with ``[{job.type} {job.id}] `` (see ``terminal_prefix``) so all
    jobs share one prefix format.

    On ``CancelledError`` the subprocess is terminated and the exception is
    re-raised; the caller's outer except block is responsible for setting
    the cancelled status.
    """
    job_obj = job_manager.get_job(job_id)
    prefix = terminal_prefix(job_obj) if tee_to_terminal else ""
    log_fp = None
    if log_path is not None:
        log_file = Path(log_path)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        log_fp = log_file.open("a", encoding="utf-8")

    command_line = shlex.join(str(part) for part in cmd)
    if tee_to_terminal:
        print(f"{prefix}$ {command_line}", flush=True)
    if log_command:
        log.info("%s", log_command)
    if log_fp is not None:
        log_fp.write(f"$ {command_line}\n")
        log_fp.flush()

    process = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
        cwd=str(cwd),
        env=env,
    )
    state: dict = {"progress": 0.0, "message": "", "params": {}}
    last_msg = ""
    last_push = 0.0
    parsers = parsers or []
    is_key = is_key_line or (lambda _t: False)
    decoder = codecs.getincrementaldecoder("utf-8")("replace")

    async def handle_output(text: str) -> None:
        nonlocal last_msg, last_push

        text = text.rstrip()
        if not text:
            return
        last_msg = text
        if job_obj is not None:
            job_obj.logs.append(text)
        if tee_to_terminal:
            print(f"{prefix}{text}", flush=True)
        if log_line is not None:
            log_text = log_line(text)
            if log_text:
                log.info("%s", log_text)
        if log_fp is not None:
            log_fp.write(text + "\n")
            log_fp.flush()
        state["message"] = text
        for p in parsers:
            m = p.pattern.search(text)
            if m:
                update = p.handler(m)
                if update:
                    params_update = update.pop("params", None)
                    if params_update:
                        state["params"] = {**state["params"], **params_update}
                    state.update(update)
        now = time.monotonic()
        if update_job and (is_key(text) or now - last_push >= push_interval):
            last_push = now
            update_kwargs = {
                "message": state["message"],
                "progress": state["progress"],
            }
            if state["params"]:
                current_params = job_obj.params if job_obj is not None else {}
                update_kwargs["params"] = {**current_params, **state["params"]}
            await job_manager.update_job(job_id, **update_kwargs)

    async def flush_completed_chunks(buffer: str) -> str:
        while True:
            newline_idx = buffer.find("\n")
            carriage_idx = buffer.find("\r")
            indexes = [idx for idx in (newline_idx, carriage_idx) if idx >= 0]
            if not indexes:
                return buffer
            split_idx = min(indexes)
            await handle_output(buffer[:split_idx])
            buffer = buffer[split_idx + 1:]

    async def stop_process() -> None:
        if process.returncode is not None:
            return
        process.terminate()
        try:
            await asyncio.wait_for(process.wait(), timeout=5)
        except asyncio.TimeoutError:
            process.kill()
            await process.wait()

    try:
        buffer = ""
        max_buffer_chars = 64 * 1024
        while True:
            chunk = await process.stdout.read(8192)
            if not chunk:
                break
            buffer += decoder.decode(chunk)
            buffer = await flush_completed_chunks(buffer)
            if len(buffer) >= max_buffer_chars:
                await handle_output(buffer)
                buffer = ""
        buffer += decoder.decode(b"", final=True)
        buffer = await flush_completed_chunks(buffer)
        if buffer:
            await handle_output(buffer)
        rc = await process.wait()
        return rc, last_msg
    except asyncio.CancelledError:
        await stop_process()
        raise
    except Exception:
        await stop_process()
        raise
    finally:
        if log_fp is not None:
            log_fp.close()


# ── Batch job finalization ────────────────────────────────────────────


async def finalize_batch_job(
    job_id: str,
    total: int,
    failed: int,
    *,
    noun: str = "videos",
    name: str | None = None,
) -> None:
    """Set the terminal status / progress / message for a multi-item batch job.

    - failed == 0:        completed, "All {total} {noun} complete"
    - failed == total:    failed,    "All {total} {noun} failed — see logs"
    - otherwise:          completed, "{ok}/{total} completed, {failed} failed"

    ``params.total`` / ``params.failed`` are written so the frontend can detect
    partial success structurally instead of sniffing the message text.
    """
    job = job_manager.get_job(job_id)
    params = {**(job.params if job else {}), "total": total, "failed": failed}
    update: dict = {"progress": 1.0, "params": params}
    if name is not None:
        update["name"] = name
    if failed == 0:
        update.update(status="completed", message=f"All {total} {noun} complete")
    elif failed == total:
        update.update(status="failed", message=f"All {total} {noun} failed — see logs")
    else:
        update.update(
            status="completed",
            message=f"{total - failed}/{total} completed, {failed} failed",
        )
    await job_manager.update_job(job_id, **update)


async def fail_job_from_exc(job_id: str, exc: BaseException) -> None:
    """Standard failure epilogue for any job.

    The traceback goes into ``job.logs``, ``error`` becomes
    ``TypeName: message``. The last progress message is left in place so the
    user can see which step died; the frontend renders ``error`` separately.
    """
    job = job_manager.get_job(job_id)
    if job is not None:
        job.logs.extend("".join(traceback.format_exception(exc)).splitlines())
    await job_manager.update_job(
        job_id, status="failed", error=f"{type(exc).__name__}: {exc}"
    )


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
