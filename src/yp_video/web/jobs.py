"""Background job manager with GPU lock."""

import asyncio
import logging
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum

log = logging.getLogger(__name__)


class JobStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class Job:
    id: str
    type: str
    name: str = ""
    status: JobStatus = JobStatus.PENDING
    progress: float = 0.0
    message: str = ""
    params: dict = field(default_factory=dict)
    error: str | None = None
    logs: list[str] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    _task: asyncio.Task | None = field(default=None, repr=False)
    _subscribers: list[asyncio.Queue] = field(default_factory=list, repr=False)

    def set_task(self, task: asyncio.Task) -> None:
        """Attach an asyncio task so the job can be cancelled."""
        self._task = task

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "type": self.type,
            "name": self.name,
            "status": self.status.value,
            "progress": self.progress,
            "message": self.message,
            "params": self.params,
            "error": self.error,
            "logs": self.logs,
            "created_at": self.created_at,
        }


class _GpuLock:
    """Async lock that auto-releases GPU memory on exit."""

    def __init__(self):
        self._lock = asyncio.Lock()

    async def __aenter__(self):
        await self._lock.acquire()
        return self

    async def __aexit__(self, *exc):
        self._lock.release()
        import gc
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass


class JobManager:
    """In-memory job manager with GPU lock for mutual exclusion."""

    def __init__(self):
        self.jobs: dict[str, Job] = {}
        self.gpu_lock = _GpuLock()
        self._vllm_using_gpu = False

    def create_job(self, job_type: str, params: dict | None = None, name: str = "") -> Job:
        job = Job(
            id=str(uuid.uuid4())[:8],
            type=job_type,
            name=name,
            params=params or {},
        )
        self.jobs[job.id] = job
        return job

    def get_job(self, job_id: str) -> Job | None:
        return self.jobs.get(job_id)

    def list_jobs(self) -> list[dict]:
        return [j.to_dict() for j in sorted(
            self.jobs.values(), key=lambda j: j.created_at, reverse=True
        )]

    def active_count(self) -> int:
        return sum(1 for j in self.jobs.values() if j.status == JobStatus.RUNNING)

    async def update_job(
        self,
        job_id: str,
        *,
        status: str | JobStatus | None = None,
        progress: float | None = None,
        message: str | None = None,
        error: str | None = None,
        name: str | None = None,
        params: dict | None = None,
    ) -> None:
        """Update mutable job fields and broadcast the new state to subscribers.

        Only the listed fields are writable from outside; private state
        (``_task``, ``_subscribers``, ``logs``) must be touched directly on
        the ``Job`` (logs append) or via ``cancel_job`` / ``attach_task``.

        ``params`` replaces the dict wholesale (callers usually do
        ``params={**job.params, "bytes_done": …}`` to merge). Used by the
        upload/download routers to pipe byte/speed/ETA telemetry through to
        the frontend without a separate channel.
        """
        job = self.jobs.get(job_id)
        if not job:
            return
        if status is not None:
            job.status = JobStatus(status) if not isinstance(status, JobStatus) else status
        if progress is not None:
            job.progress = progress
        if message is not None:
            job.message = message
        if error is not None:
            job.error = error
        if name is not None:
            job.name = name
        if params is not None:
            job.params = params
        # Notify SSE subscribers
        event = job.to_dict()
        for q in job._subscribers:
            try:
                q.put_nowait(event)
            except asyncio.QueueFull:
                log.debug("SSE queue full for job %s, dropping event", job.id)

    def subscribe(self, job_id: str) -> asyncio.Queue | None:
        job = self.jobs.get(job_id)
        if not job:
            return None
        q: asyncio.Queue = asyncio.Queue(maxsize=500)
        job._subscribers.append(q)
        # Send current state immediately
        q.put_nowait(job.to_dict())
        return q

    def unsubscribe(self, job_id: str, q: asyncio.Queue):
        job = self.jobs.get(job_id)
        if job and q in job._subscribers:
            job._subscribers.remove(q)

    async def cancel_job(self, job_id: str) -> bool:
        job = self.jobs.get(job_id)
        if not job or job.status != JobStatus.RUNNING:
            return False
        if job._task:
            job._task.cancel()
        job.status = JobStatus.CANCELLED
        job.message = "Cancelled by user"
        # Notify subscribers
        event = job.to_dict()
        for q in job._subscribers:
            try:
                q.put_nowait(event)
            except asyncio.QueueFull:
                log.debug("SSE queue full for job %s, dropping event", job.id)
        return True

    def attach_task(self, jobs: "list[Job] | Job", task: asyncio.Task) -> None:
        """Attach a cancellable task to one or more jobs."""
        if isinstance(jobs, Job):
            jobs = [jobs]
        for job in jobs:
            job.set_task(task)

    @property
    def vllm_using_gpu(self) -> bool:
        return self._vllm_using_gpu

    @vllm_using_gpu.setter
    def vllm_using_gpu(self, value: bool):
        self._vllm_using_gpu = value


def make_progress_callback(
    job_id: str,
    loop: asyncio.AbstractEventLoop,
    message_template: str = "Progress ({done}/{total})",
    *,
    manager: "JobManager | None" = None,
) -> Callable[..., None]:
    """Create a thread-safe ``(done, total[, msg]) -> None`` progress callback.

    Used by sync code running in ``run_in_executor`` to push progress back to
    a job. ``loop`` is the async caller's event loop (the callback may fire
    from any thread). ``manager`` defaults to the module-level ``job_manager``;
    pass an explicit one if you ever construct your own JobManager (e.g. tests).

    The optional third positional ``msg`` lets the caller override the
    formatted template — useful when the natural progress unit is "videos
    completed" (filling the template) but in between completions the caller
    wants to surface "currently processing X" without bumping the count.
    Pass a fractional ``done`` (e.g. 2.4 of 227) to render sub-item progress.
    """
    mgr = manager if manager is not None else job_manager

    def callback(done: float, total: float, msg: str | None = None) -> None:
        rendered = msg if msg is not None else message_template.format(
            done=int(done) if done == int(done) else done,
            total=int(total) if total == int(total) else total,
        )
        loop.call_soon_threadsafe(
            lambda d=done, t=total, m=rendered: asyncio.ensure_future(
                mgr.update_job(
                    job_id,
                    progress=d / t if t else 0,
                    message=m,
                )
            )
        )
    return callback


# Module-level instance
job_manager = JobManager()
