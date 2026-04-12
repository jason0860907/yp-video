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

    async def update_job(self, job_id: str, **kwargs):
        job = self.jobs.get(job_id)
        if not job:
            return
        for k, v in kwargs.items():
            if k == "status":
                job.status = JobStatus(v)
            elif hasattr(job, k):
                setattr(job, k, v)
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
) -> Callable[[int, int], None]:
    """Create a thread-safe progress callback for executor-based tasks.

    Args:
        job_id: Job to update.
        loop: The running event loop (from the async caller).
        message_template: Format string with {done} and {total} placeholders.

    Returns:
        A callback ``(done, total) -> None`` safe to call from any thread.
    """
    def callback(done: int, total: int) -> None:
        loop.call_soon_threadsafe(
            lambda d=done, t=total: asyncio.ensure_future(
                job_manager.update_job(
                    job_id,
                    progress=d / t if t else 0,
                    message=message_template.format(done=d, total=t),
                )
            )
        )
    return callback


# Module-level instance
job_manager = JobManager()
