"""Background job manager with GPU lock."""

import asyncio
import logging
import time
import uuid
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum

from pydantic import BaseModel

log = logging.getLogger(__name__)
MAX_LOG_LINES = 5_000
MAX_RETAINED_JOBS = 200


class JobStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class JobType(str, Enum):
    """Every ``create_job()`` type, in one place.

    The frontend keys its cache-invalidation registry (lib/job.ts
    ``STALE_QUERIES``) by these values; a type string that only exists at a
    call site is invisible there, so new types get registered here first.
    ``create_job`` validates against this enum — an unregistered string
    fails at the call site instead of becoming a silent no-op in the UI.
    """

    VLM_DETECT = "vlm_detect"
    PLAYER_DETECTION = "player_detection"
    PLAYER_TRACKING = "player_tracking"
    PLAYER_EMBED = "player_embed"
    RALLY_SPOT_TRAIN = "rally_spot_train"
    RALLY_SPOT_PREDICT = "rally_spot_predict"
    ACTION_TRAIN = "action_train"
    FUSION_MODEL_TRAIN = "fusion_model_train"
    SPOT_PRELABEL_BATCH = "spot_prelabel_batch"
    REID_DATASET_EXPORT = "reid_dataset_export"
    REID_TRAIN = "reid_train"
    ACTOR_ASSOCIATION_TRAIN = "actor_association_train"
    ACTOR_ASSOCIATION_PREDICT = "actor_association_predict"
    DOWNLOAD = "download"
    R2_UPLOAD = "r2_upload"
    R2_DOWNLOAD = "r2_download"


class JobSummary(BaseModel):
    """The wire shape of ``Job.to_dict()``.

    Declared once and used as the ``response_model`` of every endpoint that
    returns a job — the start endpoints, the /api/jobs list and each SSE
    event share it, so the frontend types against a single schema.
    """

    id: str
    type: JobType
    name: str
    status: JobStatus
    progress: float
    message: str
    params: dict
    error: str | None
    log_count: int
    created_at: float
    started_at: float | None


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
    logs: deque[str] = field(default_factory=lambda: deque(maxlen=MAX_LOG_LINES))
    created_at: float = field(default_factory=time.time)
    started_at: float | None = None
    _task: asyncio.Task | None = field(default=None, repr=False)
    _subscribers: list[asyncio.Queue] = field(default_factory=list, repr=False)

    def set_task(self, task: asyncio.Task) -> None:
        """Attach an asyncio task so the job can be cancelled."""
        self._task = task

    def to_dict(self) -> dict:
        """Small summary used by lists, status routes and SSE."""
        return {
            "id": self.id,
            "type": self.type,
            "name": self.name,
            "status": self.status.value,
            "progress": self.progress,
            "message": self.message,
            "params": self.params,
            "error": self.error,
            "log_count": len(self.logs),
            "created_at": self.created_at,
            "started_at": self.started_at,
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
        # Serializes SPOT inference jobs among themselves WITHOUT blocking on the
        # exclusive gpu_lock that training holds, so a (small-batch) pre-label can
        # run while a training job is in progress. Inference runs in a subprocess,
        # so the OS reclaims its VRAM on exit — no in-process cache flush needed.
        self.inference_lock = asyncio.Lock()
        self._vllm_using_gpu = False

    def create_job(
        self, job_type: "JobType | str", params: dict | None = None, name: str = ""
    ) -> Job:
        job = Job(
            id=str(uuid.uuid4())[:8],
            type=JobType(job_type).value,
            name=name,
            params=params or {},
        )
        self.jobs[job.id] = job
        self._prune_terminal_jobs()
        return job

    def _prune_terminal_jobs(self) -> None:
        overflow = len(self.jobs) - MAX_RETAINED_JOBS
        if overflow <= 0:
            return
        terminal = sorted(
            (
                job
                for job in self.jobs.values()
                if job.status
                in {JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED}
            ),
            key=lambda job: job.created_at,
        )
        for job in terminal[:overflow]:
            self.jobs.pop(job.id, None)

    def job_logs(self, job_id: str) -> list[str] | None:
        job = self.jobs.get(job_id)
        return list(job.logs) if job is not None else None

    def append_log(self, job_id: str, line: str) -> None:
        """Null-safe log append — the job may have been pruned meanwhile."""
        job = self.jobs.get(job_id)
        if job is not None:
            job.logs.append(line)

    def get_job(self, job_id: str) -> Job | None:
        return self.jobs.get(job_id)

    def list_jobs(self) -> list[dict]:
        return [j.to_dict() for j in sorted(
            self.jobs.values(), key=lambda j: j.created_at, reverse=True
        )]

    def active_count(self) -> int:
        return sum(1 for j in self.jobs.values() if j.status == JobStatus.RUNNING)

    def active_job(self, *types: "JobType") -> Job | None:
        """The pending-or-running job among these types, if any.

        PENDING counts as active: a job between ``create_job()`` and its
        first running update already occupies its slot, and the UI must see
        (and be able to cancel) whatever blocks the next start.
        """
        active = (JobStatus.PENDING, JobStatus.RUNNING)
        return next(
            (j for j in self.jobs.values() if j.type in types and j.status in active),
            None,
        )

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
            new_status = JobStatus(status) if not isinstance(status, JobStatus) else status
            if new_status is JobStatus.RUNNING and job.started_at is None:
                job.started_at = time.time()
            job.status = new_status
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
        # PENDING is cancellable too: routers count pending jobs as active,
        # so the UI must be able to cancel what it reports as running.
        job = self.jobs.get(job_id)
        if not job or job.status not in (JobStatus.PENDING, JobStatus.RUNNING):
            return False
        if job._task:
            job._task.cancel()
        job.status = JobStatus.CANCELLED
        job.message = "Cancelled"
        # Notify subscribers
        event = job.to_dict()
        for q in job._subscribers:
            try:
                q.put_nowait(event)
            except asyncio.QueueFull:
                log.debug("SSE queue full for job %s, dropping event", job.id)
        return True

    def attach_task(self, jobs: "list[Job] | Job", task: asyncio.Task) -> None:
        """Attach a cancellable task to one or more jobs.

        A job cancelled while still PENDING (before its task existed) must
        not be resurrected: cancel the late-arriving task immediately.
        """
        if isinstance(jobs, Job):
            jobs = [jobs]
        for job in jobs:
            job.set_task(task)
            if job.status is JobStatus.CANCELLED:
                task.cancel()

    @property
    def vllm_using_gpu(self) -> bool:
        return self._vllm_using_gpu

    @vllm_using_gpu.setter
    def vllm_using_gpu(self, value: bool):
        self._vllm_using_gpu = value


def threadsafe_update(
    job_id: str,
    loop: asyncio.AbstractEventLoop,
    *,
    manager: "JobManager | None" = None,
) -> Callable[..., None]:
    """Return an ``update(**fields)`` callable usable from any thread.

    Sync code running in ``run_in_executor`` can't await ``update_job``; this
    wraps it in ``loop.call_soon_threadsafe``. Accepts the same keyword fields
    as ``JobManager.update_job``. ``manager`` defaults to the module-level
    ``job_manager``; pass an explicit one if you ever construct your own
    JobManager (e.g. tests).
    """
    mgr = manager if manager is not None else job_manager

    def update(**fields) -> None:
        loop.call_soon_threadsafe(
            lambda: asyncio.ensure_future(mgr.update_job(job_id, **fields))
        )

    return update


# Module-level instance
job_manager = JobManager()
