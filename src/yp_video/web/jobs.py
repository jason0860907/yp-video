"""Background job manager with GPU lock."""

import asyncio
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum


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
    created_at: float = field(default_factory=time.time)
    _task: asyncio.Task | None = field(default=None, repr=False)
    _subscribers: list[asyncio.Queue] = field(default_factory=list, repr=False)

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
            "created_at": self.created_at,
        }


class JobManager:
    """In-memory job manager with GPU lock for mutual exclusion."""

    def __init__(self):
        self.jobs: dict[str, Job] = {}
        self.gpu_lock = asyncio.Lock()
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
                pass

    def subscribe(self, job_id: str) -> asyncio.Queue | None:
        job = self.jobs.get(job_id)
        if not job:
            return None
        q: asyncio.Queue = asyncio.Queue(maxsize=100)
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
                pass
        return True

    @property
    def vllm_using_gpu(self) -> bool:
        return self._vllm_using_gpu

    @vllm_using_gpu.setter
    def vllm_using_gpu(self, value: bool):
        self._vllm_using_gpu = value


# Singleton
job_manager = JobManager()
