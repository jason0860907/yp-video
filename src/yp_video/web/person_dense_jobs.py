"""Dense 2XLarge pass jobs (person/dense.py) as the web app runs them: the
per-video work, and which videos a job has yet to finish. The route that
starts them is routers/person.py; App Review reads ``queued`` for its rows.
"""

from __future__ import annotations

from pathlib import Path

from yp_video.person.dense import detect_dense, release_detector
from yp_video.web.jobs import JobType, job_manager
from yp_video.web.r2_client import materialized_cut


def queued() -> set[str]:
    """Videos a pending or running dense job has yet to finish."""
    job = job_manager.active_job(JobType.PERSON_DENSE)
    if job is None:
        return set()
    return {item["video"] for item in job.params["items"] if item["status"] in ("pending", "running")}


def run(path: Path, on_progress) -> dict:
    """One video's pass. The 2XLarge model holds ~8 GB; the next job on this
    GPU needs it back, so it is released after every video, failed or not."""
    try:
        with materialized_cut(path):
            return detect_dense(path, should_pause=lambda: False, on_progress=on_progress)
    finally:
        release_detector()
