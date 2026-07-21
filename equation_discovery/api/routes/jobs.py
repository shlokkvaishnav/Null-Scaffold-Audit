"""Discovery job submission/polling endpoints."""

from __future__ import annotations

from fastapi import APIRouter, BackgroundTasks, HTTPException

from equation_discovery.api.schemas import JobRequest, JobResult, JobStatus
from equation_discovery.api.worker import DATASETS, JOBS, create_job, run_discovery_job

router = APIRouter(prefix="/jobs", tags=["jobs"])


@router.post("", response_model=JobStatus)
async def submit_job(request: JobRequest, background_tasks: BackgroundTasks) -> JobStatus:
    """Submit a discovery job against a previously uploaded dataset.

    Schedules `run_discovery_job` to run in the background and immediately
    returns a "pending" job status; poll `GET /jobs/{id}` for the result.
    """
    if request.dataset_id not in DATASETS:
        raise HTTPException(status_code=404, detail=f"Dataset {request.dataset_id!r} not found.")

    job_id = create_job()
    background_tasks.add_task(
        run_discovery_job,
        job_id=job_id,
        dataset_id=request.dataset_id,
        backend=request.backend,
        max_iters=request.max_iters,
    )

    job = JOBS[job_id]
    return JobStatus(id=job_id, status=job["status"], created_at=job["created_at"])


@router.get("/{job_id}", response_model=JobResult)
async def get_job(job_id: str) -> JobResult:
    """Poll the status/result of a previously submitted discovery job."""
    job = JOBS.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Job {job_id!r} not found.")

    return JobResult(
        job_id=job_id,
        status=job["status"],
        equation=job["equation"],
        rmse=job["rmse"],
        confidence=job["confidence"],
        error=job["error"],
    )
