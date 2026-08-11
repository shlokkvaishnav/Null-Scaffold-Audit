"""Pydantic request/response models for the equation-discovery API."""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field


class DatasetInfo(BaseModel):
    """Metadata returned after a dataset has been uploaded and parsed."""

    id: str
    feature_names: list[str]
    n_rows: int


class JobRequest(BaseModel):
    """Request body for submitting a discovery job against an uploaded dataset."""

    dataset_id: str
    backend: str = Field(
        default="gplearn",
        description=(
            "Symbolic-regression backend to use. Defaults to 'gplearn' "
            "(pure Python, no Julia runtime), matching the SYMBOLIC_BACKEND "
            "default used by the Docker image. 'pysr' is available as an "
            "opt-in backend but requires a Julia-enabled image."
        ),
    )
    max_iters: int | None = Field(
        default=None,
        description="Optional cap on generations/iterations for the symbolic search.",
    )


class JobStatus(BaseModel):
    """Status snapshot returned immediately after submitting a job."""

    id: str
    status: Literal["pending", "running", "done", "failed"]
    created_at: datetime


class JobResult(BaseModel):
    """Final (or in-progress) result payload returned when polling a job."""

    job_id: str
    status: Literal["pending", "running", "done", "failed"]
    equation: str | None = None
    rmse: float | None = None
    confidence: float | None = None
    error: str | None = None
