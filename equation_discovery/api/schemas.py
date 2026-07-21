"""Pydantic request/response models for the equation-discovery API."""

from __future__ import annotations

from datetime import datetime
from typing import List, Literal, Optional

from pydantic import BaseModel, Field


class DatasetInfo(BaseModel):
    """Metadata returned after a dataset has been uploaded and parsed."""

    id: str
    feature_names: List[str]
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
    max_iters: Optional[int] = Field(
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
    equation: Optional[str] = None
    rmse: Optional[float] = None
    confidence: Optional[float] = None
    error: Optional[str] = None
