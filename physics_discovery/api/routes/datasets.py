"""Dataset upload/lookup endpoints."""

from __future__ import annotations

import tempfile
from pathlib import Path

from fastapi import APIRouter, Form, HTTPException, UploadFile

from physics_discovery.api.schemas import DatasetInfo
from physics_discovery.api.worker import DATASETS, register_dataset
from physics_discovery.data.tabular import load_csv

router = APIRouter(prefix="/datasets", tags=["datasets"])


@router.post("", response_model=DatasetInfo)
async def upload_dataset(
    file: UploadFile,
    target_column: str | None = Form(
        default=None,
        description="Name of the target column. Defaults to the CSV's last column if omitted.",
    ),
) -> DatasetInfo:
    """Upload a CSV file and register it as a dataset for discovery jobs.

    If `target_column` is not provided, the last column of the CSV is used
    as the regression target by default.
    """
    filename = file.filename or ""
    if not filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only .csv file uploads are supported.")

    contents = await file.read()
    if not contents:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tmp:
        tmp.write(contents)
        tmp_path = Path(tmp.name)

    try:
        if target_column is None:
            import pandas as pd

            header = pd.read_csv(tmp_path, nrows=0)
            if header.shape[1] == 0:
                raise HTTPException(status_code=400, detail="CSV has no columns.")
            target_column = header.columns[-1]

        try:
            x, y, feature_names = load_csv(tmp_path, target_column=target_column)
        except (FileNotFoundError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
    finally:
        tmp_path.unlink(missing_ok=True)

    dataset_id = register_dataset(x, y, feature_names)
    return DatasetInfo(id=dataset_id, feature_names=feature_names, n_rows=int(x.shape[0]))


@router.get("/{dataset_id}", response_model=DatasetInfo)
async def get_dataset(dataset_id: str) -> DatasetInfo:
    """Look up metadata for a previously uploaded dataset."""
    entry = DATASETS.get(dataset_id)
    if entry is None:
        raise HTTPException(status_code=404, detail=f"Dataset {dataset_id!r} not found.")

    x, _y, feature_names = entry
    return DatasetInfo(id=dataset_id, feature_names=feature_names, n_rows=int(x.shape[0]))
