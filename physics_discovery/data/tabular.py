"""Loading arbitrary user-submitted tabular datasets (CSV) for equation discovery."""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import List, Tuple, Union

import numpy as np
import pandas as pd


def load_csv(
    path: Union[str, Path],
    target_column: str,
    drop_non_numeric: bool = True,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Load a CSV file into (X, y, feature_names) for equation discovery.

    Args:
        path: Path to a CSV file.
        target_column: Name of the column to use as the regression target.
        drop_non_numeric: If True (default), non-numeric feature columns are
            dropped with a warning. If False, a non-numeric feature column
            raises a ValueError instead. The target column itself is always
            required to be numeric.

    Returns:
        Tuple of (X, y, feature_names) where X has shape (n_rows, n_features),
        y has shape (n_rows,), and feature_names is the list of surviving
        feature column names in column order.

    Raises:
        FileNotFoundError: If `path` does not exist.
        ValueError: If the target column is missing, the target column is
            non-numeric, a non-numeric feature column is found and
            `drop_non_numeric` is False, or the resulting dataset is empty.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"CSV file not found: {path}")

    df = pd.read_csv(path)

    if df.empty:
        raise ValueError(f"CSV file at {path} contains no rows.")

    if target_column not in df.columns:
        raise ValueError(
            f"Target column {target_column!r} not found in CSV columns: {list(df.columns)}"
        )

    if not pd.api.types.is_numeric_dtype(df[target_column]):
        raise ValueError(f"Target column {target_column!r} must be numeric.")

    feature_df = df.drop(columns=[target_column])

    non_numeric_cols = [
        col for col in feature_df.columns if not pd.api.types.is_numeric_dtype(feature_df[col])
    ]
    if non_numeric_cols:
        if not drop_non_numeric:
            raise ValueError(f"Non-numeric feature columns found: {non_numeric_cols}")
        warnings.warn(
            f"Dropping non-numeric feature columns: {non_numeric_cols}",
            stacklevel=2,
        )
        feature_df = feature_df.drop(columns=non_numeric_cols)

    if feature_df.shape[1] == 0:
        raise ValueError("No numeric feature columns remain after loading CSV.")

    # Drop rows with any missing values across features/target to keep
    # downstream regressors safe from NaNs.
    combined = feature_df.copy()
    combined["__target__"] = df[target_column]
    combined = combined.dropna()
    if combined.empty:
        raise ValueError("No complete rows remain after dropping missing values.")

    feature_names = list(feature_df.columns)
    X = combined[feature_names].to_numpy(dtype=float)
    y = combined["__target__"].to_numpy(dtype=float)

    return X, y, feature_names
