"""Core utilities: data loading and scaling functions.

Extracted from utils.py for better organization.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Union

import numpy as np

try:
    import xarray as xr
except ImportError:
    xr = None


def load_scalers(path: Union[str, Path]) -> Dict[str, float]:
    """Load mean/std from preprocessing scalers NetCDF.
    
    Args:
        path: Path to scalers.nc file
        
    Returns:
        Dictionary with keys like 'sst_mean', 'sst_std', etc.
        
    Example:
        >>> scalers = load_scalers("data/processed/scalers.nc")
        >>> print(scalers['sst_mean'])  # Global mean SST
    """
    if xr is None:
        raise ImportError("xarray required: pip install xarray")
    
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Scalers file not found: {path}")
    
    ds = xr.open_dataset(path, engine="netcdf4")
    out = {}
    
    for var in ds.data_vars:
        val = ds[var].values
        # Handle both scalar and array formats
        out[str(var)] = float(np.nanmean(val)) if val.size > 1 else float(val)
    
    ds.close()
    return out


def inverse_transform(
    X: np.ndarray,
    columns: List[str],
    scalers: Dict[str, float],
    epsilon: float = 1e-8,
) -> np.ndarray:
    """Un-standardize features using stored mean/std.
    
    Args:
        X: Normalized data (N, D)
        columns: Feature names in column order
        scalers: Dict from load_scalers()
        epsilon: Small constant to prevent division by zero
        
    Returns:
        Original-scale data (N, D)
        
    Scientific note:
        Inverse transform is needed for:
        1. Symbolic equation interpretation (raw SST, not z-scores)
        2. Physical plausibility checks
        3. Visualization in natural units
    """
    X = np.asarray(X, dtype=np.float64)
    out = X.copy()
    
    for i, col in enumerate(columns):
        mean_key = f"{col}_mean"
        std_key = f"{col}_std"
        
        if mean_key in scalers and std_key in scalers:
            mu = scalers[mean_key]
            sigma = scalers[std_key]
            out[:, i] = X[:, i] * (sigma + epsilon) + mu
    
    return out
