from __future__ import annotations

import torch
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple

try:
    import xarray as xr
except ImportError:
    xr = None


def load_scalers(path: str) -> Dict[str, np.ndarray]:
    """Load mean/std from preprocess scalers NetCDF. Keys: sst_mean, sst_std, ..."""
    if xr is None:
        raise ImportError("xarray required")
    ds = xr.open_dataset(path)
    out = {}
    for v in ds.data_vars:
        out[v] = np.asarray(ds[v].values, dtype=np.float64)
    return out


def inverse_transform_physics(
    X: np.ndarray,
    cols: List[str],
    scalers: Dict[str, np.ndarray],
) -> np.ndarray:
    """Un-normalize physics columns (sst, sss, log_chl). X is (N, D), cols ordered as in X."""
    X = np.asarray(X, dtype=np.float64)
    out = X.copy()
    for i, c in enumerate(cols):
        mu = scalers.get(f"{c}_mean")
        sig = scalers.get(f"{c}_std")
        if mu is not None and sig is not None:
            m = np.float64(mu) if np.ndim(mu) == 0 else np.nanmean(mu)
            s = np.float64(sig) if np.ndim(sig) == 0 else np.nanmean(sig)
            out[:, i] = X[:, i] * (s + 1e-6) + m
    return out

def prepare_gating_data(
    df: pd.DataFrame, 
    lat_col: str = 'lat', 
    lon_col: str = 'lon',
    physics_cols: Optional[List[str]] = None,
    gating_cols: Optional[List[str]] = None,
    target_col: Optional[str] = None
) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    """
    Prepares data for the Gating Network (MoE).
    Values are normalized to approximately [-1, 1] range for neural net stability.
    
    Args:
        df: Input DataFrame containing physics and coordinate columns.
        lat_col: Name of latitude column.
        lon_col: Name of longitude column.
        physics_cols: List of column names for the 'Expert' input (SST, SSS, etc.).
                      Defaults to ['sst', 'sss', 'log_chl'].
        gating_cols: List of column names for the 'Gating' input (Lat, Lon, Time).
                     Defaults to ['lat_norm', 'lon_norm', 'sin_month', 'cos_month', 'year_feature'].
        target_col: Name of target variable (e.g., 'fco2'). If None, y tensor is None.
        
    Returns:
        X_expert (Tensor): Features for experts.
        X_gate (Tensor): Features for gating network.
        y (Tensor or None): Target variable.
    """
    
    # 1. Feature Engineering
    # Normalize Lat/Lon to [-1, 1] range (Approx) for NN stability
    # Lat: -90 to 90 -> -1 to 1
    # Lon: -180 to 180 -> -1 to 1
    
    # Avoid modifying original DF in place if possible, but for performance with large DFs 
    # we just add columns.
    if 'lat_norm' not in df.columns:
        df['lat_norm'] = df[lat_col] / 90.0
    if 'lon_norm' not in df.columns:
        df['lon_norm'] = df[lon_col] / 180.0
        
    # Default Columns
    if physics_cols is None:
        physics_cols = ['sst', 'sss', 'log_chl']
        
    if gating_cols is None:
        gating_cols = ['lat_norm', 'lon_norm', 'sin_month', 'cos_month', 'year_feature']
        
    # Check if columns exist
    missing_phys = [c for c in physics_cols if c not in df.columns]
    missing_gate = [c for c in gating_cols if c not in df.columns]
    
    if missing_phys:
        raise ValueError(f"Missing physics columns in DataFrame: {missing_phys}")
    if missing_gate:
        raise ValueError(f"Missing gating columns in DataFrame: {missing_gate}")

    # Convert to Tensors
    X_expert = torch.tensor(df[physics_cols].values, dtype=torch.float32)
    X_gate = torch.tensor(df[gating_cols].values, dtype=torch.float32)
    
    y = None
    if target_col:
        if target_col in df.columns:
            y = torch.tensor(df[target_col].values, dtype=torch.float32)
        else:
            raise ValueError(f"Target column '{target_col}' not found in DataFrame.")
            
    return X_expert, X_gate, y
