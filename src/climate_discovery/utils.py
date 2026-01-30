"""Utility functions for SD-MoSE pipeline.

Includes:
- Data loading and scaling
- Feature engineering
- Tensor preparation
- Validation checks
- Regime analysis utilities
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch

try:
    import xarray as xr
except ImportError:
    xr = None


# =============================================================================
# DATA LOADING & SCALING
# =============================================================================

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


# =============================================================================
# FEATURE ENGINEERING
# =============================================================================

def add_cyclic_time_features(
    df: pd.DataFrame, 
    time_col: str = "time"
) -> pd.DataFrame:
    """Add sin/cos encodings of month for seasonal cycles.
    
    Args:
        df: DataFrame with datetime column
        time_col: Name of time column
        
    Returns:
        DataFrame with added columns: sin_month, cos_month
        
    Scientific rationale:
        sin/cos encoding ensures January and December are neighbors,
        and allows neural networks to learn seasonal periodicity.
    """
    df = df.copy()
    
    if time_col not in df.columns:
        raise KeyError(f"Time column '{time_col}' not found")
    
    # Convert to datetime if needed
    if not pd.api.types.is_datetime64_any_dtype(df[time_col]):
        df[time_col] = pd.to_datetime(df[time_col])
    
    # Month as angle: Jan=0, Dec=11 → [0, 2π)
    month_numeric = df[time_col].dt.month - 1  # 0-11
    angle = 2 * np.pi * month_numeric / 12.0
    
    df["sin_month"] = np.sin(angle)
    df["cos_month"] = np.cos(angle)
    
    return df


def add_normalized_year(
    df: pd.DataFrame,
    time_col: str = "time",
    reference_year: int = 2015,
) -> pd.DataFrame:
    """Add normalized year feature for long-term trends.
    
    Args:
        df: DataFrame with datetime column
        time_col: Name of time column
        reference_year: Center year for normalization
        
    Returns:
        DataFrame with added 'year_norm' column
        
    Note:
        year_norm = (year - reference_year) / 10.0
        This keeps values in [-1, 1] range for ~2005-2025
    """
    df = df.copy()
    
    if time_col not in df.columns:
        raise KeyError(f"Time column '{time_col}' not found")
    
    if not pd.api.types.is_datetime64_any_dtype(df[time_col]):
        df[time_col] = pd.to_datetime(df[time_col])
    
    year = df[time_col].dt.year
    df["year_norm"] = (year - reference_year) / 10.0
    
    return df


def add_log_chlorophyll(
    df: pd.DataFrame,
    chl_col: str = "chl",
    epsilon: float = 1e-3,
) -> pd.DataFrame:
    """Add log-transformed chlorophyll (handles skewness).
    
    Args:
        df: DataFrame with chlorophyll column (mg/m³)
        chl_col: Name of chlorophyll column
        epsilon: Floor value to prevent log(0)
        
    Returns:
        DataFrame with added 'log_chl' column
        
    Scientific rationale:
        Chlorophyll-a is highly skewed (oligotrophic vs eutrophic).
        log(Chl) ~ Normal distribution, better for regression.
    """
    df = df.copy()
    
    if chl_col not in df.columns:
        raise KeyError(f"Chlorophyll column '{chl_col}' not found")
    
    # Clip to positive values
    chl_safe = np.clip(df[chl_col].values, epsilon, None)
    df["log_chl"] = np.log(chl_safe)
    
    return df


def normalize_coordinates(
    df: pd.DataFrame,
    lat_col: str = "lat",
    lon_col: str = "lon",
) -> pd.DataFrame:
    """Normalize lat/lon to [-1, 1] for neural network stability.
    
    Args:
        df: DataFrame with lat/lon columns
        lat_col: Latitude column name
        lon_col: Longitude column name
        
    Returns:
        DataFrame with 'lat_norm', 'lon_norm' columns
        
    Transformation:
        lat: [-90, 90] → [-1, 1]
        lon: [-180, 180] → [-1, 1]
    """
    df = df.copy()
    
    if lat_col not in df.columns or lon_col not in df.columns:
        raise KeyError(f"Missing coordinate columns: {lat_col}, {lon_col}")
    
    df["lat_norm"] = df[lat_col] / 90.0
    df["lon_norm"] = df[lon_col] / 180.0
    
    return df


def compute_sst_gradient(
    df: pd.DataFrame,
    sst_col: str = "sst",
    lat_col: str = "lat",
    lon_col: str = "lon",
    time_col: Optional[str] = "time",
) -> pd.DataFrame:
    """Compute spatial gradient magnitude of SST.
    
    Args:
        df: DataFrame with SST and coordinate columns
        sst_col: Name of SST column (°C)
        lat_col: Latitude column name
        lon_col: Longitude column name
        time_col: Optional time column for temporal grouping
        
    Returns:
        DataFrame with added 'sst_gradient' column (°C per degree)
        
    Scientific rationale:
        |∇SST| indicates ocean fronts and mesoscale features.
        High gradients mark regime boundaries (e.g., Gulf Stream, Antarctic Convergence).
        
    Method:
        For gridded data: Use finite differences on regular grid
        For scattered data: Use local polynomial fit or nearest-neighbor approximation
        
    Example:
        >>> df = compute_sst_gradient(df)
        >>> # High gradient regions are fronts
        >>> fronts = df[df['sst_gradient'] > df['sst_gradient'].quantile(0.9)]
    """
    df = df.copy()
    
    if sst_col not in df.columns:
        raise KeyError(f"SST column '{sst_col}' not found")
    if lat_col not in df.columns or lon_col not in df.columns:
        raise KeyError(f"Missing coordinate columns: {lat_col}, {lon_col}")
    
    # Check if data is on regular grid
    lat_unique = df[lat_col].nunique()
    lon_unique = df[lon_col].nunique()
    expected_grid_size = lat_unique * lon_unique
    
    # If temporal data, compute gradients per timestep
    if time_col and time_col in df.columns:
        gradients = []
        for time_val, group in df.groupby(time_col):
            grad = _compute_gradient_single_time(
                group, sst_col, lat_col, lon_col
            )
            gradients.append(grad)
        df['sst_gradient'] = pd.concat(gradients)
    else:
        df['sst_gradient'] = _compute_gradient_single_time(
            df, sst_col, lat_col, lon_col
        )
    
    return df


def _compute_gradient_single_time(
    df: pd.DataFrame,
    sst_col: str,
    lat_col: str,
    lon_col: str,
) -> pd.Series:
    """Helper: Compute SST gradient for single timestep.
    
    Uses finite differences on approximately gridded data.
    Falls back to neighbor-based estimation for irregular grids.
    """
    # Sort by lat, lon for consistent gradient computation
    df_sorted = df.sort_values([lat_col, lon_col]).copy()
    
    # Get unique lat/lon values
    lats = np.sort(df_sorted[lat_col].unique())
    lons = np.sort(df_sorted[lon_col].unique())
    
    # Check if reasonably gridded
    n_points = len(df_sorted)
    expected_grid = len(lats) * len(lons)
    
    if n_points >= 0.7 * expected_grid:  # At least 70% coverage
        # Use grid-based finite differences
        gradient = _gradient_on_grid(df_sorted, sst_col, lat_col, lon_col, lats, lons)
    else:
        # Use neighbor-based estimation for scattered points
        gradient = _gradient_scattered(df_sorted, sst_col, lat_col, lon_col)
    
    # Return in original index order
    return gradient.reindex(df.index)


def _gradient_on_grid(
    df: pd.DataFrame,
    sst_col: str,
    lat_col: str,
    lon_col: str,
    lats: np.ndarray,
    lons: np.ndarray,
) -> pd.Series:
    """Compute gradient using finite differences on grid."""
    from scipy.ndimage import sobel
    
    # Create grid
    nlat, nlon = len(lats), len(lons)
    sst_grid = np.full((nlat, nlon), np.nan)
    
    # Fill grid
    for _, row in df.iterrows():
        i = np.searchsorted(lats, row[lat_col])
        j = np.searchsorted(lons, row[lon_col])
        if i < nlat and j < nlon:
            sst_grid[i, j] = row[sst_col]
    
    # Compute gradients using Sobel operator (robust to missing data)
    # Fill NaNs with local mean for gradient computation
    mask = ~np.isnan(sst_grid)
    if np.sum(mask) > 0:
        from scipy.ndimage import generic_filter
        sst_filled = sst_grid.copy()
        
        def local_mean(values):
            valid = values[~np.isnan(values)]
            return np.mean(valid) if len(valid) > 0 else 0
        
        # Fill NaNs with local neighborhood mean
        sst_filled = np.where(
            np.isnan(sst_grid),
            generic_filter(sst_grid, local_mean, size=3, mode='constant', cval=np.nan),
            sst_grid
        )
        sst_filled = np.nan_to_num(sst_filled, nan=np.nanmean(sst_grid))
    else:
        sst_filled = np.zeros_like(sst_grid)
    
    # Compute gradients
    dlat = np.abs(lats[1] - lats[0]) if len(lats) > 1 else 1.0
    dlon = np.abs(lons[1] - lons[0]) if len(lons) > 1 else 1.0
    
    grad_lat = sobel(sst_filled, axis=0) / dlat  # d(SST)/d(lat)
    grad_lon = sobel(sst_filled, axis=1) / dlon  # d(SST)/d(lon)
    
    # Magnitude: |∇SST| = sqrt(∂SST/∂lat² + ∂SST/∂lon²)
    grad_magnitude = np.sqrt(grad_lat**2 + grad_lon**2)
    
    # Map back to DataFrame
    gradients = []
    for _, row in df.iterrows():
        i = np.searchsorted(lats, row[lat_col])
        j = np.searchsorted(lons, row[lon_col])
        if i < nlat and j < nlon:
            gradients.append(grad_magnitude[i, j])
        else:
            gradients.append(0.0)
    
    return pd.Series(gradients, index=df.index, name='sst_gradient')


def _gradient_scattered(
    df: pd.DataFrame,
    sst_col: str,
    lat_col: str,
    lon_col: str,
    n_neighbors: int = 8,
) -> pd.Series:
    """Compute gradient using nearest neighbors for scattered points."""
    from sklearn.neighbors import NearestNeighbors
    
    coords = df[[lat_col, lon_col]].values
    sst_vals = df[sst_col].values
    
    # Fit nearest neighbors
    nbrs = NearestNeighbors(n_neighbors=min(n_neighbors, len(df))).fit(coords)
    distances, indices = nbrs.kneighbors(coords)
    
    # Estimate gradient as max SST difference / min distance to neighbors
    gradients = []
    for i in range(len(df)):
        neighbor_sst = sst_vals[indices[i, 1:]]  # Exclude self (index 0)
        neighbor_dist = distances[i, 1:]
        
        if len(neighbor_dist) > 0 and neighbor_dist[0] > 0:
            sst_diff = np.abs(neighbor_sst - sst_vals[i])
            # Gradient ~ max difference / min distance
            gradient = np.max(sst_diff) / np.min(neighbor_dist[neighbor_dist > 0])
        else:
            gradient = 0.0
        
        gradients.append(gradient)
    
    return pd.Series(gradients, index=df.index, name='sst_gradient')


# =============================================================================
# TENSOR PREPARATION
# =============================================================================

def prepare_tensors(
    df: pd.DataFrame,
    expert_cols: List[str],
    gating_cols: List[str],
    target_col: Optional[str] = None,
    drop_nan: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[pd.Index]]:
    """Prepare PyTorch tensors for SD-MoSE training.
    
    Args:
        df: Input DataFrame with all features
        expert_cols: Features for symbolic experts
        gating_cols: Features for gating network
        target_col: Target variable (fCO₂)
        drop_nan: Whether to remove rows with NaN values
        
    Returns:
        X_expert: Expert input tensor (N, D_expert)
        X_gate: Gating input tensor (N, D_gate)
        y: Target tensor (N,) or None
        valid_idx: Index of valid rows (for spatial reconstruction)
        
    Example:
        >>> X_expert, X_gate, y, idx = prepare_tensors(
        ...     df, 
        ...     expert_cols=['sst', 'sss', 'log_chl'],
        ...     gating_cols=['lat_norm', 'lon_norm', 'sst'],
        ...     target_col='fco2'
        ... )
    """
    # Validate columns
    all_cols = set(expert_cols + gating_cols)
    if target_col:
        all_cols.add(target_col)
    
    missing = all_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in DataFrame: {missing}")
    
    # Handle NaN values
    if drop_nan:
        subset = list(all_cols)
        df_clean = df.dropna(subset=subset)
        valid_idx = df_clean.index
    else:
        df_clean = df
        valid_idx = df.index
    
    if len(df_clean) == 0:
        raise ValueError("No valid data after NaN removal")
    
    # Convert to tensors
    X_expert = torch.tensor(
        df_clean[expert_cols].values, 
        dtype=torch.float32
    )
    X_gate = torch.tensor(
        df_clean[gating_cols].values, 
        dtype=torch.float32
    )
    
    y = None
    if target_col:
        y = torch.tensor(
            df_clean[target_col].values, 
            dtype=torch.float32
        )
    
    return X_expert, X_gate, y, valid_idx


# =============================================================================
# VALIDATION CHECKS
# =============================================================================

def validate_fco2_range(
    fco2: np.ndarray,
    min_val: float = 200.0,
    max_val: float = 600.0,
    warn_threshold: float = 0.05,
) -> Dict[str, Union[int, float]]:
    """Check fCO₂ values for physical plausibility.
    
    Args:
        fco2: Array of fCO₂ values (μatm)
        min_val: Minimum plausible value
        max_val: Maximum plausible value
        warn_threshold: Warn if >5% of values outside range
        
    Returns:
        Dictionary with validation statistics
    """
    n_total = len(fco2)
    n_below = np.sum(fco2 < min_val)
    n_above = np.sum(fco2 > max_val)
    n_invalid = n_below + n_above
    frac_invalid = n_invalid / n_total
    
    result = {
        "n_total": n_total,
        "n_below_min": n_below,
        "n_above_max": n_above,
        "n_invalid": n_invalid,
        "frac_invalid": frac_invalid,
        "mean": float(np.nanmean(fco2)),
        "std": float(np.nanstd(fco2)),
    }
    
    if frac_invalid > warn_threshold:
        import warnings
        warnings.warn(
            f"{frac_invalid*100:.1f}% of fCO₂ values outside [{min_val}, {max_val}] μatm. "
            f"Check data quality."
        )
    
    return result


def check_feature_scales(
    df: pd.DataFrame,
    features: List[str],
) -> pd.DataFrame:
    """Report feature value ranges (useful for debugging normalization).
    
    Args:
        df: DataFrame with features
        features: List of feature names
        
    Returns:
        DataFrame with columns: feature, min, max, mean, std
    """
    stats = []
    for feat in features:
        if feat in df.columns:
            vals = df[feat].values
            stats.append({
                "feature": feat,
                "min": float(np.nanmin(vals)),
                "max": float(np.nanmax(vals)),
                "mean": float(np.nanmean(vals)),
                "std": float(np.nanstd(vals)),
                "n_nan": int(np.sum(np.isnan(vals))),
            })
    
    return pd.DataFrame(stats)


# =============================================================================
# REGIME ANALYSIS UTILITIES
# =============================================================================

def compute_regime_entropy(
    probs: np.ndarray,
    epsilon: float = 1e-10,
) -> np.ndarray:
    """Compute Shannon entropy of regime probabilities.
    
    Args:
        probs: Regime probabilities (N, K)
        epsilon: Small constant for numerical stability
        
    Returns:
        Entropy per sample (N,)
        
    Interpretation:
        Low entropy → confident single-regime assignment
        High entropy → transition zone between regimes
        
    Formula:
        H = -Σ p_k log(p_k)
    """
    probs = np.clip(probs, epsilon, 1.0)
    entropy = -np.sum(probs * np.log(probs), axis=1)
    return entropy


def compute_max_probability(probs: np.ndarray) -> np.ndarray:
    """Get maximum regime probability (confidence measure).
    
    Args:
        probs: Regime probabilities (N, K)
        
    Returns:
        Max probability per sample (N,)
    """
    return np.max(probs, axis=1)


def get_dominant_regime(probs: np.ndarray) -> np.ndarray:
    """Assign each point to its most likely regime.
    
    Args:
        probs: Regime probabilities (N, K)
        
    Returns:
        Regime indices (N,) in range [0, K-1]
    """
    return np.argmax(probs, axis=1)


def compute_regime_transition_rate(
    regime_assignments: np.ndarray,
    spatial_neighbors: Optional[np.ndarray] = None,
) -> float:
    """Compute fraction of spatial/temporal transitions between regimes.
    
    Args:
        regime_assignments: Regime indices (N,)
        spatial_neighbors: Optional neighbor indices for spatial transitions
        
    Returns:
        Transition rate in [0, 1]
        
    For temporal: Compare consecutive timesteps
    For spatial: Compare adjacent grid cells (if neighbor info provided)
    """
    if spatial_neighbors is not None:
        # Spatial transitions
        transitions = regime_assignments[spatial_neighbors] != regime_assignments[:, None]
        return float(np.mean(transitions))
    else:
        # Temporal transitions (consecutive elements)
        transitions = regime_assignments[1:] != regime_assignments[:-1]
        return float(np.mean(transitions))


# =============================================================================
# ENSEMBLE UTILITIES
# =============================================================================

def compute_ensemble_agreement(
    regime_arrays: List[np.ndarray],
) -> np.ndarray:
    """Compute fraction of ensemble members agreeing on dominant regime.
    
    Args:
        regime_arrays: List of regime assignments (each is (N,))
        
    Returns:
        Agreement fraction per point (N,)
        
    Interpretation:
        1.0 = all ensemble members agree
        0.0 = all members assign different regimes
        
    This is a key robustness check for regime discovery.
    """
    if not regime_arrays:
        raise ValueError("Empty ensemble list")
    
    # Stack into (N_ensemble, N_points)
    stack = np.stack(regime_arrays, axis=0)
    n_ensemble = stack.shape[0]
    
    # For each point, count most common regime
    from scipy.stats import mode
    modal_regime, counts = mode(stack, axis=0, keepdims=False)
    
    # Agreement = fraction voting for modal regime
    agreement = counts / n_ensemble
    return agreement


# =============================================================================
# FILE I/O HELPERS
# =============================================================================

def save_regime_assignments(
    regime_ids: np.ndarray,
    probs: np.ndarray,
    coords: Dict[str, np.ndarray],
    path: Union[str, Path],
) -> None:
    """Save regime assignments as NetCDF for visualization.
    
    Args:
        regime_ids: Dominant regime per point (N,)
        probs: Full probability distribution (N, K)
        coords: Dict with 'lat', 'lon', 'time' arrays
        path: Output NetCDF path
    """
    if xr is None:
        raise ImportError("xarray required")
    
    n_points, n_regimes = probs.shape
    
    ds = xr.Dataset(
        {
            "regime_id": (["point"], regime_ids),
            "regime_prob": (["point", "regime"], probs),
            "entropy": (["point"], compute_regime_entropy(probs)),
            "confidence": (["point"], compute_max_probability(probs)),
        },
        coords={
            "lat": (["point"], coords["lat"]),
            "lon": (["point"], coords["lon"]),
            "time": (["point"], coords.get("time", np.arange(n_points))),
            "regime": np.arange(n_regimes),
        },
    )
    
    ds.to_netcdf(path)
    ds.close()


def load_regime_assignments(path: Union[str, Path]) -> xr.Dataset:
    """Load saved regime assignments."""
    if xr is None:
        raise ImportError("xarray required")
    return xr.open_dataset(path, engine="netcdf4")