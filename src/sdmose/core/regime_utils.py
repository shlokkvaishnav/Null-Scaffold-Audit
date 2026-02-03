"""Core utilities: regime analysis functions.

Extracted from utils.py for better organization.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Union
from pathlib import Path

import numpy as np

try:
    import xarray as xr
except ImportError:
    xr = None


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
