"""Evaluation metrics for SD-MoSE models.

Consolidated from evaluation.py and benchmark.py for cleaner organization.
Includes: R², RMSE, OOD slices, physical plausibility checks.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    from sklearn.metrics import mean_squared_error, r2_score
except ImportError:
    mean_squared_error = None
    r2_score = None


def compute_r2_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[float, float]:
    """Compute R² and RMSE metrics.
    
    Args:
        y_true: True values
        y_pred: Predicted values
        
    Returns:
        Tuple of (r2_score, rmse)
    """
    if r2_score is None:
        raise ImportError("scikit-learn required for evaluation")
    r2 = float(r2_score(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    return r2, rmse


def ood_slices(
    lat: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    bands: Optional[Dict[str, Tuple[float, float]]] = None,
) -> Dict[str, Dict[str, float]]:
    """Compute R² and RMSE per latitude band (OOD slices).
    
    Args:
        lat: Latitude values
        y_true: True values
        y_pred: Predicted values
        bands: {\"tropics\": (-20, 20), \"mid_lat_n\": (20, 50), ...}
        
    Returns:
        Dictionary of metrics by latitude band
    """
    if bands is None:
        try:
            from ..config import LAT_BANDS
            bands = LAT_BANDS
        except Exception:
            bands = {
                "tropics": (-20, 20),
                "mid_lat_n": (20, 50),
                "mid_lat_s": (-50, -20),
                "high_lat_n": (50, 90),
                "high_lat_s": (-90, -50),
            }
    out = {}
    for name, (lo, hi) in bands.items():
        m = (lat >= lo) & (lat < hi)
        if m.sum() < 10:
            continue
        r2, rmse = compute_r2_rmse(y_true[m], y_pred[m])
        out[name] = {"r2": r2, "rmse": rmse, "n": int(m.sum())}
    return out


def plausibility_metrics(
    y_pred: np.ndarray,
    y_true: Optional[np.ndarray] = None,
    fco2_min: float = 200,
    fco2_max: float = 550,
) -> Dict[str, float]:
    """Physical plausibility: fraction of predictions in [fco2_min, fco2_max].
    
    Args:
        y_pred: Predicted values
        y_true: True values (optional, for additional metrics)
        fco2_min: Minimum plausible fCO₂ value (μatm)
        fco2_max: Maximum plausible fCO₂ value (μatm)
        
    Returns:
        Dictionary of plausibility metrics
    """
    out = {}
    n = len(y_pred)
    in_range = ((y_pred >= fco2_min) & (y_pred <= fco2_max)).sum()
    out["frac_in_range"] = float(in_range) / max(n, 1)
    out["frac_out_of_range"] = 1.0 - out["frac_in_range"]
    if y_true is not None:
        # Symmetric metrics on residuals
        res = np.abs(y_pred - y_true)
        out["mae"] = float(np.mean(res))
        out["median_ae"] = float(np.median(res))
    return out


def complexity_metrics(expressions: List[str]) -> Dict[str, float]:
    """Simple complexity proxy: total length, max length, mean length.
    
    Args:
        expressions: List of equation strings
        
    Returns:
        Dictionary of complexity metrics
    """
    lens = [len(e) for e in expressions]
    return {
        "total_chars": sum(lens),
        "max_chars": max(lens) if lens else 0,
        "mean_chars": np.mean(lens) if lens else 0,
    }


def validate_fco2_range(
    fco2: np.ndarray,
    min_val: float = 200.0,
    max_val: float = 600.0,
    warn_threshold: float = 0.05,
) -> Dict[str, float]:
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
