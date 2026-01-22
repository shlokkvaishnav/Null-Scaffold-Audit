"""
Evaluation metrics for SD-MoSE: R², RMSE, OOD slices, physical plausibility.
"""
from __future__ import annotations

import numpy as np
from typing import Dict, List, Optional, Tuple

try:
    from sklearn.metrics import mean_squared_error, r2_score
except ImportError:
    mean_squared_error = None
    r2_score = None


def compute_r2_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[float, float]:
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
    """
    Compute R² and RMSE per latitude band (OOD slices).
    bands: {"tropics": (-20, 20), "mid_lat_n": (20, 50), ...}
    """
    if bands is None:
        try:
            from .config import LAT_BANDS
            bands = LAT_BANDS
        except Exception:
            bands = {"tropics": (-20, 20), "mid_lat_n": (20, 50), "mid_lat_s": (-50, -20), "high_lat_n": (50, 90), "high_lat_s": (-90, -50)}
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
    """
    Physical plausibility: fraction of predictions in [fco2_min, fco2_max],
    and optionally monotonicity / gradient checks if y_true + full feature matrix available.
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
    """Simple complexity proxy: total length, max length, mean length."""
    lens = [len(e) for e in expressions]
    return {
        "total_chars": sum(lens),
        "max_chars": max(lens) if lens else 0,
        "mean_chars": np.mean(lens) if lens else 0,
    }
