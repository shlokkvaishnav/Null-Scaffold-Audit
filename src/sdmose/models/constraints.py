"""
Physics constraints for constraint-guided symbolic discovery.
- Temperature sensitivity: d(fCO2)/d(SST) sign (e.g. solubility decreases with T -> often positive)
- Output bounds: plausible fCO2 range
- Monotonicity in SST for key regimes
"""

from __future__ import annotations

from typing import Callable, Optional, Tuple

import numpy as np


def check_temperature_sensitivity(
    X: np.ndarray,
    predict_fn: Callable[[np.ndarray], np.ndarray],
    sst_idx: int = 0,
    eps: float = 1e-4,
    expect_positive: bool = True,
) -> float:
    """
    Fraction of samples where d(y)/d(SST) has wrong sign (violation ratio).
    """
    y0 = predict_fn(X)
    X_plus = X.astype(np.float64, copy=True)
    X_plus[:, sst_idx] += eps
    y_plus = predict_fn(X_plus)
    dydsst = (y_plus - y0) / (eps + 1e-12)
    if expect_positive:
        wrong = np.sum(dydsst < 0)
    else:
        wrong = np.sum(dydsst > 0)
    n = np.sum(np.isfinite(dydsst))
    return float(wrong) / max(n, 1)


def check_output_bounds(
    y_pred: np.ndarray,
    y_min: float = 200,
    y_max: float = 550,
) -> float:
    """Fraction of predictions outside [y_min, y_max]."""
    y_pred = np.asarray(y_pred).ravel()
    valid = np.isfinite(y_pred)
    n = np.sum(valid)
    if n == 0:
        return 1.0
    out = np.sum((y_pred < y_min) | (y_pred > y_max))
    return float(out) / n


def check_monotonicity_sst(
    X: np.ndarray,
    predict_fn: Callable[[np.ndarray], np.ndarray],
    sst_idx: int = 0,
    eps: float = 1e-4,
) -> float:
    """Fraction of samples where d(y)/d(SST) < 0 (violation if we expect monotonic increase)."""
    return check_temperature_sensitivity(
        X, predict_fn, sst_idx, eps, expect_positive=True
    )


def constraint_score(
    X: np.ndarray,
    y_pred: np.ndarray,
    predict_fn: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    sst_idx: int = 0,
    y_min: float = 200,
    y_max: float = 550,
    w_bounds: float = 1.0,
    w_sst: float = 1.0,
) -> Tuple[float, dict]:
    """
    Combined constraint penalty. Lower is better.
    Returns (score, {"bounds_viol": ..., "sst_viol": ...}).
    """
    bounds_viol = check_output_bounds(y_pred, y_min, y_max)
    sst_viol = 0.0
    if predict_fn is not None:
        sst_viol = check_temperature_sensitivity(
            X, predict_fn, sst_idx, expect_positive=True
        )
    score = w_bounds * bounds_viol + w_sst * sst_viol
    return float(score), {"bounds_viol": bounds_viol, "sst_viol": sst_viol}
