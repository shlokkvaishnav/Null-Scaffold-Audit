"""Reusable fit-quality and statistical-comparison metrics for benchmark runs."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import numpy as np
from scipy import stats
from sklearn.metrics import mean_absolute_error, mean_squared_error

from engine.expressions.expression_eval import node_count


def calibration_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean-residual bias normalized by the target's standard deviation."""
    residual = np.asarray(y_true) - np.asarray(y_pred)
    return float(np.abs(np.mean(residual)) / (np.std(y_true) + 1e-12))


def compute_fit_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    equation: str | None = None,
) -> dict[str, float]:
    """Compute standard regression fit metrics for a candidate model's predictions.

    Args:
        y_true: Ground-truth target values.
        y_pred: Model predictions.
        equation: Optional string representation of the discovered equation,
            measured as its expression-tree node count.

    Returns:
        Dict with keys: rmse, mae, calibration_error, symbolic_complexity.

    `symbolic_complexity` counts nodes, not characters. String length made
    `x0*x1` and `x0 * x1` different sizes and tracked how a backend chose to
    print a coefficient rather than the complexity of the law -- and the audit
    registers a pre-registered margin against this metric, so the unit decides
    verdicts. Nodes are also what the symbolic-regression literature reports,
    which is what lets a figure here be compared against a published one.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    complexity = node_count(equation) if equation else float("nan")

    return {
        "rmse": rmse,
        "mae": mae,
        "calibration_error": calibration_error(y_true, y_pred),
        "symbolic_complexity": complexity,
    }


def confidence_interval(
    values: Iterable[float],
    confidence_level: float = 0.95,
) -> tuple[float, float, float]:
    """Compute (mean, ci_low, ci_high) via a t-distribution confidence interval."""
    arr = np.asarray(list(values), dtype=float)
    mean = float(np.mean(arr))
    if len(arr) < 2:
        return mean, mean, mean
    sem = stats.sem(arr)
    lo, hi = stats.t.interval(confidence_level, len(arr) - 1, loc=mean, scale=sem)
    return mean, float(lo), float(hi)


def paired_significance(
    per_seed_rows: list[dict[str, Any]],
    metrics: list[str],
    baseline: str,
) -> list[dict[str, Any]]:
    """Paired t-test of each non-baseline model's metrics against a baseline model.

    Args:
        per_seed_rows: List of per-(model, seed) result dicts, each with a
            "model", "seed", and one entry per metric name.
        metrics: Metric names to test.
        baseline: The model name to compare all other models against.

    Returns:
        List of dicts with keys: model, baseline, metric, t_stat, p_value,
        mean_delta. Empty if the baseline model has no rows.
    """
    rows: list[dict[str, Any]] = []
    by_model: dict[str, list[dict[str, Any]]] = {}
    for row in per_seed_rows:
        by_model.setdefault(row["model"], []).append(row)

    if baseline not in by_model:
        return rows

    baseline_rows = sorted(by_model[baseline], key=lambda r: int(r["seed"]))

    for model, model_rows in by_model.items():
        if model == baseline:
            continue
        model_rows = sorted(model_rows, key=lambda r: int(r["seed"]))
        if len(model_rows) != len(baseline_rows):
            continue
        for metric in metrics:
            x = np.array([float(r[metric]) for r in model_rows])
            y = np.array([float(r[metric]) for r in baseline_rows])
            t_stat, p_val = stats.ttest_rel(x, y)
            rows.append(
                {
                    "model": model,
                    "baseline": baseline,
                    "metric": metric,
                    "t_stat": float(t_stat),
                    "p_value": float(p_val),
                    "mean_delta": float(np.mean(x - y)),
                }
            )
    return rows
