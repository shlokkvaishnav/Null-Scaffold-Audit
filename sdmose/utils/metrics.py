"""Evaluation metrics utilities for SD-MoSE outputs."""

from __future__ import annotations

from typing import Dict, Iterable, Optional

import numpy as np


_EPS = 1e-12


def _validate_probabilities(y_prob: np.ndarray) -> np.ndarray:
    probs = np.asarray(y_prob, dtype=float)
    if probs.ndim != 2:
        raise ValueError(f"y_prob must be 2D (n_samples, n_classes), got {probs.shape}")
    if probs.shape[0] == 0 or probs.shape[1] == 0:
        raise ValueError("y_prob must have at least one sample and one class")
    row_sums = probs.sum(axis=1, keepdims=True)
    if np.any(row_sums <= 0):
        raise ValueError("Each probability row must have positive mass")
    return probs / row_sums


def _to_one_hot(y_true: np.ndarray, n_classes: int) -> np.ndarray:
    labels = np.asarray(y_true, dtype=int)
    if labels.ndim != 1:
        raise ValueError(f"y_true must be 1D labels, got shape {labels.shape}")
    if labels.size == 0:
        raise ValueError("y_true must contain at least one label")
    if np.any(labels < 0) or np.any(labels >= n_classes):
        raise ValueError("y_true contains labels outside [0, n_classes)")

    one_hot = np.zeros((labels.shape[0], n_classes), dtype=float)
    one_hot[np.arange(labels.shape[0]), labels] = 1.0
    return one_hot


def expected_calibration_error(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    """Compute multiclass Expected Calibration Error (ECE) from confidence bins."""
    if n_bins < 1:
        raise ValueError("n_bins must be >= 1")

    probs = _validate_probabilities(y_prob)
    labels = np.asarray(y_true, dtype=int)
    if labels.shape[0] != probs.shape[0]:
        raise ValueError("y_true and y_prob must have same number of samples")

    confidences = probs.max(axis=1)
    predictions = probs.argmax(axis=1)
    correctness = (predictions == labels).astype(float)

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    total = float(probs.shape[0])
    ece = 0.0

    for lower, upper in zip(bin_edges[:-1], bin_edges[1:]):
        if upper == 1.0:
            in_bin = (confidences >= lower) & (confidences <= upper)
        else:
            in_bin = (confidences >= lower) & (confidences < upper)

        if not np.any(in_bin):
            continue

        accuracy = float(correctness[in_bin].mean())
        avg_confidence = float(confidences[in_bin].mean())
        ece += (in_bin.sum() / total) * abs(accuracy - avg_confidence)

    return float(ece)


def brier_score_multiclass(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Compute multiclass Brier score (mean squared error over classes)."""
    probs = _validate_probabilities(y_prob)
    one_hot = _to_one_hot(y_true, probs.shape[1])
    return float(np.mean(np.sum((probs - one_hot) ** 2, axis=1)))


def negative_log_likelihood(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Compute multiclass negative log likelihood."""
    probs = _validate_probabilities(y_prob)
    labels = np.asarray(y_true, dtype=int)
    if labels.shape[0] != probs.shape[0]:
        raise ValueError("y_true and y_prob must have same number of samples")
    chosen = probs[np.arange(probs.shape[0]), labels]
    return float(-np.mean(np.log(np.clip(chosen, _EPS, 1.0))))


def calibration_diagnostics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
    include: Optional[Iterable[str]] = None,
) -> Dict[str, float]:
    """Compute calibration diagnostics for probabilistic belief outputs."""
    include_set = set(include or ["ece", "brier", "nll"])
    diagnostics: Dict[str, float] = {}

    if "ece" in include_set:
        diagnostics["ece"] = expected_calibration_error(y_true, y_prob, n_bins=n_bins)
    if "brier" in include_set:
        diagnostics["brier"] = brier_score_multiclass(y_true, y_prob)
    if "nll" in include_set:
        diagnostics["nll"] = negative_log_likelihood(y_true, y_prob)

    return diagnostics


def calculate_metrics(y_true, y_pred):
    """Backwards-compatible placeholder API kept for legacy imports."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    mae = float(np.mean(np.abs(y_true - y_pred)))
    return {"rmse": rmse, "mae": mae}
