"""Synthetic tabular regression dataset generation.

Used for quick smoke tests and ablation studies where no external dataset
(or Feynman benchmark equation) is required. The generated relationship
combines linear terms, an interaction term, and a nonlinear (sinusoidal)
term so that symbolic-regression and baseline models alike have a
nontrivial target to fit.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class SplitData:
    """Train/test split for a tabular regression dataset."""

    x_train: np.ndarray
    y_train: np.ndarray
    x_test: np.ndarray
    y_test: np.ndarray


def generate_synthetic_regression(
    seed: int,
    n_samples: int = 1000,
    n_features: int = 6,
    train_fraction: float = 0.8,
    noise_std: float = 0.1,
) -> SplitData:
    """Generate a synthetic tabular regression dataset with a known closed-form target.

    y = 1.5*x0 - 0.8*x1 + 0.5*x2*x3 + sin(x4) + noise

    Args:
        seed: Seed for the random number generator (deterministic sampling).
        n_samples: Total number of samples to generate.
        n_features: Number of input features (must be >= 5 for the formula above).
        train_fraction: Fraction of samples assigned to the training split.
        noise_std: Standard deviation of additive Gaussian noise on the target.

    Returns:
        SplitData with x_train/y_train/x_test/y_test arrays.
    """
    if n_features < 5:
        raise ValueError("n_features must be >= 5 for the synthetic regression formula.")

    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n_samples, n_features))
    y = (
        1.5 * x[:, 0]
        - 0.8 * x[:, 1]
        + 0.5 * x[:, 2] * x[:, 3]
        + np.sin(x[:, 4])
        + noise_std * rng.normal(size=n_samples)
    )
    split = int(train_fraction * n_samples)
    return SplitData(
        x_train=x[:split],
        y_train=y[:split],
        x_test=x[split:],
        y_test=y[split:],
    )
