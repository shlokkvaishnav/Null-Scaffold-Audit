"""
HMM-style transition model for dynamic regime persistence.
Pr(z_t | z_{t-1}) = A[z_{t-1}, z_t]. Used to regularize gating over time.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from typing import Optional

try:
    from hmmlearn import hmm
except ImportError:
    hmm = None


def fit_transition_matrix(
    assignments: np.ndarray,
    n_regimes: int,
) -> np.ndarray:
    """
    Estimate K x K transition matrix A from hard regime sequence.
    assignments: (T,) or (T, N) integer labels in [0, K-1].
    """
    assignments = np.asarray(assignments).ravel()
    A = np.zeros((n_regimes, n_regimes))
    for t in range(1, len(assignments)):
        i, j = int(assignments[t - 1]), int(assignments[t])
        if 0 <= i < n_regimes and 0 <= j < n_regimes:
            A[i, j] += 1
    row_sum = A.sum(axis=1, keepdims=True)
    row_sum = np.where(row_sum > 0, row_sum, 1.0)
    A = A / row_sum
    return A.astype(np.float32)


def fit_transition_matrix_hmmlearn(
    assignments: np.ndarray,
    n_regimes: int,
) -> np.ndarray:
    """Use hmmlearn.GaussianHMM to fit transition matrix from hard assignments."""
    if hmm is None:
        return fit_transition_matrix(assignments, n_regimes)
    assignments = np.asarray(assignments).ravel().reshape(-1, 1)
    model = hmm.GaussianHMM(n_components=n_regimes, covariance_type="full", n_iter=50)
    model.fit(assignments)
    return np.asarray(model.transmat_, dtype=np.float32)


class HMMTransitionLoss(nn.Module):
    """
    Penalize deviation of soft regime probs from HMM dynamics:
    pi(t) ~ A^T @ pi(t-1). Loss = || pi(t) - A^T pi(t-1) ||.
    """

    def __init__(self, A: np.ndarray, weight: float = 0.1):
        super().__init__()
        self.weight = weight
        self.register_buffer("A", torch.from_numpy(np.asarray(A, dtype=np.float32)))

    def forward(self, probs_sequence: torch.Tensor) -> torch.Tensor:
        """
        probs_sequence: (B, T, K). Uses consecutive pairs (t-1, t).
        """
        if probs_sequence.dim() != 3 or probs_sequence.size(1) < 2:
            return probs_sequence.new_zeros(1)
        p_prev = probs_sequence[:, :-1]
        p_cur = probs_sequence[:, 1:]
        pred = torch.einsum("kj,btj->btk", self.A.T, p_prev)
        diff = (p_cur - pred).pow(2).mean()
        return self.weight * diff
