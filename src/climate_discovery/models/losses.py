"""Spatial smoothness, temporal consistency, and L1 sparsity losses."""
import torch
import torch.nn as nn


class RegimeConsistencyLoss(nn.Module):
    """Total variation (spatial smoothness) over regime probabilities."""

    def __init__(self, weight=0.1):
        super().__init__()
        self.weight = weight

    def forward(self, probs_map):
        # probs_map: (B, H, W, K)
        diff_h = torch.abs(probs_map[:, 1:, :, :] - probs_map[:, :-1, :, :])
        diff_w = torch.abs(probs_map[:, :, 1:, :] - probs_map[:, :, :-1, :])
        return self.weight * (diff_h.mean() + diff_w.mean())


class TemporalConsistencyLoss(nn.Module):
    """Total variation over time for regime probabilities."""

    def __init__(self, weight=0.1):
        super().__init__()
        self.weight = weight

    def forward(self, probs_sequence):
        diff_t = torch.abs(probs_sequence[:, 1:] - probs_sequence[:, :-1])
        return self.weight * diff_t.mean()


class L1SparseLoss(nn.Module):
    """L1 on first-layer weights for sparsity."""

    def __init__(self, weight=0.01):
        super().__init__()
        self.weight = weight

    def forward(self, model):
        for name, param in model.named_parameters():
            if "weight" in name:
                return self.weight * torch.norm(param, 1)
        return torch.tensor(0.0)
