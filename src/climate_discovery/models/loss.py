import torch
import torch.nn as nn

class RegimeConsistencyLoss(nn.Module):
    def __init__(self, weight=0.1):
        super().__init__()
        self.weight = weight

    def forward(self, probs_map):
        """
        Calculates Total Variation (Spatial Smoothness) loss.
        Args:
            probs_map: (Batch, Height, Width, N_Regimes)
        """
        # Vertical Smoothness (Height)
        diff_h = torch.abs(probs_map[:, 1:, :, :] - probs_map[:, :-1, :, :])
        # Horizontal Smoothness (Width)
        diff_w = torch.abs(probs_map[:, :, 1:, :] - probs_map[:, :, :-1, :])

        # We mean over all dimensions to get a scalar
        loss = diff_h.mean() + diff_w.mean()

        return self.weight * loss