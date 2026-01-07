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

class L1SparseLoss(nn.Module):
    def __init__(self, weight=0.01):
        super().__init__()
        self.weight = weight

    def forward(self, model):
        """
        Calculates L1 loss on the weights of the *first layer* of the model.
        This encourages sparsity in feature selection.
        """
        l1_reg = torch.tensor(0., requires_grad=True)
        
        # We assume the first layer is the one connected to inputs
        # For GatingNetwork, it is model.net[0] (Linear)
        # We iterate to find the first Linear layer just to be safe/generic
        for name, param in model.named_parameters():
            if 'weight' in name:
                l1_reg = l1_reg + torch.norm(param, 1)
                # We only want the FIRST layer for feature selection sparsity
                # But typically L1 on all layers is fine for general sparsity.
                # For specific "Variable Selection", we care most about the input weights.
                # Let's target the input layer specifically if possible, but general L1 is okay too.
                # To be precise for "Variable Discovery", we should punish the Input->Hidden weights strongly.
                break # ONLY punish the first layer found!
        
        return self.weight * l1_reg