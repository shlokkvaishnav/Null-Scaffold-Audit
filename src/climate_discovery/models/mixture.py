import torch
import torch.nn as nn


class MixtureOfExperts(nn.Module):
    def __init__(self, gating_network, experts):
        """
        Args:
            gating_network: nn.Module that returns (log_probs, probs)
            experts: List of callables (models) or nn.Modules.
                     Each expert should accept the input features and return a prediction tensor.
        """
        super().__init__()
        self.gating_network = gating_network
        # If experts are nn.Modules, register them. If they are static functions/sklearn models, just store them.
        if experts:
            self.experts = (
                nn.ModuleList(experts) if isinstance(experts[0], nn.Module) else experts
            )
            self.is_torch_experts = isinstance(experts[0], nn.Module)
        else:
            self.experts = []
            self.is_torch_experts = False

    def forward(self, x_gate, x_expert):
        """
        Args:
            x_gate: Input to gating network (e.g. Lat, Lon, Time)
            x_expert: Input to experts (e.g. SST, SSS, Chl)
        """
        # Get Regime Probabilities
        _, probs = self.gating_network(x_gate)  # (Batch, K)

        # Get Expert Predictions
        expert_preds = []
        for i, expert in enumerate(self.experts):
            if self.is_torch_experts:
                pred = expert(x_expert)
            else:
                # Assuming expert is a sklearn/PySR model that expects numpy
                # We need to handle tensor->numpy->tensor conversion if not careful
                # For efficiency in training loop, simpler to use simple torch experts FIRST
                # or wrap symbolic equations as torch functions.
                pass
                # Placeholder: In the real notebook, we will likely wrap experts as simple torch linear layers
                # or pre-computed values if they are fixed.

                # For now, let's assume we pass in PRE-COMPUTED expert predictions to avoid mixed framework issues
                pass

        return probs

    def forward_with_precomputed(self, x_gate, expert_preds_tensor):
        """
        Forward pass when expert predictions are already computed (e.g. from PySR).
        Args:
            x_gate: (Batch, Gate_Features)
            expert_preds_tensor: (Batch, K) containing predictions from each expert.
        """
        _, probs = self.gating_network(x_gate)  # (Batch, K)

        # Weighted Sum: sum_k (pi_k * y_k)
        # probs: (B, K), expert_preds: (B, K) -> (B, 1) or (B,)

        y_hat = (probs * expert_preds_tensor).sum(dim=1)
        return y_hat, probs
