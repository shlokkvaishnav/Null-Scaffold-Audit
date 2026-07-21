import torch
import torch.nn as nn
import torch.nn.functional as F


class SymbolicLibrary(nn.Module):
    """
    Transforms raw input features into a library of basis functions.
    Phi = { x, x^2, log(|x|+eps), sin(x), exp(x) }
    """
    def __init__(self, num_features: int):
        super().__init__()
        self.num_features = num_features
        self.num_bases_per_feature = 5
        self.output_dim = num_features * self.num_bases_per_feature

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        eps = 1e-6
        bases = [
            x,
            x ** 2,
            torch.log(torch.abs(x) + eps),
            torch.sin(x),
            torch.exp(x)
        ]
        return torch.cat(bases, dim=1)


class L0STE(torch.autograd.Function):
    """
    Straight-Through Estimator enabling hard L0 sparsity
    on continuous networks during gradient descent.
    """
    @staticmethod
    def forward(ctx, w, threshold):
        mask = (w.abs() > threshold).float()
        return w * mask

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None


class SymbolicGate(nn.Module):
    """
    Structured composition of symbolic primitives routing observational
    covariates to specific hypothesis-generation experts.
    """
    def __init__(self, num_features: int, num_experts: int, l0_threshold: float = 0.05):
        super().__init__()
        self.num_features = num_features
        self.num_experts = num_experts
        self.l0_threshold = l0_threshold

        self.symbolic_lib = SymbolicLibrary(num_features)
        lib_output_dim = self.symbolic_lib.output_dim

        # Learnable coefficients (alpha) and biases (beta) for each expert
        self.alpha = nn.Parameter(torch.randn(num_experts, lib_output_dim) * 0.1)
        self.beta = nn.Parameter(torch.zeros(num_experts))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Get symbolic basis features (Batch, Num_Bases)
        phi_x = self.symbolic_lib(x)

        # Apply strict L0 pseudo-differentiable mask
        sparse_alpha = L0STE.apply(self.alpha, self.l0_threshold)

        # Calculate routing logits: sparse_alpha * phi(x) + beta
        logits = F.linear(phi_x, sparse_alpha, self.beta)

        # Output regime routing probabilities
        return F.softmax(logits, dim=1)

    def sparsity_loss(self) -> torch.Tensor:
        """
        Calculate formal L0 surrogate penalty
        (counting active gates).
        """
        sparse_alpha = L0STE.apply(self.alpha, self.l0_threshold)
        # Active gate sum approximation tracking strictly non-zero boundaries
        return torch.sum(sparse_alpha.abs() > 0.0).float()
