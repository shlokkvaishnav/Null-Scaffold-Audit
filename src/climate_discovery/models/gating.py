"""Gating network for soft regime assignment in SD-MoSE.

The gating network learns a continuous, probabilistic mapping from spatiotemporal
context to regime membership. Unlike hard clustering (K-means), soft gating:
- Represents fronts as smooth transitions (high entropy zones)
- Allows regime overlap for ambiguous regions
- Learns data-driven regime structure

Scientific design:
- Input: Lat, Lon, SST, SSS, Chl, Time → captures spatial + physical state
- Output: π_k(x) ∈ [0,1]^K with Σπ_k = 1 (simplex constraint via softmax)
- Temperature scaling: Controls regime sharpness (low temp → hard assignment)
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class GatingNetwork(nn.Module):
    """Neural network for soft regime assignment.
    
    Architecture:
        Input → [Linear → ReLU → Dropout → BatchNorm]* → Linear → Softmax → π_k
    
    Args:
        input_dim: Number of gating features (typically 5-8)
        num_regimes: Number of regimes K
        hidden_dims: List of hidden layer sizes
        dropout: Dropout probability for regularization
        temperature: Softmax temperature (lower = sharper boundaries)
        use_batchnorm: Whether to use batch normalization
        
    Example:
        >>> gating = GatingNetwork(
        ...     input_dim=7,  # lat, lon, sst, sss, log_chl, sin_month, cos_month
        ...     num_regimes=6,
        ...     hidden_dims=[64, 32],
        ...     dropout=0.1,
        ...     temperature=1.0
        ... )
        >>> probs = gating(x_gate)  # (N, K) probabilities
    """
    
    def __init__(
        self,
        input_dim: int,
        num_regimes: int,
        hidden_dims: Optional[List[int]] = None,
        dropout: float = 0.1,
        temperature: float = 1.0,
        use_batchnorm: bool = True,
        activation: str = "relu",
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.num_regimes = num_regimes
        self.temperature = temperature
        self.use_batchnorm = use_batchnorm
        
        # Default architecture
        if hidden_dims is None:
            hidden_dims = [64, 32]
        
        # Build network layers
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            # Linear layer
            layers.append(nn.Linear(prev_dim, hidden_dim))
            
            # Activation
            if activation == "relu":
                layers.append(nn.ReLU())
            elif activation == "gelu":
                layers.append(nn.GELU())
            elif activation == "tanh":
                layers.append(nn.Tanh())
            else:
                raise ValueError(f"Unknown activation: {activation}")
            
            # Dropout for regularization
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            
            # Batch normalization for stable training
            if use_batchnorm:
                layers.append(nn.BatchNorm1d(hidden_dim))
            
            prev_dim = hidden_dim
        
        # Output layer (no activation, will apply softmax)
        layers.append(nn.Linear(prev_dim, num_regimes))
        
        self.network = nn.Sequential(*layers)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Xavier initialization for stable gradients."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(
        self, 
        x: torch.Tensor,
        return_logits: bool = False
    ) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass with temperature-scaled softmax.
        
        Args:
            x: Gating features (N, input_dim)
            return_logits: If True, return (probs, logits)
            
        Returns:
            probs: Regime probabilities (N, num_regimes)
            logits: Raw logits (optional, if return_logits=True)
        """
        # Compute logits
        logits = self.network(x)
        
        # Temperature scaling: lower temp → sharper boundaries
        scaled_logits = logits / self.temperature
        
        # Softmax to get probabilities
        probs = F.softmax(scaled_logits, dim=1)
        
        if return_logits:
            return probs, logits
        return probs
    
    def get_entropy(self, probs: torch.Tensor, epsilon: float = 1e-10) -> torch.Tensor:
        """Compute Shannon entropy of regime distribution.
        
        H = -Σ π_k log(π_k)
        
        High entropy → uncertain regime assignment (transition zones)
        Low entropy → confident assignment (regime cores)
        
        Args:
            probs: Regime probabilities (N, K)
            epsilon: Numerical stability constant
            
        Returns:
            Entropy per sample (N,)
        """
        probs_safe = torch.clamp(probs, min=epsilon, max=1.0)
        entropy = -torch.sum(probs * torch.log(probs_safe), dim=1)
        return entropy
    
    def get_dominant_regime(self, probs: torch.Tensor) -> torch.Tensor:
        """Get most likely regime for each sample.
        
        Args:
            probs: Regime probabilities (N, K)
            
        Returns:
            Regime indices (N,) in range [0, K-1]
        """
        return torch.argmax(probs, dim=1)
    
    def set_temperature(self, temperature: float):
        """Update softmax temperature (useful for annealing)."""
        self.temperature = temperature


class EnsembleGatingNetwork(nn.Module):
    """Ensemble of gating networks for regime robustness validation.
    
    Trains multiple gating networks with different random seeds.
    Ensemble agreement indicates regime robustness.
    
    Args:
        input_dim: Number of gating features
        num_regimes: Number of regimes
        ensemble_size: Number of ensemble members
        **gating_kwargs: Arguments passed to each GatingNetwork
        
    Example:
        >>> ensemble = EnsembleGatingNetwork(
        ...     input_dim=7,
        ...     num_regimes=6,
        ...     ensemble_size=5,
        ...     hidden_dims=[64, 32],
        ... )
        >>> mean_probs, std_probs = ensemble(x_gate)
    """
    
    def __init__(
        self,
        input_dim: int,
        num_regimes: int,
        ensemble_size: int = 5,
        **gating_kwargs,
    ):
        super().__init__()
        
        self.ensemble_size = ensemble_size
        self.num_regimes = num_regimes
        
        # Create ensemble members
        self.members = nn.ModuleList([
            GatingNetwork(input_dim, num_regimes, **gating_kwargs)
            for _ in range(ensemble_size)
        ])
    
    def forward(
        self, 
        x: torch.Tensor,
        return_all: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through ensemble.
        
        Args:
            x: Gating features (N, input_dim)
            return_all: If True, return all member predictions
            
        Returns:
            mean_probs: Mean probabilities (N, K)
            std_probs: Standard deviation across ensemble (N, K)
            all_probs: All member predictions (ensemble_size, N, K) if return_all=True
        """
        # Get predictions from all members
        all_probs = torch.stack([member(x) for member in self.members], dim=0)
        # Shape: (ensemble_size, N, K)
        
        # Compute statistics
        mean_probs = torch.mean(all_probs, dim=0)  # (N, K)
        std_probs = torch.std(all_probs, dim=0)    # (N, K)
        
        if return_all:
            return mean_probs, std_probs, all_probs
        return mean_probs, std_probs
    
    def compute_agreement(self, x: torch.Tensor) -> torch.Tensor:
        """Compute fraction of ensemble members agreeing on dominant regime.
        
        Args:
            x: Gating features (N, input_dim)
            
        Returns:
            Agreement fraction per sample (N,) in [0, 1]
        """
        # Get all predictions
        all_probs = torch.stack([member(x) for member in self.members], dim=0)
        # (ensemble_size, N, K)
        
        # Get dominant regime for each member
        dominant = torch.argmax(all_probs, dim=2)  # (ensemble_size, N)
        
        # Mode (most common regime)
        mode_regime, _ = torch.mode(dominant, dim=0)  # (N,)
        
        # Fraction agreeing with mode
        agreement = torch.mean((dominant == mode_regime.unsqueeze(0)).float(), dim=0)
        
        return agreement


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def compute_regime_balance(probs: torch.Tensor) -> torch.Tensor:
    """Compute load balancing metric: how evenly samples are distributed.
    
    Ideal: Each regime gets 1/K of samples
    
    Args:
        probs: Regime probabilities (N, K)
        
    Returns:
        Balance score (scalar) in [0, 1], where 1 = perfectly balanced
    """
    # Expected count per regime (soft assignment)
    regime_counts = torch.sum(probs, dim=0)  # (K,)
    
    # Coefficient of variation (lower = more balanced)
    mean_count = torch.mean(regime_counts)
    std_count = torch.std(regime_counts)
    cv = std_count / (mean_count + 1e-6)
    
    # Convert to [0,1] score (0 = unbalanced, 1 = balanced)
    balance = torch.exp(-cv)
    
    return balance


def analyze_regime_structure(
    probs: torch.Tensor,
    coords: dict,
) -> dict:
    """Analyze learned regime structure for scientific interpretation.
    
    Args:
        probs: Regime probabilities (N, K)
        coords: Dict with 'lat', 'lon', 'time' arrays
        
    Returns:
        Dictionary with analysis results
    """
    n_samples, k = probs.shape
    
    # Entropy (transition zones)
    entropy = -torch.sum(probs * torch.log(probs + 1e-10), dim=1)
    
    # Dominant regime
    dominant = torch.argmax(probs, dim=1)
    
    # Confidence (max probability)
    confidence = torch.max(probs, dim=1).values
    
    # Regime usage (how many samples per regime)
    regime_usage = torch.sum(probs, dim=0)
    
    analysis = {
        "mean_entropy": float(torch.mean(entropy)),
        "std_entropy": float(torch.std(entropy)),
        "mean_confidence": float(torch.mean(confidence)),
        "regime_usage": regime_usage.cpu().numpy(),
        "dominant_regime": dominant.cpu().numpy(),
        "entropy": entropy.cpu().numpy(),
        "confidence": confidence.cpu().numpy(),
    }
    
    return analysis