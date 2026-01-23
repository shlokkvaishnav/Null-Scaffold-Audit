"""SD-MoSE: Soft-Dynamic Mixture of Symbolic Experts.

Combines:
1. Neural gating network → soft regime assignment
2. Symbolic experts (PySR) → interpretable laws per regime
3. Weighted mixture → f(x) = Σ π_k(x) * f_k(x)

Training alternates between:
- Fixing experts, training gating (differentiable)
- Fixing gating, refitting experts (symbolic regression)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class SDMoSE(nn.Module):
    """SD-MoSE: Soft-Dynamic Mixture of Symbolic Experts.
    
    Args:
        gating_network: Neural network for regime assignment
        num_regimes: Number of regimes K
        expert_features: List of feature names for experts
        device: CPU or CUDA
        
    The model operates in two modes:
    1. Training mode: Gating network learns, experts are external (PySR)
    2. Inference mode: Both gating and experts are fixed
    
    Example:
        >>> from climate_discovery.models.gating import GatingNetwork
        >>> 
        >>> gating = GatingNetwork(input_dim=7, num_regimes=6)
        >>> model = SDMoSE(gating, num_regimes=6)
        >>> 
        >>> # Training: Get regime probs, then fit symbolic experts externally
        >>> probs = model.get_regime_probs(x_gate)
        >>> # ... fit PySR experts using probs as weights ...
        >>> 
        >>> # Inference: Provide pre-computed expert predictions
        >>> y_pred = model.forward_mixture(x_gate, expert_predictions)
    """
    
    def __init__(
        self,
        gating_network: nn.Module,
        num_regimes: int,
        expert_features: Optional[List[str]] = None,
        device: str = "cpu",
    ):
        super().__init__()
        
        self.gating_network = gating_network
        self.num_regimes = num_regimes
        self.expert_features = expert_features or []
        self.device = device
        
        # Move gating to device
        self.gating_network.to(device)
        
        # Storage for symbolic experts (fitted externally)
        self.symbolic_experts = None
        self.expert_equations = {}
    
    def get_regime_probs(
        self, 
        x_gate: torch.Tensor
    ) -> torch.Tensor:
        """Get soft regime assignments from gating network.
        
        Args:
            x_gate: Gating features (N, D_gate)
            
        Returns:
            Regime probabilities (N, K)
        """
        return self.gating_network(x_gate)
    
    def forward_mixture(
        self,
        x_gate: torch.Tensor,
        expert_predictions: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Mixture prediction: y = Σ π_k(x) * f_k(x)
        
        Args:
            x_gate: Gating features (N, D_gate)
            expert_predictions: Pre-computed expert outputs (N, K)
            
        Returns:
            y_pred: Weighted mixture prediction (N,)
            probs: Regime probabilities (N, K)
        """
        # Get regime probabilities
        probs = self.get_regime_probs(x_gate)  # (N, K)
        
        # Weighted sum of expert predictions
        # (N, K) ⊙ (N, K) → (N, K) → (N,)
        y_pred = torch.sum(probs * expert_predictions, dim=1)
        
        return y_pred, probs
    
    def forward(
        self,
        x_gate: torch.Tensor,
        expert_predictions: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.
        
        If expert_predictions provided: Full mixture prediction
        If not: Return regime probabilities only (for training gating)
        
        Args:
            x_gate: Gating features (N, D_gate)
            expert_predictions: Optional expert predictions (N, K)
            
        Returns:
            If expert_predictions given: (y_pred, probs)
            Otherwise: (probs, probs)
        """
        probs = self.get_regime_probs(x_gate)
        
        if expert_predictions is not None:
            y_pred = torch.sum(probs * expert_predictions, dim=1)
            return y_pred, probs
        else:
            # Training mode: return probs for gating optimization
            return probs, probs
    
    def attach_symbolic_experts(self, experts):
        """Attach fitted symbolic experts for inference.
        
        Args:
            experts: MixtureOfSymbolicExperts instance (from symbolic.py)
        """
        self.symbolic_experts = experts
        self.expert_equations = experts.get_all_equations()
        logger.info(f"Attached {self.num_regimes} symbolic experts")
    
    def predict_numpy(
        self,
        X_gate: np.ndarray,
        X_expert: np.ndarray,
    ) -> np.ndarray:
        """End-to-end prediction (NumPy interface for sklearn compatibility).
        
        Args:
            X_gate: Gating features (N, D_gate)
            X_expert: Expert features (N, D_expert)
            
        Returns:
            Predictions (N,)
        """
        if self.symbolic_experts is None:
            raise ValueError("No symbolic experts attached. Call attach_symbolic_experts() first.")
        
        # Convert to torch
        x_gate_torch = torch.from_numpy(X_gate).float().to(self.device)
        
        # Get regime probabilities
        with torch.no_grad():
            probs = self.get_regime_probs(x_gate_torch).cpu().numpy()
        
        # Get expert predictions (NumPy)
        y_pred = self.symbolic_experts.predict(X_expert, probs)
        
        return y_pred
    
    def save(self, path: str | Path):
        """Save gating network weights.
        
        Note: Symbolic experts saved separately via MixtureOfSymbolicExperts.save_equations()
        """
        path = Path(path)
        torch.save({
            'gating_state_dict': self.gating_network.state_dict(),
            'num_regimes': self.num_regimes,
            'expert_features': self.expert_features,
            'expert_equations': self.expert_equations,
        }, path)
        logger.info(f"Model saved to {path}")
    
    def load(self, path: str | Path):
        """Load gating network weights."""
        path = Path(path)
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        
        self.gating_network.load_state_dict(checkpoint['gating_state_dict'])
        self.num_regimes = checkpoint['num_regimes']
        self.expert_features = checkpoint['expert_features']
        self.expert_equations = checkpoint.get('expert_equations', {})
        
        logger.info(f"Model loaded from {path}")


class TorchSymbolicExpertWrapper(nn.Module):
    """Wraps a PySR equation as a PyTorch module for end-to-end training.
    
    Useful for fine-tuning or joint optimization after symbolic discovery.
    
    Args:
        equation_str: Symbolic equation from PySR
        variable_names: Feature names
    """
    
    def __init__(
        self,
        equation_str: str,
        variable_names: List[str],
    ):
        super().__init__()
        self.equation_str = equation_str
        self.variable_names = variable_names
        
        # TODO: Parse equation and create torch computation graph
        # For now, this is a placeholder for future work
        raise NotImplementedError(
            "Automatic equation → PyTorch conversion not yet implemented.\n"
            "Use experts in NumPy mode via predict_numpy()."
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Evaluate equation on input
        raise NotImplementedError


# =============================================================================
# ALTERNATING OPTIMIZATION UTILITIES
# =============================================================================

def alternating_training_step(
    model: SDMoSE,
    train_loader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    expert_predictions: torch.Tensor,
    device: str = "cpu",
) -> float:
    """Single training step for gating network with fixed expert predictions.
    
    Args:
        model: SDMoSE model
        train_loader: DataLoader for training data
        optimizer: Optimizer for gating network
        criterion: Loss function
        expert_predictions: Pre-computed expert outputs (N, K)
        device: CPU or CUDA
        
    Returns:
        Average loss for epoch
    """
    model.train()
    total_loss = 0.0
    n_batches = 0
    
    for batch_idx, (X_expert, X_gate, y) in enumerate(train_loader):
        # Move to device
        X_gate = X_gate.to(device)
        y = y.to(device)
        
        # Get expert predictions for this batch
        batch_expert_preds = expert_predictions[batch_idx].to(device)
        
        # Forward pass
        y_pred, probs = model.forward_mixture(X_gate, batch_expert_preds)
        
        # Compute loss
        loss = criterion(y_pred, y)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        n_batches += 1
    
    return total_loss / n_batches


def compute_expert_predictions_batch(
    symbolic_experts,
    X_expert: np.ndarray,
    regime_probs: np.ndarray,
    batch_size: int = 1000,
) -> np.ndarray:
    """Compute predictions from all symbolic experts.
    
    Args:
        symbolic_experts: MixtureOfSymbolicExperts instance
        X_expert: Expert features (N, D_expert)
        regime_probs: Regime probabilities (N, K)
        batch_size: Process in batches to save memory
        
    Returns:
        Expert predictions (N, K)
    """
    n_samples = len(X_expert)
    k = len(symbolic_experts.experts)
    expert_preds = np.zeros((n_samples, k))
    
    # Process in batches
    for i in range(0, n_samples, batch_size):
        batch_end = min(i + batch_size, n_samples)
        X_batch = X_expert[i:batch_end]
        
        for k_idx, expert in enumerate(symbolic_experts.experts):
            try:
                expert_preds[i:batch_end, k_idx] = expert.predict(X_batch)
            except Exception as e:
                logger.warning(f"Expert {k_idx} prediction failed: {e}")
                expert_preds[i:batch_end, k_idx] = 0.0
    
    return expert_preds


# =============================================================================
# EVALUATION UTILITIES
# =============================================================================

def evaluate_mixture(
    model: SDMoSE,
    X_gate: np.ndarray,
    X_expert: np.ndarray,
    y_true: np.ndarray,
    symbolic_experts,
) -> Dict:
    """Evaluate SD-MoSE performance.
    
    Returns:
        Dictionary with evaluation metrics
    """
    # Get predictions
    y_pred = model.predict_numpy(X_gate, X_expert)
    
    # Get regime probabilities
    x_gate_torch = torch.from_numpy(X_gate).float().to(model.device)
    with torch.no_grad():
        probs = model.get_regime_probs(x_gate_torch).cpu().numpy()
    
    # Compute metrics
    mse = np.mean((y_pred - y_true) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(y_pred - y_true))
    
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1 - ss_res / (ss_tot + 1e-10)
    
    # Regime statistics
    dominant_regime = np.argmax(probs, axis=1)
    entropy = -np.sum(probs * np.log(probs + 1e-10), axis=1)
    
    return {
        "mse": float(mse),
        "rmse": float(rmse),
        "mae": float(mae),
        "r2": float(r2),
        "mean_entropy": float(np.mean(entropy)),
        "regime_usage": np.bincount(dominant_regime, minlength=model.num_regimes),
    }