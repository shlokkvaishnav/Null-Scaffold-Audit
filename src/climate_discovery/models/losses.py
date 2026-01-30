"""Loss functions and regularization for SD-MoSE training.

Includes:
1. Prediction loss (MSE/MAE)
2. Entropy regularization (encourage confident assignments)
3. Load balancing (prevent regime collapse)
4. Spatial smoothness (for gridded data)
5. Temporal consistency (for time series)
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class SDMoSELoss(nn.Module):
    """Combined loss for SD-MoSE training.
    
    L_total = L_pred + λ_entropy * L_entropy + λ_balance * L_balance + ...
    
    Args:
        prediction_loss: Base prediction loss ('mse' or 'mae')
        entropy_weight: Weight for entropy regularization
        balance_weight: Weight for load balancing
        spatial_weight: Weight for spatial smoothness
        temporal_weight: Weight for temporal consistency
        
    Example:
        >>> criterion = SDMoSELoss(
        ...     prediction_loss='mse',
        ...     entropy_weight=0.01,
        ...     balance_weight=0.1
        ... )
        >>> loss = criterion(y_pred, y_true, regime_probs)
    """
    
    def __init__(
        self,
        prediction_loss: str = "mse",
        entropy_weight: float = 0.005,  # Mild entropy penalty (reduced from 0.01)
        balance_weight: float = 0.0,    # Disabled by default (use spatial/temporal instead)
        spatial_weight: float = 0.05,   # Enable spatial smoothness
        temporal_weight: float = 0.03,  # Enable temporal consistency
    ):
        super().__init__()
        
        # Base prediction loss
        if prediction_loss == "mse":
            self.prediction_criterion = nn.MSELoss()
        elif prediction_loss == "mae":
            self.prediction_criterion = nn.L1Loss()
        else:
            raise ValueError(f"Unknown prediction loss: {prediction_loss}")
        
        # Regularization weights
        self.entropy_weight = entropy_weight
        self.balance_weight = balance_weight
        self.spatial_weight = spatial_weight
        self.temporal_weight = temporal_weight
    
    def forward(
        self,
        y_pred: torch.Tensor,
        y_true: torch.Tensor,
        regime_probs: torch.Tensor,
        spatial_coords: Optional[torch.Tensor] = None,
        temporal_indices: Optional[torch.Tensor] = None,
    ) -> dict:
        """Compute total loss with breakdown.
        
        Args:
            y_pred: Predictions (N,)
            y_true: Ground truth (N,)
            regime_probs: Regime probabilities (N, K)
            spatial_coords: Optional (lat, lon) for spatial smoothness (N, 2)
            temporal_indices: Optional timestep indices (N,) for temporal consistency
            
        Returns:
            Dictionary with 'total' loss and component losses
        """
        # 1. Prediction loss
        loss_pred = self.prediction_criterion(y_pred, y_true)
        
        # 2. Entropy regularization (encourage confident regime assignment)
        loss_entropy = entropy_loss(regime_probs) if self.entropy_weight > 0 else 0.0
        
        # 3. Load balancing (prevent regime collapse)
        loss_balance = load_balance_loss(regime_probs) if self.balance_weight > 0 else 0.0
        
        # 4. Spatial smoothness (optional, for gridded data)
        loss_spatial = 0.0
        if self.spatial_weight > 0 and spatial_coords is not None:
            loss_spatial = spatial_smoothness_loss(regime_probs, spatial_coords)
        
        # 5. Temporal consistency (optional, for time series)
        loss_temporal = 0.0
        if self.temporal_weight > 0 and temporal_indices is not None:
            loss_temporal = temporal_consistency_loss(regime_probs, temporal_indices)
        
        # Total weighted loss
        total_loss = (
            loss_pred
            + self.entropy_weight * loss_entropy
            + self.balance_weight * loss_balance
            + self.spatial_weight * loss_spatial
            + self.temporal_weight * loss_temporal
        )
        
        return {
            "total": total_loss,
            "prediction": loss_pred,
            "entropy": loss_entropy,
            "balance": loss_balance,
            "spatial": loss_spatial,
            "temporal": loss_temporal,
        }


# =============================================================================
# INDIVIDUAL LOSS COMPONENTS
# =============================================================================

def entropy_loss(probs: torch.Tensor, epsilon: float = 1e-10) -> torch.Tensor:
    """Entropy regularization: Encourage confident regime assignments.
    
    L_entropy = mean(-Σ π_k log π_k)
    
    Lower entropy = more confident assignments = clearer regime boundaries
    
    Args:
        probs: Regime probabilities (N, K)
        epsilon: Numerical stability
        
    Returns:
        Scalar loss
    """
    probs_safe = torch.clamp(probs, min=epsilon, max=1.0)
    entropy = -torch.sum(probs * torch.log(probs_safe), dim=1)
    return torch.mean(entropy)


def load_balance_loss(probs: torch.Tensor) -> torch.Tensor:
    """Load balancing: Encourage equal usage of all regimes.
    
    Prevents regime collapse (all samples assigned to one regime).
    
    L_balance = std(regime_usage) / mean(regime_usage)
    
    Args:
        probs: Regime probabilities (N, K)
        
    Returns:
        Scalar loss
    """
    # Expected count per regime (soft assignment)
    regime_usage = torch.sum(probs, dim=0)  # (K,)
    
    # Coefficient of variation
    mean_usage = torch.mean(regime_usage)
    std_usage = torch.std(regime_usage)
    cv = std_usage / (mean_usage + 1e-10)
    
    return cv


def spatial_smoothness_loss(
    probs: torch.Tensor,
    coords: torch.Tensor,
    neighbor_threshold: float = 5.0,
) -> torch.Tensor:
    """Spatial smoothness: Nearby points should have similar regime assignments.
    
    For gridded ocean data, adjacent grid cells should have smooth regime transitions
    unless there's a physical front.
    
    Args:
        probs: Regime probabilities (N, K)
        coords: Spatial coordinates (N, 2) - [lat, lon]
        neighbor_threshold: Distance threshold for neighbors (degrees)
        
    Returns:
        Scalar loss
    """
    n_samples = probs.shape[0]
    
    # Subsample for efficiency (computing full distance matrix is O(N²))
    if n_samples > 1000:
        indices = torch.randperm(n_samples)[:1000]
        probs_sub = probs[indices]
        coords_sub = coords[indices]
    else:
        probs_sub = probs
        coords_sub = coords
    
    # Compute pairwise distances (Euclidean approximation)
    # More accurate: Haversine distance, but this is faster
    dist_matrix = torch.cdist(coords_sub, coords_sub, p=2)  # (N', N')
    
    # Find neighbors (within threshold)
    neighbors_mask = (dist_matrix < neighbor_threshold) & (dist_matrix > 0)
    
    if torch.sum(neighbors_mask) == 0:
        return torch.tensor(0.0, device=probs.device)
    
    # Compute probability difference for neighbors
    prob_diff = torch.abs(probs_sub.unsqueeze(1) - probs_sub.unsqueeze(0))  # (N', N', K)
    prob_diff = torch.sum(prob_diff, dim=2)  # (N', N') - Total variation distance
    
    # Average over neighbors
    neighbor_diff = prob_diff[neighbors_mask]
    smoothness_loss = torch.mean(neighbor_diff)
    
    return smoothness_loss


def temporal_consistency_loss(
    probs: torch.Tensor,
    time_indices: torch.Tensor,
    window_size: int = 3,
) -> torch.Tensor:
    """Temporal consistency: Regime assignments should be stable over time.
    
    For time series data, consecutive timesteps should have similar regimes
    unless there's a rapid physical transition.
    
    Args:
        probs: Regime probabilities (N, K)
        time_indices: Timestep indices (N,) - integer month/year identifiers
        window_size: Temporal window for smoothness (months)
        
    Returns:
        Scalar loss
    """
    # Sort by time
    sorted_indices = torch.argsort(time_indices)
    probs_sorted = probs[sorted_indices]
    time_sorted = time_indices[sorted_indices]
    
    # Find consecutive timesteps (within window)
    time_diff = time_sorted[1:] - time_sorted[:-1]
    consecutive_mask = time_diff <= window_size
    
    if torch.sum(consecutive_mask) == 0:
        return torch.tensor(0.0, device=probs.device)
    
    # Compute probability difference for consecutive steps
    prob_diff = torch.sum(torch.abs(probs_sorted[1:] - probs_sorted[:-1]), dim=1)
    
    # Average over valid consecutive pairs
    consistency_loss = torch.mean(prob_diff[consecutive_mask])
    
    return consistency_loss


# =============================================================================
# AUXILIARY LOSSES
# =============================================================================

class WeightedMSELoss(nn.Module):
    """MSE loss with sample weights (for regime-weighted training).
    
    Args:
        reduction: 'mean' or 'sum'
    """
    
    def __init__(self, reduction: str = "mean"):
        super().__init__()
        self.reduction = reduction
    
    def forward(
        self,
        y_pred: torch.Tensor,
        y_true: torch.Tensor,
        weights: torch.Tensor,
    ) -> torch.Tensor:
        """Compute weighted MSE.
        
        Args:
            y_pred: Predictions (N,)
            y_true: Ground truth (N,)
            weights: Sample weights (N,) - typically regime probabilities
            
        Returns:
            Scalar loss
        """
        squared_error = (y_pred - y_true) ** 2
        weighted_error = squared_error * weights
        
        if self.reduction == "mean":
            return torch.sum(weighted_error) / (torch.sum(weights) + 1e-10)
        elif self.reduction == "sum":
            return torch.sum(weighted_error)
        else:
            return weighted_error


class DiversityLoss(nn.Module):
    """Encourage diversity in expert predictions.
    
    Prevents all experts from learning the same function.
    
    L_diversity = -mean(pairwise_distance(expert_predictions))
    """
    
    def __init__(self, weight: float = 0.01):
        super().__init__()
        self.weight = weight
    
    def forward(self, expert_preds: torch.Tensor) -> torch.Tensor:
        """Compute diversity loss.
        
        Args:
            expert_preds: Expert predictions (N, K)
            
        Returns:
            Scalar loss
        """
        # Compute pairwise correlation between experts
        k = expert_preds.shape[1]
        
        # Normalize predictions
        expert_preds_norm = F.normalize(expert_preds, p=2, dim=0)
        
        # Correlation matrix
        corr = torch.mm(expert_preds_norm.t(), expert_preds_norm)  # (K, K)
        
        # Penalize high correlation (want diversity)
        # Average off-diagonal elements
        mask = ~torch.eye(k, dtype=torch.bool, device=expert_preds.device)
        avg_corr = torch.mean(torch.abs(corr[mask]))
        
        return self.weight * avg_corr


# =============================================================================
# TRAINING UTILITIES
# =============================================================================

def compute_loss_with_metrics(
    criterion: SDMoSELoss,
    y_pred: torch.Tensor,
    y_true: torch.Tensor,
    regime_probs: torch.Tensor,
) -> dict:
    """Compute loss and additional metrics for logging.
    
    Returns:
        Dictionary with loss components and metrics
    """
    # Compute loss
    loss_dict = criterion(y_pred, y_true, regime_probs)
    
    # Additional metrics
    with torch.no_grad():
        # R²
        ss_res = torch.sum((y_true - y_pred) ** 2)
        ss_tot = torch.sum((y_true - torch.mean(y_true)) ** 2)
        r2 = 1 - ss_res / (ss_tot + 1e-10)
        
        # Regime statistics
        dominant_regime = torch.argmax(regime_probs, dim=1)
        regime_counts = torch.bincount(dominant_regime, minlength=regime_probs.shape[1])
        
        # Add to dict
        loss_dict.update({
            "r2": r2.item(),
            "regime_usage": regime_counts.cpu().numpy(),
        })
    
    return loss_dict


class EarlyStopping:
    """Early stopping to prevent overfitting.
    
    Args:
        patience: Number of epochs to wait before stopping
        min_delta: Minimum change to qualify as improvement
        
    Example:
        >>> early_stop = EarlyStopping(patience=10)
        >>> for epoch in range(100):
        ...     val_loss = train_epoch(...)
        ...     if early_stop(val_loss):
        ...         break
    """
    
    def __init__(self, patience: int = 10, min_delta: float = 1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
    
    def __call__(self, val_loss: float) -> bool:
        """Check if should stop training.
        
        Returns:
            True if should stop, False otherwise
        """
        if self.best_loss is None:
            self.best_loss = val_loss
            return False
        
        if val_loss < self.best_loss - self.min_delta:
            # Improvement
            self.best_loss = val_loss
            self.counter = 0
            return False
        else:
            # No improvement
            self.counter += 1
            if self.counter >= self.patience:
                return True
            return False