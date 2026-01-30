"""Hierarchical loss function for multi-scale SD-MoSE.

Add to losses.py or import separately.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional

from .losses import spatial_smoothness_loss, temporal_consistency_loss


class HierarchicalSDMoSELoss(nn.Module):
    """Loss function for hierarchical SD-MoSE.
    
    Combines prediction loss with hierarchical regularization:
    1. MSE/MAE prediction loss
    2. Coarse entropy (encourage diversity at Level 1)
    3. Fine entropy (encourage diversity at Level 2, given coarse)
    4. Hierarchy consistency (fine regimes should differ across coarse)
    5. Spatial/temporal smoothness (optional, inherited from base)
    
    Args:
        prediction_loss: "mse" or "mae"
        coarse_entropy_weight: Weight for coarse regime diversity
        fine_entropy_weight: Weight for fine regime diversity
        consistency_weight: Weight for hierarchy consistency
        spatial_weight: Spatial smoothness weight
        temporal_weight: Temporal smoothness weight
        
    Example:
        >>> criterion = HierarchicalSDMoSELoss(
        ...     coarse_entropy_weight=0.01,
        ...     fine_entropy_weight=0.005,
        ...     consistency_weight=0.02,
        ... )
        >>> loss_dict = criterion(y_pred, y_true, p_coarse, p_fine, coords, time_idx)
    """
    
    def __init__(
        self,
        prediction_loss: str = "mse",
        coarse_entropy_weight: float = 0.01,
        fine_entropy_weight: float = 0.005,
        consistency_weight: float = 0.02,
        spatial_weight: float = 0.0,
        temporal_weight: float = 0.0,
    ):
        super().__init__()
        
        self.prediction_loss = prediction_loss
        self.coarse_entropy_weight = coarse_entropy_weight
        self.fine_entropy_weight = fine_entropy_weight
        self.consistency_weight = consistency_weight
        self.spatial_weight = spatial_weight
        self.temporal_weight = temporal_weight
    
    def forward(
        self,
        y_pred: torch.Tensor,
        y_true: torch.Tensor,
        p_coarse: torch.Tensor,
        p_fine: torch.Tensor,
        spatial_coords: Optional[torch.Tensor] = None,
        time_indices: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Compute hierarchical loss.
        
        Args:
            y_pred: Predictions (N,)
            y_true: Ground truth (N,)
            p_coarse: Coarse probabilities (N, n_coarse)
            p_fine: Fine probabilities (N, n_coarse, n_fine)
            spatial_coords: Spatial coordinates (N, 2) for smoothness
            time_indices: Time indices (N,) for smoothness
            
        Returns:
            Dictionary with loss components
        """
        # 1. Prediction loss
        if self.prediction_loss == "mse":
            loss_pred = F.mse_loss(y_pred, y_true)
        elif self.prediction_loss == "mae":
            loss_pred = F.l1_loss(y_pred, y_true)
        else:
            raise ValueError(f"Unknown loss: {self.prediction_loss}")
        
        # 2. Coarse entropy (encourage diversity across ocean basins)
        # H = -Σ p_k log(p_k)
        eps = 1e-10
        H_coarse = -torch.sum(
            p_coarse * torch.log(p_coarse + eps),
            dim=1
        ).mean()
        
        # 3. Fine entropy (encourage diversity within each coarse regime)
        # H(fine | coarse=k) = -Σ p_j|k log(p_j|k)
        H_fine = -torch.sum(
            p_fine * torch.log(p_fine + eps),
            dim=2  # Sum over fine regimes
        ).mean()  # Average over samples and coarse regimes
        
        # 4. Hierarchy consistency
        # Fine regimes of different coarse parents should be distinct
        # Maximize KL divergence between p_fine[k1] and p_fine[k2]
        n_coarse = p_coarse.shape[1]
        consistency_loss = torch.tensor(0.0, device=y_pred.device)
        
        if self.consistency_weight > 0 and n_coarse > 1:
            # Average fine distributions for each coarse regime
            # p_fine_avg[k] = E[p_fine[:, k, :]]
            p_fine_avg = torch.mean(p_fine, dim=0)  # (n_coarse, n_fine)
            
            # Pairwise KL divergence
            kl_sum = 0.0
            n_pairs = 0
            for k1 in range(n_coarse):
                for k2 in range(k1 + 1, n_coarse):
                    # KL(p_k1 || p_k2)
                    kl = F.kl_div(
                        torch.log(p_fine_avg[k1] + eps),
                        p_fine_avg[k2],
                        reduction='sum',
                    )
                    kl_sum += kl
                    n_pairs += 1
            
            # Maximize divergence = minimize negative
            if n_pairs > 0:
                consistency_loss = -kl_sum / n_pairs
        
        # 5. Spatial smoothness (optional, on flattened joint probs)
        spatial_loss = torch.tensor(0.0, device=y_pred.device)
        if self.spatial_weight > 0 and spatial_coords is not None:
            # Flatten joint probabilities for smoothness
            p_joint = p_coarse.unsqueeze(2) * p_fine  # (N, n_coarse, n_fine)
            p_flat = p_joint.reshape(p_joint.shape[0], -1)  # (N, n_coarse*n_fine)
            spatial_loss = spatial_smoothness_loss(p_flat, spatial_coords)
        
        # 6. Temporal smoothness (optional)
        temporal_loss = torch.tensor(0.0, device=y_pred.device)
        if self.temporal_weight > 0 and time_indices is not None:
            p_joint = p_coarse.unsqueeze(2) * p_fine
            p_flat = p_joint.reshape(p_joint.shape[0], -1)
            temporal_loss = temporal_consistency_loss(p_flat, time_indices)
        
        # Total loss
        total = (
            loss_pred
            - self.coarse_entropy_weight * H_coarse  # Negative = maximize entropy
            - self.fine_entropy_weight * H_fine
            + self.consistency_weight * consistency_loss
            + self.spatial_weight * spatial_loss
            + self.temporal_weight * temporal_loss
        )
        
        return {
            "total": total,
            "prediction": loss_pred,
            "coarse_entropy":H_coarse,
            "fine_entropy": H_fine,
            "consistency": consistency_loss,
            "spatial": spatial_loss,
            "temporal": temporal_loss,
        }
