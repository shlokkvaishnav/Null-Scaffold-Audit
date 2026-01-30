"""Hierarchical SD-MoSE: Multi-scale regime discovery for ocean carbon.

Implements nested regime structure:
    Level 1 (Coarse): Ocean basins / climate zones (geography-driven)
    Level 2 (Fine): Physical/biological processes within zones (state-driven)

Example hierarchy:
    Tropical → {Equatorial Upwelling, Oligotrophic Gyre, Coastal}
    Mid-Latitude → {Frontal Zones, Mixed Layer, Biological Bloom}
    Polar → {Ice-Covered, Seasonal Ice, Deep Convection}

Scientific rationale:
    - Ocean processes operate at multiple scales
    - Geography determines baseline conditions (coarse)
    - Local physics/biology modulate response (fine)
    - Hierarchical structure improves interpretability and accuracy
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .gating import GatingNetwork, AttentionGatingNetwork
from .mixture import SDMoSE
from .symbolic import MixtureOfSymbolicExperts


class HierarchicalGatingNetwork(nn.Module):
    """Two-level hierarchical gating: coarse (geography) → fine (processes).
    
    Architecture:
        Level 1: P(coarse | x) - broad regime assignment (e.g., ocean basins)
        Level 2: P(fine | coarse, x) - process identification within coarse regime
        Joint: P(coarse, fine | x) = P(coarse | x) * P(fine | coarse, x)
    
    Args:
        input_dim: Number of gating features
        n_coarse: Number of coarse regimes (Level 1)
        n_fine_per_coarse: Number of fine regimes per coarse regime (Level 2)
        coarse_features: Indices of features for coarse gating (spatial/temporal)
        fine_features: Indices of features for fine gating (physical/biological)
        gating_type: "mlp" or "attention"
        dropout: Dropout probability
        temperature: Softmax temperature
        
    Example:
        >>> gating = HierarchicalGatingNetwork(
        ...     input_dim=10,
        ...     n_coarse=3,  # Tropical, Mid-Lat, Polar
        ...     n_fine_per_coarse=3,  # 3 processes each
        ...     coarse_features=[0,1,6,7,8],  # lat, lon, sin/cos(month), year
        ...     fine_features=[2,3,4,5],  # sst, sss, log_chl, sst_grad
        ... )
        >>> p_coarse, p_fine, p_joint = gating(X)
        >>> # p_coarse: (N, 3)
        >>> # p_fine: (N, 3, 3)
        >>> # p_joint: (N, 3, 3) - flattens to (N, 9) for experts
    """
    
    def __init__(
        self,
        input_dim: int,
        n_coarse: int = 3,
        n_fine_per_coarse: int = 3,
        coarse_features: Optional[List[int]] = None,
        fine_features: Optional[List[int]] = None,
        gating_type: str = "mlp",
        dropout: float = 0.1,
        temperature: float = 1.0,
        use_attention_fine: bool = False,
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.n_coarse = n_coarse
        self.n_fine_per_coarse = n_fine_per_coarse
        self.total_regimes = n_coarse * n_fine_per_coarse
        self.temperature = temperature
        
        # Feature indices for each level
        if coarse_features is None:
            # Default: spatial + temporal for coarse
            coarse_features = [0, 1, 6, 7, 8]  # lat, lon, sin/cos(month), year
        if fine_features is None:
            # Default: physical + biological for fine
            fine_features = [2, 3, 4, 5]  # sst, sss, log_chl, sst_gradient
        
        self.coarse_features = coarse_features
        self.fine_features = fine_features
        
        # Level 1: Coarse gating (geography-driven)
        coarse_input_dim = len(coarse_features)
        if gating_type == "attention":
            self.coarse_gating = AttentionGatingNetwork(
                input_dim=coarse_input_dim,
                num_regimes=n_coarse,
                n_heads=min(5, coarse_input_dim),
                dropout=dropout,
                temperature=temperature,
            )
        else:
            self.coarse_gating = GatingNetwork(
                input_dim=coarse_input_dim,
                num_regimes=n_coarse,
                hidden_dims=[64, 32],
                dropout=dropout,
                temperature=temperature,
            )
        
        # Level 2: Fine gating for each coarse regime (process-driven)
        # Condition on both fine features + coarse probabilities
        fine_input_dim = len(fine_features) + n_coarse
        
        self.fine_gating = nn.ModuleList()
        for k in range(n_coarse):
            if use_attention_fine and fine_input_dim % 2 == 0:
                gating_k = AttentionGatingNetwork(
                    input_dim=fine_input_dim,
                    num_regimes=n_fine_per_coarse,
                    n_heads=min(2, fine_input_dim // 2),
                    dropout=dropout,
                    temperature=temperature,
                )
            else:
                gating_k = GatingNetwork(
                    input_dim=fine_input_dim,
                    num_regimes=n_fine_per_coarse,
                    hidden_dims=[32, 16],
                    dropout=dropout,
                    temperature=temperature,
                )
            self.fine_gating.append(gating_k)
    
    def forward(
        self,
        x: torch.Tensor,
        return_all: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass through hierarchical gating.
        
        Args:
            x: Full feature vector (N, input_dim)
            return_all: If True, return intermediate outputs
            
        Returns:
            p_coarse: Coarse regime probabilities (N, n_coarse)
            p_fine: Fine regime probabilities (N, n_coarse, n_fine_per_coarse)
            p_joint: Joint probabilities (N, n_coarse, n_fine_per_coarse)
        """
        batch_size = x.shape[0]
        
        # Extract features for coarse gating
        x_coarse = x[:, self.coarse_features]
        
        # Level 1: Coarse regime assignment
        p_coarse = self.coarse_gating(x_coarse)  # (N, n_coarse)
        
        # Extract features for fine gating
        x_fine = x[:, self.fine_features]
        
        # Level 2: Fine regime assignment (conditioned on coarse)
        p_fine_list = []
        for k in range(self.n_coarse):
            # Concatenate: fine features + coarse probabilities (conditioning)
            x_cond = torch.cat([x_fine, p_coarse], dim=1)  # (N, fine_dim + n_coarse)
            
            # Fine gating for coarse regime k
            p_fine_k = self.fine_gating[k](x_cond)  # (N, n_fine_per_coarse)
            p_fine_list.append(p_fine_k)
        
        # Stack: (N, n_coarse, n_fine_per_coarse)
        p_fine = torch.stack(p_fine_list, dim=1)
        
        # Joint probability: P(k, j) = P(k) * P(j|k)
        p_joint = p_coarse.unsqueeze(2) * p_fine  # (N, n_coarse, n_fine_per_coarse)
        
        return p_coarse, p_fine, p_joint
    
    def get_flat_probabilities(self, x: torch.Tensor) -> torch.Tensor:
        """Get flattened joint probabilities for expert mixture.
        
        Args:
            x: Features (N, input_dim)
            
        Returns:
            Flattened probabilities (N, n_coarse * n_fine_per_coarse)
        """
        _, _, p_joint = self.forward(x)
        return p_joint.reshape(x.shape[0], -1)
    
    def get_regime_labels(
        self,
        x: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get dominant coarse and fine regime labels.
        
        Args:
            x: Features (N, input_dim)
            
        Returns:
            coarse_labels: Dominant coarse regime (N,)
            fine_labels: Dominant fine regime (N,) - flattened index
        """
        p_coarse, _, p_joint = self.forward(x)
        
        # Coarse: argmax over first dimension
        coarse_labels = torch.argmax(p_coarse, dim=1)  # (N,)
        
        # Fine: argmax over flattened joint
        p_flat = p_joint.reshape(x.shape[0], -1)
        fine_labels = torch.argmax(p_flat, dim=1)  # (N,)
        
        return coarse_labels, fine_labels
    
    def set_temperature(self, temperature: float):
        """Update temperature for all gating networks."""
        self.temperature = temperature
        self.coarse_gating.set_temperature(temperature)
        for fine_gate in self.fine_gating:
            fine_gate.set_temperature(temperature)


class HierarchicalSDMoSE(nn.Module):
    """Hierarchical SD-MoSE with nested regime structure.
    
    Combines hierarchical gating with symbolic experts at leaf nodes.
    
    Args:
        gating_network: HierarchicalGatingNetwork instance
        num_coarse: Number of coarse regimes
        num_fine_per_coarse: Number of fine regimes per coarse
        expert_features: Feature names for symbolic experts
        device: Device for computation
        
    Example:
        >>> gating = HierarchicalGatingNetwork(input_dim=10, n_coarse=3, n_fine_per_coarse=3)
        >>> model = HierarchicalSDMoSE(
        ...     gating_network=gating,
        ...     num_coarse=3,
        ...     num_fine_per_coarse=3,
        ...     expert_features=['sst', 'sss', 'log_chl', 'sst_gradient'],
        ... )
        >>> y_pred, p_coarse, p_fine = model(X_expert, X_gate)
    """
    
    def __init__(
        self,
        gating_network: HierarchicalGatingNetwork,
        num_coarse: int,
        num_fine_per_coarse: int,
        expert_features: List[str],
        device: str = "cpu",
    ):
        super().__init__()
        
        self.gating_network = gating_network
        self.num_coarse = num_coarse
        self.num_fine_per_coarse = num_fine_per_coarse
        self.total_regimes = num_coarse * num_fine_per_coarse
        self.expert_features = expert_features
        self.device = device
        
        # Symbolic experts at leaf level (fine regimes)
        self.experts = None  # Initialized during training
        
        # Store expert predictions (pre-computed for alternating optimization)
        self.expert_predictions = None
    
    def forward(
        self,
        X_expert: torch.Tensor,
        X_gate: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass through hierarchical model.
        
        Args:
            X_expert: Features for experts (N, D_expert)
            X_gate: Features for gating (N, D_gate)
            
        Returns:
            y_pred: Mixture prediction (N,)
            p_coarse: Coarse probabilities (N, n_coarse)
            p_fine: Fine probabilities (N, n_coarse, n_fine_per_coarse)
        """
        # Hierarchical gating
        p_coarse, p_fine, p_joint = self.gating_network(X_gate)
        
        # Flatten for expert mixture
        p_flat = p_joint.reshape(X_gate.shape[0], -1)  # (N, total_regimes)
        
        # Mixture prediction
        if self.expert_predictions is not None:
            # Use pre-computed expert predictions (during gating training)
            y_pred = torch.sum(p_flat * self.expert_predictions, dim=1)
        else:
            # Direct prediction (requires experts)
            raise ValueError("Expert predictions not set. Call set_expert_predictions() first.")
        
        return y_pred, p_coarse, p_fine
    
    def forward_mixture(
        self,
        X_gate: torch.Tensor,
        expert_preds: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward with pre-computed expert predictions (for training).
        
        Args:
            X_gate: Gating features (N, D_gate)
            expert_preds: Expert predictions (N, total_regimes)
            
        Returns:
            y_pred: Mixture prediction (N,)
            p_flat: Flattened regime probabilities (N, total_regimes)
        """
        p_flat = self.gating_network.get_flat_probabilities(X_gate)
        y_pred = torch.sum(p_flat * expert_preds, dim=1)
        return y_pred, p_flat
    
    def set_expert_predictions(self, expert_preds: torch.Tensor):
        """Set pre-computed expert predictions."""
        self.expert_predictions = expert_preds.to(self.device)
    
    def get_regime_interpretation(self) -> Dict:
        """Get human-readable regime names and hierarchy.
        
        Returns:
            Dictionary with regime hierarchy and names
        """
        # Default names (can be customized based on discovered patterns)
        coarse_names = {
            0: "Tropical",
            1: "Mid-Latitude",
            2: "Polar",
        }
        
        fine_names = {
            (0, 0): "Equatorial Upwelling",
            (0, 1): "Oligotrophic Gyre",
            (0, 2): "Coastal Tropical",
            (1, 0): "Frontal Zones",
            (1, 1): "Mixed Layer",
            (1, 2): "Biological Bloom",
            (2, 0): "Ice-Covered",
            (2, 1): "Seasonal Ice",
            (2, 2): "Deep Convection",
        }
        
        hierarchy = {}
        for k_coarse in range(self.num_coarse):
            coarse_name = coarse_names.get(k_coarse, f"Coarse {k_coarse}")
            children = []
            for k_fine in range(self.num_fine_per_coarse):
                fine_name = fine_names.get((k_coarse, k_fine), f"Fine {k_coarse}.{k_fine}")
                flat_idx = k_coarse * self.num_fine_per_coarse + k_fine
                children.append({
                    "name": fine_name,
                    "idx": flat_idx,
                })
            hierarchy[k_coarse] = {
                "name": coarse_name,
                "children": children,
            }
        
        return hierarchy
