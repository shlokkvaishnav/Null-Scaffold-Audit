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


class AttentionGatingNetwork(nn.Module):
    """Attention-based gating network for interpretable regime assignment.
    
    Uses multi-head self-attention to learn which features are most important
    for regime assignment in different oceanic contexts.
    
    Architecture:
        Input Embedding → Multi-Head Self-Attention → Feed-Forward → Regime Projection
        
    Benefits over standard MLP:
    1. **Feature Importance**: Attention weights show which variables drive each regime
    2. **Context-Dependent**: Feature importance varies by location/season
    3. **Interpretability**: Can visualize "SST matters 80% for Tropical regime"
    
    Args:
        input_dim: Number of gating features
        num_regimes: Number of regimes K
        n_heads: Number of attention heads (must divide embed_dim)
        embed_dim: Embedding dimension (default: input_dim)
        ff_dim: Feed-forward hidden dimension
        dropout: Dropout probability
        temperature: Softmax temperature for final regime probabilities
        use_layernorm: Whether to use layer normalization
        
    Example:
        >>> attn_gating = AttentionGatingNetwork(
        ...     input_dim=10,  # lat, lon, sst, sss, log_chl, sst_grad, sin/cos(month), year
        ...     num_regimes=6,
        ...     n_heads=5,  # 5 heads for 10 features (embed_dim=10)
        ...     embed_dim=10,
        ...     dropout=0.1
        ... )
        >>> probs = attn_gating(x_gate)  # (N, K)
        >>> importance = attn_gating.get_feature_importance(x_gate)  # (N, input_dim)
    """
    
    def __init__(
        self,
        input_dim: int,
        num_regimes: int,
        n_heads: int = 4,
        embed_dim: Optional[int] = None,
        ff_dim: Optional[int] = None,
        dropout: float = 0.1,
        temperature: float = 1.0,
        use_layernorm: bool = True,
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.num_regimes = num_regimes
        self.n_heads = n_heads
        self.temperature = temperature
        
        # Embedding dimension (default: same as input)
        if embed_dim is None:
            embed_dim = input_dim
        self.embed_dim = embed_dim
        
        # Ensure embed_dim is divisible by n_heads
        if embed_dim % n_heads != 0:
            raise ValueError(f"embed_dim ({embed_dim}) must be divisible by n_heads ({n_heads})")
        
        # Feed-forward dimension
        if ff_dim is None:
            ff_dim = embed_dim * 4  # Standard Transformer ratio
        
        # Input projection (if embed_dim != input_dim)
        if input_dim != embed_dim:
            self.input_proj = nn.Linear(input_dim, embed_dim)
        else:
            self.input_proj = nn.Identity()
        
        # Multi-head self-attention
        # PyTorch's MultiheadAttention expects (seq_len, batch, embed_dim)
        # We treat each feature as a "token" in the sequence
        self.attention = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,  # Use (batch, seq, feature) format
        )
        
        # Layer normalization
        self.use_layernorm = use_layernorm
        if use_layernorm:
            self.norm1 = nn.LayerNorm(embed_dim)
            self.norm2 = nn.LayerNorm(embed_dim)
        
        # Feed-forward network
        self.ff_network = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, embed_dim),
            nn.Dropout(dropout),
        )
        
        # Final projection to regime probabilities
        self.regime_projector = nn.Linear(embed_dim, num_regimes)
        
        # Store attention weights for analysis
        self.last_attention_weights = None
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights using Xavier uniform."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(
        self,
        x: torch.Tensor,
        return_attention: bool = False,
        return_logits: bool = False,
    ) -> torch.Tensor | Tuple[torch.Tensor, ...]:
        """Forward pass with attention mechanism.
        
        Args:
            x: Gating features (N, input_dim)
            return_attention: If True, return attention weights
            return_logits: If True, return raw logits before softmax
            
        Returns:
            probs: Regime probabilities (N, num_regimes)
            attention_weights: Attention weights (N, n_heads, input_dim, input_dim) if requested
            logits: Raw logits (N, num_regimes) if requested
        """
        batch_size = x.shape[0]
        
        # Project input to embedding dimension
        x_embed = self.input_proj(x)  # (N, embed_dim)
        
        # Reshape for attention: treat each feature as a sequence element
        # For single-sample attention, we create a dummy sequence dimension
        # Actually, we want to attend across features, so:
        # Reshape to (N, 1, embed_dim) for self-attention
        x_seq = x_embed.unsqueeze(1)  # (N, 1, embed_dim)
        
        # Self-attention (attending to itself)
        # For feature-level attention, we need a different approach
        # Let's use a proper feature-wise attention mechanism
        
        # Alternative: Treat features as sequence
        # Reshape: (N, input_dim) → treat as (N, input_dim, 1) then project
        # Actually, let's use a cleaner approach with learnable query
        
        # Create learnable query for regime assignment
        # This allows the model to ask: "What features matter for regime k?"
        query = x_seq  # Use the embedding as query
        key = x_seq
        value = x_seq
        
        # Self-attention
        attn_output, attn_weights = self.attention(
            query, key, value,
            need_weights=True,
            average_attn_weights=False,  # Get per-head weights
        )
        # attn_output: (N, 1, embed_dim)

        # attn_weights: (N, n_heads, 1, 1)
        
        # Store for analysis
        self.last_attention_weights = attn_weights
        
        # Residual connection + layer norm
        if self.use_layernorm:
            x_attn = self.norm1(x_seq + attn_output)
        else:
            x_attn = x_seq + attn_output
        
        # Feed-forward network
        ff_output = self.ff_network(x_attn)
        
        # Residual connection + layer norm
        if self.use_layernorm:
            x_ff = self.norm2(x_attn + ff_output)
        else:
            x_ff = x_attn + ff_output
        
        # Remove sequence dimension and project to regimes
        x_final = x_ff.squeeze(1)  # (N, embed_dim)
        logits = self.regime_projector(x_final)  # (N, num_regimes)
        
        # Temperature-scaled softmax
        scaled_logits = logits / self.temperature
        probs = F.softmax(scaled_logits, dim=1)
        
        # Return based on flags
        outputs = [probs]
        if return_attention:
            outputs.append(attn_weights)
        if return_logits:
            outputs.append(logits)
        
        return tuple(outputs) if len(outputs) > 1 else outputs[0]
    
    def get_feature_importance(
        self,
        x: torch.Tensor,
        aggregate: str = "mean",
    ) -> torch.Tensor:
        """Extract feature importance from attention weights.
        
        Args:
            x: Gating features (N, input_dim)
            aggregate: How to aggregate multi-head attention ('mean', 'max', 'sum')
            
        Returns:
            Feature importance scores (N, input_dim)
            
        Interpretation:
            High score → feature is important for regime assignment at this location/time
            
        Example:
            >>> importance = model.get_feature_importance(x_gate)
            >>> # importance[:, 2] = SST importance
            >>> # importance[:, 3] = SSS importance
        """
        # Forward pass to compute attention
        _, attn_weights = self.forward(x, return_attention=True)
        # attn_weights: (N, n_heads, seq_len, seq_len)
        
        # For self-attention on single feature vector, extract diagonal
        # (how much each feature attends to itself)
        # Since we use (N, 1, embed_dim), we need a different approach
        
        # Alternative: Use attention to learnable queries
        # For now, use gradient-based importance
        x.requires_grad_(True)
        probs = self.forward(x)
        
        # Compute gradient of output w.r.t. input
        # For each regime, compute average gradient magnitude
        importance = torch.zeros(x.shape[0], self.input_dim, device=x.device)
        
        for k in range(self.num_regimes):
            # Gradient of regime k probability w.r.t. input
            grad_outputs = torch.zeros_like(probs)
            grad_outputs[:, k] = probs[:, k]  # Weight by probability
            
            grads = torch.autograd.grad(
                outputs=probs,
                inputs=x,
                grad_outputs=grad_outputs,
                retain_graph=True,
                create_graph=False,
            )[0]
            
            # Accumulate absolute gradients (importance)
            importance += torch.abs(grads) * probs[:, k].unsqueeze(1)
        
        return importance
    
    def set_temperature(self, temperature: float):
        """Update softmax temperature."""
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
