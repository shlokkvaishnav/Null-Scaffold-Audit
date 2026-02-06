"""
Belief State Module.
Maintains soft regime beliefs and uncertainty estimates.
"""

import numpy as np
from typing import List, Dict, Optional, Any


class BeliefState:
    """
    Maintains soft regime beliefs (pi) and uncertainty estimates.
    """
    
    # Constants
    EPSILON = 1e-12  # Numerical stability
    ENTROPY_FLOOR = 0.1  # Minimum entropy threshold
    REGULARIZATION_STRENGTH = 0.1  # Entropy regularization weight
    
    def __init__(self, num_regimes: int = 3):
        """
        Initialize belief state.
        
        Args:
            num_regimes: Number of regimes to track
        """
        if num_regimes < 1:
            raise ValueError(f"num_regimes must be >= 1, got {num_regimes}")
        
        self.num_regimes = num_regimes
        self.pi = np.ones(num_regimes) / num_regimes  # Uniform initialization
        self.prev_beliefs: Optional[np.ndarray] = None
        self.history: List[Dict] = []
    
    @property
    def beliefs(self) -> np.ndarray:
        """Alias for backward compatibility."""
        return self.pi
    
    def update(self, hypotheses: List, temperature: float = 1.0) -> np.ndarray:
        """
        Bayesian-style belief update using hypothesis scores.
        
        Args:
            hypotheses: List of scored Hypothesis objects
            temperature: Softmax temperature (higher = more uniform)
        
        Returns:
            Updated belief distribution
        """
        if temperature <= 0:
            raise ValueError(f"temperature must be > 0, got {temperature}")
        
        # Store previous beliefs for convergence check
        self.prev_beliefs = self.pi.copy()
        
        # Aggregate scores per regime
        scores = np.zeros(self.num_regimes)
        
        for h in hypotheses:
            regime_id = getattr(h, 'regime_id', None)
            score = getattr(h, 'score', None)
            
            if regime_id is not None and score is not None:
                if 0 <= regime_id < self.num_regimes:
                    scores[regime_id] += score
        
        # Softmax with numerical stability
        scores_normalized = scores / temperature
        scores_normalized -= np.max(scores_normalized)  # Prevent overflow
        exp_scores = np.exp(scores_normalized)
        pi = exp_scores / (exp_scores.sum() + self.EPSILON)
        
        # Entropy regularization (prevents collapse)
        entropy = self._compute_entropy(pi)
        if entropy < self.ENTROPY_FLOOR:
            uniform = np.ones(self.num_regimes) / self.num_regimes
            pi = (1 - self.REGULARIZATION_STRENGTH) * pi + self.REGULARIZATION_STRENGTH * uniform
        
        self.pi = pi
        
        # Record update for analysis
        self.history.append({
            "scores": scores.copy(),
            "beliefs": self.pi.copy(),
            "entropy": entropy
        })
        
        return self.pi
    
    def _compute_entropy(self, distribution: np.ndarray) -> float:
        """Compute Shannon entropy of a distribution."""
        # Filter out zeros to avoid log(0)
        p = distribution[distribution > self.EPSILON]
        return -np.sum(p * np.log(p))
    
    def get_entropy(self) -> float:
        """Get current belief entropy."""
        return self._compute_entropy(self.pi)
    
    def get_weights(self, regime_id: int) -> float:
        """
        Get belief weight for a specific regime.
        
        Args:
            regime_id: Regime index
            
        Returns:
            Belief weight (0.0 if invalid regime_id)
        """
        if 0 <= regime_id < self.num_regimes:
            return float(self.pi[regime_id])
        return 0.0
    
    def get_dominant_regime(self) -> int:
        """Get the regime with highest belief."""
        return int(np.argmax(self.pi))
    
    def is_converged(self, tol: float = 1e-3) -> bool:
        """
        Check if beliefs have converged.
        
        Args:
            tol: Convergence tolerance
        
        Returns:
            True if converged
        """
        if self.prev_beliefs is None:
            return False
        
        return float(np.linalg.norm(self.pi - self.prev_beliefs)) < tol
    
    def reset(self) -> None:
        """Reset to uniform beliefs."""
        self.pi = np.ones(self.num_regimes) / self.num_regimes
        self.prev_beliefs = None
        self.history = []
