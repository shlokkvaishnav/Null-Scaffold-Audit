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

class EquationBeliefState:
    """
    Maintains the soft variational distribution q_k(h) over symbolic 
    hypotheses for a specific regime k, along with its partition function Z_k.
    Transforms discrete memory banks into probabilistic trees.
    """
    def __init__(self, regime_id: int):
        self.regime_id = regime_id
        # Map of hypothesis string -> float probability (q_k(h))
        self.q_h: Dict[str, float] = {}
        # Unnormalized log probabilities (scores)
        self.h_scores: Dict[str, float] = {}
        # Regime-specific partition function
        self.Z_k: float = 0.0
        
    def update(self, hypotheses: List[Any], temperature: float = 1.0) -> Dict[str, float]:
        """
        Updates q_k(h) probabilities across the current active symbolic memory bank 
        based on external hypothesis scores.
        """
        if not hypotheses:
            self.q_h = {}
            self.Z_k = 0.0
            return {}
            
        # Collect scores strictly for this regime
        scores = []
        valid_hypotheses = []
        for h in hypotheses:
            if getattr(h, 'regime_id', None) == self.regime_id:
                valid_hypotheses.append(h)
                scores.append(getattr(h, 'score', 0.0))
                
        if not valid_hypotheses:
            return self.q_h
            
        scores = np.array(scores)
        
        # Apply numerical stability for the partition function Z_k computation
        scores_normalized = scores / temperature
        scores_normalized -= np.max(scores_normalized)
        
        exp_scores = np.exp(scores_normalized)
        self.Z_k = np.sum(exp_scores)
        
        # Calculate full q_k(h) distribution
        probs = exp_scores / (self.Z_k + 1e-12)
        
        # Map back to hypothesis equations cleanly
        self.q_h = {h.equation: float(p) for h, p in zip(valid_hypotheses, probs)}
        self.h_scores = {h.equation: float(s) for h, s in zip(valid_hypotheses, scores)}
        
        return self.q_h
        
    def get_probability(self, equation: str) -> float:
        """Retrieve probability of a specific symbolic tree in this regime."""
        return self.q_h.get(equation, 0.0)

class FactorGraphBeliefState:
    """
    Temporal Factor Graph for tracking joint (regime, equation) beliefs.
    Implements marginal belief propagation:
    pi_k(t) = (1/Z) * product_{f in F} psi_f(k, h_k) * pi_k(t-1)^gamma
    """
    def __init__(self, num_regimes: int, gamma: float = 0.9):
        """
        Args:
            num_regimes: Total distinct climate regimes modeled
            gamma: Temporal forgetting factor in [0, 1]. 
                   0 = memoryless instantaneous state tracking.
                   1 = infinite structural memory.
        """
        self.num_regimes = num_regimes
        self.gamma = max(0.0, min(1.0, gamma))
        self.pi = np.ones(num_regimes) / num_regimes
        self.history = []
        
    @property
    def beliefs(self) -> np.ndarray:
        return self.pi
        
    def update(self, hypotheses: List[Any], physics_constraints: Any) -> np.ndarray:
        """
        Perform Belief Propagation over regimes using explicit physical clique potentials
        multiplied by the prior retention factor.
        """
        # Calculate product of clique potentials per regime
        clique_products = np.zeros(self.num_regimes)
        
        for h in hypotheses:
            regime_id = getattr(h, 'regime_id', None)
            equation = getattr(h, 'equation', "")
            
            if regime_id is not None and 0 <= regime_id < self.num_regimes:
                # Extract structural clique potentials from physics engine
                p_cons = physics_constraints.psi_conservation(equation)
                p_thermo = physics_constraints.psi_thermo(equation)
                p_stab = physics_constraints.psi_stability(equation)
                
                # Product of all hard factor nodes f in F
                total_psi = p_cons * p_thermo * p_stab
                
                # Aggregate the strongest valid hypothesis potential for the regime
                clique_products[regime_id] = max(clique_products[regime_id], total_psi)
                
        # Safeguard against 0.0 collapsing regimes permanently
        clique_products = np.maximum(clique_products, 1e-12)
        
        # Temporal message passing from previous iteration
        prior_retention = np.power(self.pi, self.gamma)
        
        # Unnormalized updated node belief
        unnormalized_pi = clique_products * prior_retention
        
        # Calculate Partition Function Z
        Z = np.sum(unnormalized_pi)
        
        # Marginal normalized assignment probability
        self.pi = unnormalized_pi / (Z + 1e-12)
        
        self.history.append({
            "clique_products": clique_products.copy(),
            "beliefs": self.pi.copy()
        })
        
        return self.pi

class InformationGeometricBeliefState:
    """
    Tracks regime beliefs pi(t) as a smooth geodesic trajectory on the 
    statistical manifold of categorical distributions using the Fisher-Rao metric.
    
    Transforms abrupt Bayesian/Score updates into mathematically minimal Information Loss paths:
    pi(t+1) = pi(t)^(1-eta) * pi_target(t+1)^eta / Z
    """
    def __init__(self, num_regimes: int, eta: float = 0.5):
        """
        Args:
            num_regimes: Number of distinct climate regimes to track
            eta: Natural gradient step size parameter in [0, 1].
                 0.0 = pi(t) never evolves (permanently anchored to prior)
                 1.0 = pi(t) instantly teleports entirely to pi_target (memoryless)
                 0.5 = explicit geometric mean interpolation across the probability simplex
        """
        self.num_regimes = num_regimes
        self.eta = max(0.0, min(1.0, eta))
        # Initializing uniformly on the simplex interior
        self.pi = np.ones(num_regimes) / num_regimes
        self.history = []
        
    @property
    def beliefs(self) -> np.ndarray:
        return self.pi
        
    def update(self, pi_target: np.ndarray) -> np.ndarray:
        """
        Propagates the regime belief systematically along the Fisher-Rao geodesic.
        
        Args:
            pi_target: The idealized categorical target distribution pi_target(t+1)
                       derived from raw observations, Factor Graph clique products, or ELBO bounds.
                       Must be a valid probability distribution summing to 1.0.
        Returns:
            np.ndarray: Updated belief distribution pi(t+1)
        """
        # Ensure numerical stability bounding (avoid strict 0 bases which break logs/exponents)
        safe_pi_t = np.maximum(self.pi, 1e-12)
        safe_pi_target = np.maximum(pi_target, 1e-12)
        
        # Compute unnormalized geodesic metric interpolation step
        unnormalized_new_pi = np.power(safe_pi_t, 1.0 - self.eta) * np.power(safe_pi_target, self.eta)
        
        # Extract the continuous normalization Partition function Z
        Z = np.sum(unnormalized_new_pi)
        
        # Normalize to maintain the closed probability simplex
        self.pi = unnormalized_new_pi / (Z + 1e-12)
        
        self.history.append({
            "target": pi_target.copy(),
            "beliefs": self.pi.copy()
        })
        
        return self.pi
