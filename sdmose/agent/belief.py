import numpy as np

class BeliefState:
    """
    Maintains soft regime beliefs (pi) and uncertainty estimates.
    """
    def __init__(self, num_regimes=3):
        self.num_regimes = num_regimes
        # Uniform initialization
        self.pi = np.ones(num_regimes) / num_regimes
        self.beliefs = self.pi  # Alias for backward compatibility
        self.prev_beliefs = None
        self.history = []
    
    def update(self, hypotheses, temperature=1.0):
        """
        Bayesian-style belief update using hypothesis scores.
        
        Args:
            hypotheses: List of scored Hypothesis objects
            temperature: Softmax temperature (higher = more uniform)
        """
        # Store previous beliefs for convergence check
        self.prev_beliefs = self.pi.copy()
        
        # Aggregate scores per regime
        scores = np.zeros(self.num_regimes)
        
        for h in hypotheses:
            if h.score is not None:
                scores[h.regime_id] += h.score
        
        # Softmax update for probabilistic belief revision
        exp_scores = np.exp(scores / temperature)
        self.pi = exp_scores / (exp_scores.sum() + 1e-12)
        self.beliefs = self.pi  # Update alias
        
        # Record update
        self.history.append({
            "scores": scores.copy(),
            "beliefs": self.pi.copy()
        })
        
        return self.pi
        
    def get_weights(self, regime_id):
        """
        Get belief weight for a specific regime.
        """
        return self.pi[regime_id] if regime_id < self.num_regimes else 0.0
    
    def is_converged(self, tol=1e-3):
        """
        Check if beliefs have converged.
        
        Args:
            tol: Convergence tolerance
        
        Returns:
            bool: True if converged
        """
        if self.prev_beliefs is None:
            return False
        
        return np.linalg.norm(self.pi - self.prev_beliefs) < tol
