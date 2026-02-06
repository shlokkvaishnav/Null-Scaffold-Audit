import numpy as np

class BeliefState:
    """
    Maintains soft regime beliefs (pi) and uncertainty estimates.
    """
    def __init__(self, num_regimes=3):
        self.num_regimes = num_regimes
        # Uniform initialization
        self.beliefs = np.ones(num_regimes) / num_regimes
        self.prev_beliefs = None
        self.history = []
    
    def update(self, evidence):
        """
        Update beliefs based on new evidence.
        
        Args:
            evidence: dict with hypothesis performance metrics
        """
        # Store previous beliefs for convergence check
        self.prev_beliefs = self.beliefs.copy()
        
        # Placeholder: simple uniform update
        # TODO: Implement actual EM update logic
        self.history.append(evidence)
        
    def get_weights(self, regime_id):
        """
        Get belief weight for a specific regime.
        """
        return self.beliefs[regime_id] if regime_id < self.num_regimes else 0.0
    
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
        
        return np.linalg.norm(self.beliefs - self.prev_beliefs) < tol
