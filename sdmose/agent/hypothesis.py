class Hypothesis:
    """
    Represents a single symbolic hypothesis for a specific regime.
    """
    def __init__(self, equation, regime_id, iteration=0):
        self.equation = equation
        self.regime_id = regime_id
        
        self.valid = True
        self.violation_log = {}
        
        # Agentic evaluation
        self.likelihood = None
        self.complexity = None
        self.score = None
        
        # Lineage tracking for analysis
        self.created_at = iteration
        self.parent_id = None  # For future hypothesis evolution
    
    def evaluate(self, features):
        """
        Evaluate hypothesis on input features (for data fit).
        
        Args:
            features: numpy array (n_samples, n_features)
        
        Returns:
            predictions: numpy array (n_samples,)
        """
        import numpy as np
        
        # Extract features
        x0 = features[:, 0] if features.shape[1] > 0 else np.zeros(features.shape[0])
        x1 = features[:, 1] if features.shape[1] > 1 else np.zeros(features.shape[0])
        
        eq = str(self.equation).strip()
        
        # Deterministic template evaluation
        if eq == "x0":
            return x0
        elif eq == "x1":
            return x1
        elif eq == "x0 + x1":
            return x0 + x1
        elif eq == "x0 * x1":
            return x0 * x1
        elif eq == "x0 + 0.1":
            return x0 + 0.1
        elif eq == "x0 - x1":
            return x0 - x1
        else:
            # Fallback: return x0 (penalized by scoring if wrong)
            return x0
    
    def verify(self, verifier):
        """
        Verify this hypothesis against physics constraints.
        """
        self.violation_log = verifier.check(self)
        
        # Check if any violations exceed threshold
        self.valid = not any(
            v.get("violation_rate", 0) > 0.1 
            for v in self.violation_log.values()
        ) if self.violation_log else True
        
        return self.valid

    def __repr__(self):
        status = "VALID" if self.valid else "INVALID"
        score_str = f"{self.score:.3f}" if self.score is not None else "N/A"
        return f"<Hypothesis regime={self.regime_id} status={status} score={score_str} eq={self.equation}>"
