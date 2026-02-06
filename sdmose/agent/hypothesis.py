class Hypothesis:
    """
    Represents a single symbolic hypothesis for a specific regime.
    """
    def __init__(self, equation, regime_id):
        self.equation = equation
        self.regime_id = regime_id
        
        self.valid = True
        self.violation_log = {}
        
        # Agentic evaluation (NEW)
        self.likelihood = None
        self.complexity = None
        self.score = None
    
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
