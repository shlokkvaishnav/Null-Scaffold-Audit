class VerificationModule:
    """
    Verifies proposed hypotheses against physics constraints and data support.
    This is the self-critique step - non-negotiable for GRAIL-V.
    """
    def __init__(self, config=None):
        self.config = config or {}
        self.constraints_checker = None
    
    def check(self, hypothesis):
        """
        Verify a hypothesis against physics constraints.
        
        Args:
            hypothesis: Hypothesis object
        
        Returns:
            dict: Violation log {constraint_name: {violation_rate, details}}
        """
        # Lazy load constraints
        if self.constraints_checker is None:
            from ..science.constraints import PhysicsConstraints
            self.constraints_checker = PhysicsConstraints()
        
        # Check hypothesis validity
        violation_log = self.constraints_checker.check_constraints(
            hypothesis.equation
        )
        
        return violation_log
    
    def score_hypothesis(self, hypothesis, observation=None):
        """
        Assigns a scalar score using data fit + physics + complexity.
        Higher score = better hypothesis.
        
        Args:
            hypothesis: Hypothesis object
            observation: Observation dict with features/targets
        
        Returns:
            float: Hypothesis score
        """
        import numpy as np
        
        # Physics violations (hard penalty)
        violation_penalty = len(hypothesis.violation_log)
        
        # Data fit proxy (soft)
        data_misfit = 0.0
        if observation is not None and "features" in observation and "targets" in observation:
            try:
                y_pred = hypothesis.evaluate(observation["features"])
                residual = observation["targets"] - y_pred
                data_misfit = np.mean(residual ** 2)
            except Exception:
                data_misfit = 1e3  # Catastrophic failure
        
        # Complexity penalty
        complexity = len(str(hypothesis.equation))
        
        # Store components
        hypothesis.likelihood = -data_misfit
        hypothesis.complexity = complexity
        
        # Combined score: data fit + physics + simplicity
        hypothesis.score = (
            -data_misfit
            - 10.0 * violation_penalty
            - 0.01 * complexity
        )
        
        return hypothesis.score
