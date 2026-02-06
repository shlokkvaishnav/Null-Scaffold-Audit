class Hypothesis:
    """
    First-class object representing a discovered symbolic equation for a regime.
    """
    def __init__(self, equation, regime_id):
        self.equation = equation
        self.regime_id = regime_id
        self.valid = True
        self.violation_log = {}
        self.score = None

    def verify(self, verifier):
        """
        Verify this hypothesis against physics constraints.
        """
        self.violation_log = verifier.check(self.equation)
        self.valid = not any(
            v.get("violation_rate", 0) > 0.1 
            for v in self.violation_log.values()
        )
        return self.valid

    def __repr__(self):
        status = "VALID" if self.valid else "INVALID"
        return f"<Hypothesis regime={self.regime_id} status={status} eq={self.equation}>"
