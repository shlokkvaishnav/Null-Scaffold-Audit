"""
Hypothesis Scorer Module.
Validates hypotheses against equation-validity constraints and data support.
"""

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    # The runtime import stays inside check(), which is where the validator is
    # first needed; naming the type here costs nothing at import time.
    from ..validation.equation_validity import EquationValidator


class HypothesisScorer:
    """
    Verifies proposed hypotheses against equation-validity constraints and data support.
    Critical self-verification step before a hypothesis is trusted.
    """

    # Scoring weights
    PHYSICS_PENALTY_WEIGHT = 10.0
    COMPLEXITY_PENALTY_WEIGHT = 0.01
    FAILURE_PENALTY = 1e3

    def __init__(self, config: dict | None = None):
        """
        Initialize the hypothesis scorer.

        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self.constraints_checker: EquationValidator | None = None

    def check(self, hypothesis) -> dict:
        """
        Verify a hypothesis against equation-validity constraints.

        Args:
            hypothesis: Hypothesis object

        Returns:
            dict: Violation log {constraint_name: {violation_rate, details}}
        """
        if hypothesis is None:
            return {}

        # Lazy load validity checker
        if self.constraints_checker is None:
            from ..validation.equation_validity import EquationValidator
            self.constraints_checker = EquationValidator()

        # Get equation string safely
        equation = getattr(hypothesis, 'equation', None)
        if equation is None:
            return {"missing_equation": {"violation_rate": 1.0, "details": "No equation"}}

        return self.constraints_checker.check_constraints(equation)

    def score_hypothesis(self, hypothesis, observation: dict | None = None) -> float:
        """
        Assigns a scalar score using data fit + validity + complexity.
        Higher score = better hypothesis.

        Args:
            hypothesis: Hypothesis object
            observation: Observation dict with features/targets

        Returns:
            float: Hypothesis score
        """
        if hypothesis is None:
            return float('-inf')

        # Validity violations (hard penalty)
        violation_log = getattr(hypothesis, 'violation_log', {}) or {}
        violation_penalty = len(violation_log)

        # Data fit (soft penalty)
        data_misfit = self._compute_data_misfit(hypothesis, observation)

        # Complexity penalty
        equation = getattr(hypothesis, 'equation', '')
        complexity = len(str(equation))

        # Store components on hypothesis
        hypothesis.likelihood = -data_misfit
        hypothesis.complexity = complexity

        # Combined score
        hypothesis.score = (
            -data_misfit
            - self.PHYSICS_PENALTY_WEIGHT * violation_penalty
            - self.COMPLEXITY_PENALTY_WEIGHT * complexity
        )

        return hypothesis.score

    def _compute_data_misfit(self, hypothesis, observation: dict | None) -> float:
        """
        Compute MSE between hypothesis predictions and targets.

        Args:
            hypothesis: Hypothesis object
            observation: Observation dict

        Returns:
            float: Mean squared error
        """
        if observation is None:
            return 0.0

        features = observation.get("features")
        targets = observation.get("targets")

        if features is None or targets is None:
            return 0.0

        # Ensure arrays
        if not isinstance(features, np.ndarray):
            features = np.array(features)
        if not isinstance(targets, np.ndarray):
            targets = np.array(targets)

        if len(features) == 0 or len(targets) == 0:
            return 0.0

        try:
            y_pred = hypothesis.evaluate(features)

            # Handle shape mismatches
            if y_pred is None or len(y_pred) == 0:
                return self.FAILURE_PENALTY

            if len(y_pred) != len(targets):
                return self.FAILURE_PENALTY

            residual = targets - y_pred
            return float(np.mean(residual ** 2))

        # Blanket by necessity: hypothesis.evaluate runs an arbitrary
        # search-generated expression. A candidate that cannot be evaluated
        # scores as maximally unfit, which is what FAILURE_PENALTY encodes --
        # the same answer for every exception type it could raise.
        except Exception:  # noqa: BLE001
            return self.FAILURE_PENALTY

    def batch_verify(self, hypotheses: list, observation: dict | None = None) -> list:
        """
        Verify and score multiple hypotheses.

        Args:
            hypotheses: List of Hypothesis objects
            observation: Observation dict

        Returns:
            list: Valid hypotheses only
        """
        valid = []
        for h in hypotheses:
            h.verify(self)
            self.score_hypothesis(h, observation)
            if h.valid:
                valid.append(h)
        return valid
