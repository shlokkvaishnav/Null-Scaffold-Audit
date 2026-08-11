"""Ranks candidate hypotheses by fit, validity and complexity.

This is the selection rule. Both arms of the null-scaffold audit are ranked by
it, so a change here changes what the audit reports -- which is why the
constraint rules arrive by injection rather than by import.

The scorer knows that violating a constraint is bad. It does not know what a
constraint is. A `ConstraintValidator` supplies that, and without one the
scorer records no violations: there is nothing to check against, which is not
the same as having checked and found nothing wrong. Callers that want violation
penalties must pass a validator, and every caller in this repository does.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from engine.plugin import ConstraintValidator


class HypothesisScorer:
    """Verifies candidates against injected constraints and scores them against data."""

    # Weight on constraint violations. Named for what it penalises -- the count
    # of broken rules, whatever field those rules came from.
    VIOLATION_PENALTY_WEIGHT = 10.0
    COMPLEXITY_PENALTY_WEIGHT = 0.01
    FAILURE_PENALTY = 1e3

    def __init__(
        self,
        config: dict | None = None,
        validator: ConstraintValidator | None = None,
    ):
        """
        Args:
            config: Configuration dictionary
            validator: Supplies the domain's constraint rules. Omitting it means
                no rules apply, not that the rules passed.
        """
        self.config = config or {}
        self.constraints_checker = validator

    def check(self, hypothesis: Any) -> dict:
        """Verify a hypothesis against the injected constraints.

        Returns:
            dict: Violation log {constraint_name: {violation_rate, details}}
        """
        if hypothesis is None:
            return {}

        equation = getattr(hypothesis, "equation", None)
        if equation is None:
            return {"missing_equation": {"violation_rate": 1.0, "details": "No equation"}}

        if self.constraints_checker is None:
            return {}

        return self.constraints_checker.check_constraints(equation)

    def score_hypothesis(self, hypothesis: Any, observation: dict | None = None) -> float:
        """Assign a scalar score from data fit, validity and complexity.

        Higher is better.
        """
        if hypothesis is None:
            return float("-inf")

        # Validity violations (hard penalty)
        violation_log = getattr(hypothesis, "violation_log", {}) or {}
        violation_penalty = len(violation_log)

        # Data fit (soft penalty)
        data_misfit = self._compute_data_misfit(hypothesis, observation)

        # Complexity penalty
        equation = getattr(hypothesis, "equation", "")
        complexity = len(str(equation))

        # Store components on hypothesis
        hypothesis.likelihood = -data_misfit
        hypothesis.complexity = complexity

        # Combined score
        hypothesis.score = (
            -data_misfit
            - self.VIOLATION_PENALTY_WEIGHT * violation_penalty
            - self.COMPLEXITY_PENALTY_WEIGHT * complexity
        )

        return hypothesis.score

    def _compute_data_misfit(self, hypothesis: Any, observation: dict | None) -> float:
        """Mean squared error between the hypothesis' predictions and the targets."""
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
            return float(np.mean(residual**2))

        # Blanket by necessity: hypothesis.evaluate runs an arbitrary
        # search-generated expression. A candidate that cannot be evaluated
        # scores as maximally unfit, which is what FAILURE_PENALTY encodes --
        # the same answer for every exception type it could raise.
        except Exception:  # noqa: BLE001
            return self.FAILURE_PENALTY

    def batch_verify(self, hypotheses: list, observation: dict | None = None) -> list:
        """Verify and score many hypotheses, returning only the valid ones."""
        valid = []
        for h in hypotheses:
            h.verify(self)
            self.score_hypothesis(h, observation)
            if h.valid:
                valid.append(h)
        return valid
