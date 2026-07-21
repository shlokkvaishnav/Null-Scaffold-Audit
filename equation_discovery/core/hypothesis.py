"""
Hypothesis Module.
First-class symbolic hypothesis objects.
"""

import numpy as np
from typing import Dict, Optional, Any

from .expression_eval import safe_evaluate


class Hypothesis:
    """
    Represents a single symbolic hypothesis for a specific regime.
    """

    def __init__(self, equation: str, regime_id: int, iteration: int = 0):
        """
        Initialize a hypothesis.

        Args:
            equation: Symbolic equation string
            regime_id: Regime this hypothesis applies to
            iteration: Agent iteration when created
        """
        self.equation = str(equation)
        self.regime_id = regime_id

        # Validation state
        self.valid = True
        self.violation_log: Dict = {}

        # Evaluation scores
        self.likelihood: Optional[float] = None
        self.complexity: Optional[int] = None
        self.score: Optional[float] = None

        # Lineage tracking
        self.created_at = iteration
        self.parent_id: Optional[str] = None

    def evaluate(self, features: np.ndarray) -> np.ndarray:
        """
        Evaluate hypothesis on input features.

        Args:
            features: numpy array (n_samples, n_features)

        Returns:
            predictions: numpy array (n_samples,)
        """
        if features is None or len(features) == 0:
            return np.array([])

        # Ensure 2D
        if features.ndim == 1:
            features = features.reshape(-1, 1)

        n_samples, n_features = features.shape

        # Handles both ordinary infix notation (e.g. "x0 * x1", PySR/template
        # output) and gplearn's prefix function notation (e.g. "mul(X0, X1)"),
        # with any number of variables -- see core/expression_eval.py.
        result = safe_evaluate(self.equation, features)
        if result is not None and result.shape == (n_samples,):
            return result

        # Fallback: equation couldn't be parsed/evaluated -- return the first
        # feature column rather than raising, so a bad hypothesis degrades to
        # a poor (but harmless) score instead of crashing the agent loop.
        return features[:, 0] if n_features > 0 else np.zeros(n_samples)

    def verify(self, verifier) -> bool:
        """
        Verify this hypothesis against equation-validity constraints.

        Args:
            verifier: HypothesisScorer instance

        Returns:
            True if valid
        """
        self.violation_log = verifier.check(self) or {}

        # Check if any violations exceed threshold
        self.valid = not any(
            v.get("violation_rate", 0) > 0.1
            for v in self.violation_log.values()
        ) if self.violation_log else True

        return self.valid

    def get_score_components(self) -> Dict[str, Optional[float]]:
        """Get breakdown of score components."""
        return {
            "likelihood": self.likelihood,
            "complexity": self.complexity,
            "total_score": self.score
        }

    def __repr__(self) -> str:
        status = "VALID" if self.valid else "INVALID"
        score_str = f"{self.score:.3f}" if self.score is not None else "N/A"
        return f"<Hypothesis regime={self.regime_id} status={status} score={score_str} eq={self.equation}>"

    def __str__(self) -> str:
        return self.equation

    def __eq__(self, other) -> bool:
        if not isinstance(other, Hypothesis):
            return False
        return self.equation == other.equation and self.regime_id == other.regime_id

    def __hash__(self) -> int:
        return hash((self.equation, self.regime_id))

    def compute_composite_score(self, y_true: np.ndarray, y_pred: np.ndarray,
                                screener: Any, x: np.ndarray,
                                lambdas: Dict[str, float]) -> float:
        """
        Compute total evaluation score accounting for accuracy, validity,
        parsimony, and dynamic Lyapunov stability.

        score(h) = -MSE(y, y_hat) - lambda_v * violations - lambda_c * complexity - lambda_s * Omega_stab(f_k)
        """
        if y_true is None or y_pred is None or len(y_true) == 0:
            return 0.0

        # Ensure 1D arrays for MSE summation
        y_true_flat = np.asarray(y_true).ravel()
        y_pred_flat = np.asarray(y_pred).ravel()

        # 1. Mean Squared Error (Accuracy fit)
        self.mse = float(np.mean((y_true_flat - y_pred_flat)**2))

        # 2. Hard validity violations
        self.violations_penalty = 0.0
        if hasattr(self, 'violation_log') and self.violation_log:
            self.violations_penalty = sum(v.get("violation_rate", 0) for v in self.violation_log.values())

        # 3. Complexity (Parsimony proxy)
        self.complexity = float(self.complexity if self.complexity is not None else len(self.equation))

        # 4. Lyapunov Stability (Dynamical Stability Screening)
        self.stability_penalty = screener.compute_stability_penalty(self, x) if screener else 0.0

        # Fetch lambda weights
        l_v = lambdas.get("v", 10.0)
        l_c = lambdas.get("c", 0.01)
        l_s = lambdas.get("s", 1.0)

        # Final penalty subtraction
        self.score = -self.mse - (l_v * self.violations_penalty) - (l_c * self.complexity) - (l_s * self.stability_penalty)
        return self.score
