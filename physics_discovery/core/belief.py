"""
Confidence Tracking Module.
Maintains soft regime confidence distributions and uncertainty estimates.
"""

import numpy as np
from typing import List, Dict, Optional, Any


class ConfidenceTracker:
    """
    Maintains a soft confidence distribution (pi) over regimes, updated by
    a softmax over accumulated hypothesis scores (a Bayesian-style
    exponentiated-score update), with an entropy floor to prevent premature
    collapse onto a single regime.
    """

    # Constants
    EPSILON = 1e-12  # Numerical stability
    ENTROPY_FLOOR = 0.1  # Minimum entropy threshold
    REGULARIZATION_STRENGTH = 0.1  # Entropy regularization weight

    def __init__(self, num_regimes: int = 3):
        if num_regimes < 1:
            raise ValueError(f"num_regimes must be >= 1, got {num_regimes}")

        self.num_regimes = num_regimes
        self.pi = np.ones(num_regimes) / num_regimes
        self.prev_beliefs: Optional[np.ndarray] = None
        self.history: List[Dict] = []

    @property
    def beliefs(self) -> np.ndarray:
        return self.pi

    def update(self, hypotheses: List, temperature: float = 1.0) -> np.ndarray:
        if temperature <= 0:
            raise ValueError(f"temperature must be > 0, got {temperature}")

        self.prev_beliefs = self.pi.copy()
        scores = np.zeros(self.num_regimes)

        for h in hypotheses:
            regime_id = getattr(h, "regime_id", None)
            score = getattr(h, "score", None)
            if regime_id is not None and score is not None and 0 <= regime_id < self.num_regimes:
                scores[regime_id] += score

        scores_normalized = scores / temperature
        scores_normalized -= np.max(scores_normalized)
        exp_scores = np.exp(scores_normalized)
        pi = exp_scores / (exp_scores.sum() + self.EPSILON)

        entropy = self._compute_entropy(pi)
        if entropy < self.ENTROPY_FLOOR:
            uniform = np.ones(self.num_regimes) / self.num_regimes
            pi = (1 - self.REGULARIZATION_STRENGTH) * pi + self.REGULARIZATION_STRENGTH * uniform

        self.pi = pi
        self.history.append({"scores": scores.copy(), "beliefs": self.pi.copy(), "entropy": entropy})
        return self.pi

    def _compute_entropy(self, distribution: np.ndarray) -> float:
        p = distribution[distribution > self.EPSILON]
        return -np.sum(p * np.log(p))

    def get_entropy(self) -> float:
        return self._compute_entropy(self.pi)

    def get_weights(self, regime_id: int) -> float:
        if 0 <= regime_id < self.num_regimes:
            return float(self.pi[regime_id])
        return 0.0

    def get_dominant_regime(self) -> int:
        return int(np.argmax(self.pi))

    def is_converged(self, tol: float = 1e-3) -> bool:
        if self.prev_beliefs is None:
            return False
        return float(np.linalg.norm(self.pi - self.prev_beliefs)) < tol

    def reset(self) -> None:
        self.pi = np.ones(self.num_regimes) / self.num_regimes
        self.prev_beliefs = None
        self.history = []


class EquationConfidenceTracker:
    """Maintains the soft variational distribution q_k(h) over candidate equations for a given regime."""

    def __init__(self, regime_id: int):
        self.regime_id = regime_id
        self.q_h: Dict[str, float] = {}
        self.h_scores: Dict[str, float] = {}
        self.Z_k: float = 0.0

    def update(self, hypotheses: List[Any], temperature: float = 1.0) -> Dict[str, float]:
        if not hypotheses:
            self.q_h = {}
            self.Z_k = 0.0
            return {}

        scores = []
        valid_hypotheses = []
        for h in hypotheses:
            if getattr(h, "regime_id", None) == self.regime_id:
                valid_hypotheses.append(h)
                scores.append(getattr(h, "score", 0.0))

        if not valid_hypotheses:
            return self.q_h

        scores = np.array(scores)
        scores_normalized = scores / temperature
        scores_normalized -= np.max(scores_normalized)

        exp_scores = np.exp(scores_normalized)
        self.Z_k = np.sum(exp_scores)
        probs = exp_scores / (self.Z_k + 1e-12)

        self.q_h = {h.equation: float(p) for h, p in zip(valid_hypotheses, probs)}
        self.h_scores = {h.equation: float(s) for h, s in zip(valid_hypotheses, scores)}
        return self.q_h

    def get_probability(self, equation: str) -> float:
        return self.q_h.get(equation, 0.0)


class FactorGraphConfidenceUpdater:
    """
    Temporal factor-graph confidence propagation over (regime, equation) structure.

    Combines a per-regime validity score (a clique potential, provided by the
    caller) with an exponentially-decayed prior over previous regime
    confidence to produce the updated distribution.
    """

    def __init__(self, num_regimes: int, gamma: float = 0.9):
        self.num_regimes = num_regimes
        self.gamma = max(0.0, min(1.0, gamma))
        self.pi = np.ones(num_regimes) / num_regimes
        self.history = []

    @property
    def beliefs(self) -> np.ndarray:
        return self.pi

    def update(self, hypotheses: List[Any], validity_checker: Any) -> np.ndarray:
        """
        Args:
            hypotheses: candidate hypotheses being scored this round
            validity_checker: object exposing per-equation clique-potential
                methods used to compute a combined validity score in [0, 1]
        """
        clique_products = np.zeros(self.num_regimes)

        for h in hypotheses:
            regime_id = getattr(h, "regime_id", None)
            equation = getattr(h, "equation", "")
            if regime_id is not None and 0 <= regime_id < self.num_regimes:
                potential = validity_checker.loss_penalty(equation)
                # loss_penalty is a violation penalty (>=0); convert to a
                # potential in (0, 1], where 0 violation -> potential of 1.
                p = 1.0 / (1.0 + potential)
                clique_products[regime_id] = max(clique_products[regime_id], p)

        clique_products = np.maximum(clique_products, 1e-12)
        prior_retention = np.power(self.pi, self.gamma)
        unnormalized_pi = clique_products * prior_retention
        self.pi = unnormalized_pi / (np.sum(unnormalized_pi) + 1e-12)

        self.history.append({"clique_products": clique_products.copy(), "beliefs": self.pi.copy()})
        return self.pi


class GeodesicConfidenceUpdater:
    """
    Geodesic/KL-regularized confidence update.

    A KL-regularized Bayesian update implemented as geodesic interpolation
    on the probability simplex:

        pi_{t+1}(k) proportional to pi_t(k)^(1-eta) * pi_target(k)^eta

    This update is guaranteed to stay in the simplex interior and yields a
    bounded interpolation trajectory for eta in [0, 1] between the previous
    belief and a target distribution.
    """

    EPSILON = 1e-12

    def __init__(self, num_regimes: int, eta: float = 0.5):
        self.num_regimes = num_regimes
        self.eta = max(0.0, min(1.0, eta))
        self.pi = np.ones(num_regimes) / num_regimes
        self.history = []

    @property
    def beliefs(self) -> np.ndarray:
        return self.pi

    @staticmethod
    def kl_divergence(p: np.ndarray, q: np.ndarray) -> float:
        """KL(p || q) with numerical clipping for stability."""
        p_safe = np.maximum(np.asarray(p, dtype=float), GeodesicConfidenceUpdater.EPSILON)
        q_safe = np.maximum(np.asarray(q, dtype=float), GeodesicConfidenceUpdater.EPSILON)
        p_safe = p_safe / p_safe.sum()
        q_safe = q_safe / q_safe.sum()
        return float(np.sum(p_safe * np.log(p_safe / q_safe)))

    def update(self, pi_target: np.ndarray) -> np.ndarray:
        pi_target = np.asarray(pi_target, dtype=float)
        if pi_target.shape[0] != self.num_regimes:
            raise ValueError(
                f"pi_target must have shape ({self.num_regimes},), got {pi_target.shape}"
            )

        safe_pi_t = np.maximum(self.pi, self.EPSILON)
        safe_pi_target = np.maximum(pi_target, self.EPSILON)

        unnormalized_new_pi = np.power(safe_pi_t, 1.0 - self.eta) * np.power(safe_pi_target, self.eta)
        self.pi = unnormalized_new_pi / (np.sum(unnormalized_new_pi) + self.EPSILON)

        self.history.append(
            {
                "target": (safe_pi_target / safe_pi_target.sum()).copy(),
                "beliefs": self.pi.copy(),
                "kl_to_target": self.kl_divergence(self.pi, safe_pi_target),
            }
        )
        return self.pi
