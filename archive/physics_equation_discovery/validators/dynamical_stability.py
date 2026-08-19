from typing import Any

import numpy as np


class LyapunovScreener:
    """
    Applies dynamical systems theory to stabilize discovered equations.
    Evaluates the approximate Jacobian matrix using empirical finite-differences
    to calculate dynamical stability penalties Omega_stab(f_k).
    """

    def __init__(self, epsilon: float = 1e-4, delta: float = 1.0):
        """
        Args:
            epsilon: Step size for finite differences
            delta: User-set stability threshold for maximum eigenvalue
        """
        self.epsilon = epsilon
        self.delta = delta

    def compute_jacobian(self, hypothesis: Any, x: np.ndarray) -> np.ndarray:
        """
        Computes the empirical Jacobian J_k = d f_k / d x.
        Args:
            hypothesis: The symbolic Hypothesis object
            x: (N, D) covariates matrix
        Returns:
            J: (N, D) Jacobian matrix where J[i, j] = df/dx_j at sample i
        """
        N, D = x.shape
        f_x = hypothesis.evaluate(x)

        # Ensure outputs are at least 1D
        if f_x is None or len(f_x) == 0:
            return np.zeros((N, D))

        J = np.zeros((N, D))

        for j in range(D):
            x_shifted = x.copy()
            x_shifted[:, j] += self.epsilon
            f_x_shifted = hypothesis.evaluate(x_shifted)

            # Finite difference approx
            derivative = (f_x_shifted - f_x) / self.epsilon
            J[:, j] = derivative

        return J

    def compute_stability_penalty(self, hypothesis: Any, x: np.ndarray) -> float:
        """
        Computes Omega_stab(f_k) = max(0, lambda_max(J_k^T J_k) - delta)
        This penalizes equations that imply unstable/exponentially growing
        dynamics under small perturbations.
        """
        if x is None or len(x) == 0:
            return 0.0

        try:
            J = self.compute_jacobian(hypothesis, x)

            # Normalize symmetric expansion matrix by sample count N
            # to keep scalar penalty comparable across batch splits.
            JT_J = (J.T @ J) / max(1, J.shape[0])

            # Extract dominant multiplier
            eigenvalues = np.linalg.eigvals(JT_J)

            # We strictly extract real parts because J^T J is symmetric PSD
            lambda_max = np.max(np.real(eigenvalues))

            penalty = max(0.0, float(lambda_max) - self.delta)
            return penalty

        # Blanket by necessity: this differentiates and evaluates an arbitrary
        # search-generated expression, then takes eigenvalues of the result.
        # sympy, numpy and linalg each raise their own families here, and an
        # expression whose stability cannot be established is treated as
        # unstable -- the conservative direction for a screening penalty.
        except Exception:  # noqa: BLE001
            return 100.0
