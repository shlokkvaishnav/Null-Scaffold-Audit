"""
Loop Control Module.
Determines convergence and reports progress for the discovery agent's main loop.

This replaces the previously-empty `agent/control.py` stub with a real
implementation: the convergence-check and progress-reporting logic that used
to live inline inside the agent's `run_loop` method.
"""

import numpy as np


class ConvergenceController:
    """
    Tracks belief-state and hypothesis-set history across agent iterations
    and decides when the discovery loop has converged.

    Convergence criterion: the regime confidence distribution has stabilized
    (L2 change below tolerance) AND the set of equations currently held in
    the archive is unchanged from the previous iteration.
    """

    def __init__(self, tol: float = 1e-3):
        self.tol = tol

    def check_convergence(
        self,
        pi: np.ndarray | None,
        prev_pi: np.ndarray | None,
        current_eqs: list[str],
        prev_hypotheses: list[str] | None,
        tol: float | None = None,
        iteration: int = 0,
    ) -> bool:
        """Check if the agent has converged."""
        tol = tol if tol is not None else self.tol

        if prev_pi is None or pi is None:
            return False

        delta = np.linalg.norm(pi - prev_pi)

        if delta < tol:
            print("  [+] Beliefs converged")

            if prev_hypotheses is not None and set(current_eqs) == set(prev_hypotheses):
                print(
                    f"\n[CONVERGED] Belief + hypothesis set stabilized at iteration {iteration + 1}"
                )
                return True

        return False

    def report_progress(
        self,
        n_proposed: int,
        n_verified: int,
        pi: np.ndarray | None,
        prev_pi: np.ndarray | None,
    ) -> None:
        """Report iteration progress."""
        n_rejected = n_proposed - n_verified

        print(f"  Proposed: {n_proposed}, Verified: {n_verified}, Rejected: {n_rejected}")

        if pi is not None:
            print(f"  Belief state: {pi}")

            if prev_pi is not None:
                delta = np.linalg.norm(pi - prev_pi)
                print(f"  Belief change: {delta:.6f}")
