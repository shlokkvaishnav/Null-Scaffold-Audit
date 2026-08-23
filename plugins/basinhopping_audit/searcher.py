"""LocalMinimizerRestart: this plugin's base searcher, per SPEC.md.

One `search()` call is one `scipy.optimize.minimize` call from a
uniform-random starting point within the function's bounds -- the literal
`B_restart` control the issue asks for: "independent calls to
`scipy.optimize.minimize` ... from uniform-random `x0`". `restart_cost = 1`
because the budget unit here is "count of local-minimization calls," and one
`search()` call spends exactly one.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize

from engine.audit.arms import SearchOutcome
from plugins.basinhopping_audit.functions import BenchmarkFunction

METRIC = "objective"
LOCAL_METHOD = "L-BFGS-B"
"""SPEC.md's "Confounds considered": a reasonable, common default, stated
here and not tuned after seeing verdicts -- see plugins/basinhopping_audit's
module docstring for why this must match the treatment's minimizer exactly."""

REPRESENTATION_DECIMALS = 4
"""Rounding for the degeneracy/identical-representation string. Coarse
enough that two runs converging to the same basin read as identical (the
comparison this check exists to make) without being so coarse that distinct
basins collide."""


@dataclass
class LocalMinimizerRestart:
    restart_cost: int = 1

    def search(self, problem: BenchmarkFunction, seed: int) -> SearchOutcome:
        rng = np.random.default_rng(seed)
        low = np.array([b[0] for b in problem.bounds])
        high = np.array([b[1] for b in problem.bounds])
        x0 = rng.uniform(low, high)
        result = minimize(problem.func, x0, method=LOCAL_METHOD, bounds=problem.bounds)
        rounded = tuple(round(float(v), REPRESENTATION_DECIMALS) for v in result.x)
        return SearchOutcome(
            metrics={METRIC: float(result.fun)},
            evaluations_used=1,
            representation=str(rounded),
        )

    def select(self, outcomes: Sequence[SearchOutcome]) -> SearchOutcome:
        return min(outcomes, key=lambda outcome: outcome.metrics[METRIC])
