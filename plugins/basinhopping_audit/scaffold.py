"""BasinhoppingScaffold: the pipeline under audit, per SPEC.md.

Wraps `scipy.optimize.basinhopping` exactly as any user would call it --
no modification to `scipy`'s code. `unwrap()` returns
`LocalMinimizerRestart`, the same local minimizer (`L-BFGS-B`) basinhopping
calls internally on every hop, so the control arm restarts the identical
primitive the treatment's scaffold logic wraps.

Budget accounting: one `run()` call performs `niter + 1` calls to
`scipy.optimize.minimize` (the initial minimization from `x0`, plus one per
hop) -- confirmed empirically (`res.nit == niter`) rather than assumed, since
`LocalMinimizerRestart.restart_cost = 1` makes this count the unit the
control arm's restarts are matched against (SPEC.md, "Budget-unit
fairness").
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import basinhopping

from engine.audit.arms import BaseSearcher, SearchOutcome
from plugins.basinhopping_audit.functions import BenchmarkFunction
from plugins.basinhopping_audit.searcher import (
    LOCAL_METHOD,
    METRIC,
    REPRESENTATION_DECIMALS,
    LocalMinimizerRestart,
)

DEFAULT_STEPSIZE = 0.5
"""scipy's own documented default -- SPEC.md: "Use scipy's documented
default unless the feasibility pre-check indicates it needs tuning, and
state explicitly which value was used." The feasibility probe (see
plugins/basinhopping_audit/run_audit.py) did not indicate a need to tune it,
so it is unchanged here."""


@dataclass
class BasinhoppingScaffold:
    niter: int
    name: str = "basinhopping_audit.basinhopping"
    stepsize: float = DEFAULT_STEPSIZE
    _base: LocalMinimizerRestart = field(default_factory=LocalMinimizerRestart)

    def unwrap(self) -> BaseSearcher:
        return self._base

    def run(self, problem: BenchmarkFunction, seed: int) -> SearchOutcome:
        rng = np.random.default_rng(seed)
        low = np.array([b[0] for b in problem.bounds])
        high = np.array([b[1] for b in problem.bounds])
        x0 = rng.uniform(low, high)

        # Every hop's locally-minimized candidate, in order -- this is what
        # makes the degeneracy pre-check (engine.audit.degeneracy.assess_
        # degeneracy) meaningful rather than "not assessed": it is exactly
        # the founding-incident check (does each hop actually land somewhere
        # different?), applied to basinhopping's own trajectory.
        proposals: list[str] = []

        def record(x: np.ndarray, f: float, accept: bool) -> None:
            del accept
            proposals.append(str(round(float(f), REPRESENTATION_DECIMALS)))

        result = basinhopping(
            problem.func,
            x0,
            niter=self.niter,
            stepsize=self.stepsize,
            minimizer_kwargs={"method": LOCAL_METHOD, "bounds": problem.bounds},
            seed=seed,
            callback=record,
        )
        rounded = tuple(round(float(v), REPRESENTATION_DECIMALS) for v in result.x)
        return SearchOutcome(
            metrics={METRIC: float(result.fun)},
            # niter hops + the initial minimization from x0; see module
            # docstring -- this is what makes the control's restart count
            # exactly match the number of local-minimizer calls this run
            # actually performed, not merely `niter`.
            evaluations_used=self.niter + 1,
            representation=str(rounded),
            intermediate_representations=tuple(proposals),
        )
