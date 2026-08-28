"""Cross-arm margin degeneracy: is the pre-registered margin still anchored
to real variability, or has it collapsed to a numerical floor? (issue #27)

A margin is conventionally derived from the control arm's own spread (e.g.
``fraction * control_spread``). When the control arm's independent restarts
already converge to (numerically) the same point on every seed -- not
because the wrapper is uninteresting, but because the problem's own control
arm has nothing left to explore -- that spread collapses toward zero and the
derived margin has to fall back on an arbitrary numerical floor with no
domain meaning. A verdict computed against that floor is then decided by
floating-point-scale noise in whichever seeds were drawn, not by anything
about the wrapper's actual behaviour (`MARGIN_DEGENERACY_SPEC.md`'s
Research question; the concrete incident is issue #25/PR #26's Griewank
row, whose verdict flipped between two audits with the same design).

This module does not know the margin's derivation formula or its floor
constant -- both are plugin-level choices (Article 5, domain
independence). What it *can* see, from data every ``audit()`` call already
has, is whether the control arm's cross-seed spread on a metric is
degenerate **relative to the treatment arm's spread on the same metric,
same problem, same seeds**. A treatment arm exploring normally while its
paired control arm is frozen at floating-point-noise scale is the
structural signature of a control-derived margin having collapsed -- and,
per this file's own validation (`MARGIN_DEGENERACY_SPEC.md`'s Results),
that signature separates every already-published `plugins/basinhopping_
audit/` row into two clusters eight orders of magnitude apart, with no
row in between: the two untrustworthy Griewank readings at a control:
treatment spread ratio of ~4-5e-10, and every trusted row (Griewank's
domain-scaled reading included) at a ratio between ~0.2 and ~2.1.

Reported alongside the verdict, the same relationship ``DegeneracyReport``
already has to `AuditReport` -- it identifies a mechanism, not merely an
outcome, and does not itself change what the verdict asserts.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

DEFAULT_RATIO_THRESHOLD = 1e-4
"""Below this control:treatment spread ratio, the control arm's cross-seed
variability is judged too small, relative to the treatment arm operating on
the same problem and seeds, to have produced a domain-meaningful margin.

Picked from deep inside an eight-order-of-magnitude gap in this project's
own committed history, not tuned to an edge: every already-published
`plugins/basinhopping_audit/` row's ratio falls either at or below ~5e-10
(both untrustworthy Griewank `run_audit.py` readings, issues #15 and #25)
or at or above ~0.23 (every trusted row, including Griewank's
domain-scaled-stepsize `NULL` reading) -- see `MARGIN_DEGENERACY_SPEC.md`'s
Results for the full table. `1e-4` sits roughly in the middle of that gap
on a log scale.
"""


@dataclass(frozen=True)
class MarginDegeneracyReport:
    """Whether a metric's control-arm spread is degenerate relative to its
    paired treatment-arm spread on the same problem and seeds.
    """

    assessed: bool
    control_spread: float = 0.0
    treatment_spread: float = 0.0
    ratio: float = 0.0
    threshold: float = 0.0

    @property
    def degenerate(self) -> bool:
        """True when the control arm's spread is degenerate relative to the
        treatment arm's -- the condition that forces a control-derived margin
        down to an arbitrary numerical floor.

        Requires ``assessed``: an unassessed report (fewer than two
        observations in an arm, or a treatment arm with zero spread of its
        own -- see ``assess_margin_degeneracy``) makes no claim either way.
        """
        return self.assessed and self.ratio < self.threshold

    def summary(self) -> str:
        if not self.assessed:
            return "not assessed (fewer than two observations, or treatment spread is zero)"
        label = "MARGIN_DEGENERATE" if self.degenerate else "anchored"
        return (
            f"{label}: control spread {self.control_spread:.3g} is "
            f"{self.ratio:.3g}x the treatment spread {self.treatment_spread:.3g} "
            f"(threshold {self.threshold:.3g})"
        )


def assess_margin_degeneracy(
    control: Sequence[float],
    treatment: Sequence[float],
    *,
    threshold: float = DEFAULT_RATIO_THRESHOLD,
) -> MarginDegeneracyReport:
    """Compare control-arm spread to treatment-arm spread on one metric.

    Needs at least two observations per arm to define a spread; fewer than
    that reports unassessed rather than a spurious zero. A treatment arm
    with zero spread of its own is also left unassessed rather than
    reported degenerate or not: both arms converging together is
    ``_guard_vacuous_comparison``'s structural failure (arms.py), a
    different mechanism from this one, and the ratio this check reports is
    undefined (0/0), not zero.
    """
    if len(control) < 2 or len(treatment) < 2:
        return MarginDegeneracyReport(assessed=False)

    control_spread = float(np.std(np.asarray(control, dtype=float), ddof=1))
    treatment_spread = float(np.std(np.asarray(treatment, dtype=float), ddof=1))

    if treatment_spread == 0.0:
        return MarginDegeneracyReport(
            assessed=False, control_spread=control_spread, treatment_spread=treatment_spread
        )

    return MarginDegeneracyReport(
        assessed=True,
        control_spread=control_spread,
        treatment_spread=treatment_spread,
        ratio=control_spread / treatment_spread,
        threshold=threshold,
    )
