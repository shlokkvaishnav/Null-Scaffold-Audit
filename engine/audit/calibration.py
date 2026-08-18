"""Scaffolds whose verdicts are known before the audit runs.

An instrument that has only ever returned one answer is indistinguishable from
an instrument that can only return one answer. This audit has reported ``NULL``
and ``INCONCLUSIVE`` on every pipeline put to it so far, and that record is
compatible with two very different worlds: one where the pipelines genuinely
contributed nothing, and one where the audit cannot detect a contribution at
all. Nothing in the audit's own output separates them.

The scaffolds here separate them. Each wraps a base searcher in a way whose
sign is fixed by construction rather than discovered by measurement, so the
verdict each *should* receive is known in advance:

* :class:`NullScaffold` draws from exactly the same distribution as the control
  arm. Any difference between the arms is noise. It should read ``NULL``.
* :class:`WastefulScaffold` spends the same budget and then deliberately keeps
  the worst of what it found. It should read ``HARMFUL``.
* :class:`OracleScaffold` selects using information the control is not allowed
  to see. It should read ``CONTRIBUTES``.

A run of all three is a calibration, not an experiment. If the audit cannot
produce all three verdicts on demand, then a ``NULL`` it reports about a real
pipeline is not evidence about that pipeline -- and which of the three it fails
on says what is wrong. Failing ``NULL`` while passing the other two is a power
problem. Failing ``CONTRIBUTES`` means the metrics do not capture what the
scaffold changed. Failing ``HARMFUL`` means the orientation is inverted.

These are deliberately *not* proposals. ``OracleScaffold`` cheats, and cheating
is the point: it is a ruler with a known length, not a method anyone should use.

:func:`selection_ceiling` is the other half, and is worth running *first*. A
wrapper at matched budget can only win by selecting better among the candidates
it generated or by generating better ones, and the ceiling measures the most the
first could ever be worth. Where it falls below the pre-registered margin, every
selection-only wrapper is null on that problem by construction -- which is a
fact about the design, obtainable in one arm and a few seeds, rather than
something to rediscover one underpowered sweep at a time. It also says when
``OracleScaffold`` is the wrong ruler to reach for, because a problem with no
selection gap leaves that scaffold nothing to exploit.

Nothing here names a domain. Each takes a ``BaseSearcher`` and a problem it
never inspects, so the same calibration runs against any plugin's primitive.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy import stats

from engine.audit.arms import BaseSearcher, SearchOutcome, paired_seed

__all__ = ["NullScaffold", "OracleScaffold", "WastefulScaffold", "selection_ceiling"]

# The control arm derives its restart seeds from the arm seed, by one of two
# rules depending on whether common random numbers are in force. A calibration
# scaffold must not collide with either, or the arms become literally identical
# and every difference is exactly zero -- which yields `NULL` from a comparison
# that never happened. Offsetting the arm seed into a disjoint region keeps the
# draws independent while leaving them identically distributed.
_CALIBRATION_OFFSET = 1_500_450_271

_CEILING_ALPHA = 0.05
"""One-sided error rate for the ceiling's upper bound.

Matches the per-side alpha of the two-sided 90% intervals the verdicts use, so a
ceiling and a verdict are quoted at the same confidence and can be read together.
"""


def selection_ceiling(
    base: BaseSearcher,
    problem: Any,
    seeds: Sequence[int],
    *,
    metric: str,
    restarts: int = 3,
    higher_is_better: bool = False,
) -> dict[str, float]:
    """The most any selection-only wrapper could gain, at this budget.

    A wrapper that spends its budget on the same searcher can beat the restart
    baseline in exactly two ways: by choosing better among the candidates it
    generated, or by generating better candidates. This measures the ceiling on
    the first. No rule can select better than taking the best candidate by the
    metric being measured, so the gap between what the pipeline's own rule picks
    and the best available *is* the upper bound -- reached by no real wrapper,
    exceeded by none either.

    That makes it worth far more than a diagnostic. Where the ceiling sits below
    the pre-registered margin, every wrapper that only reorders or filters what
    the searcher produced is null by construction on that problem, and no seed
    count rescues it. The audit can then report a design fact instead of
    spending days rediscovering it one ``INCONCLUSIVE`` at a time.

    It is also cheap relative to what it saves: ``len(seeds) * restarts``
    searches and no second arm, against a full two-arm sweep.

    ``metric_spread`` -- how much the candidates within one seed differ on the
    metric -- is returned alongside, because a low ceiling has two very different
    causes and the ceiling alone cannot tell them apart:

    * The spread is also low: the restarts all landed in the same place, so there
      was nothing to choose between. That is a statement about the problem's
      difficulty, and on an easy benchmark it says little about wrappers.
    * The spread is high but the ceiling is not: the candidates differed a great
      deal and the pipeline's own rule *already picked the best one*. That is a
      statement about the selection signal being faithful, and it is the stronger
      finding -- there is no selection alpha for a wrapper to capture, however
      much variety the searcher produced.

    Both were observed. On one benchmark problem the candidates' error varied by
    twice the pre-registered margin while the ceiling sat at three hundredths of
    it, which is the second case in its sharpest form.

    ``ceiling_upper`` is the number to compare against a margin. The claim this
    quantity supports is a *bound* -- "no selection-only wrapper can clear the margin
    here" -- and asserting a bound from a point estimate is precisely the error the rest
    of this subsystem refuses. Measured on one benchmark the distinction moved a single
    problem of thirty-six, which is small but is the difference between an average and
    a guarantee.

    Returns the mean ceiling, that upper bound, the spread across seeds, the within-seed
    metric spread, and the design it was measured under. Deliberately not a verdict: the
    margin lives with the caller that pre-registered it, and this module has no
    business deciding what counts as a practically interesting gain in someone
    else's domain.
    """
    if restarts < 2:
        raise ValueError(f"a ceiling needs at least 2 candidates to choose between, got {restarts}")

    gains: list[float] = []
    spreads: list[float] = []
    for seed in seeds:
        outcomes = [
            base.search(problem, paired_seed(seed + _CALIBRATION_OFFSET, restart))
            for restart in range(restarts)
        ]
        missing = [o for o in outcomes if metric not in o.metrics]
        if missing:
            raise KeyError(
                f"{metric!r} is not reported by the searcher; it reports {sorted(missing[0].metrics)}"
            )

        values = [float(o.metrics[metric]) for o in outcomes]
        chosen = float(base.select(outcomes).metrics[metric])
        best = max(values) if higher_is_better else min(values)
        # Oriented so positive always means "better selection would have helped".
        gains.append(best - chosen if higher_is_better else chosen - best)
        spreads.append(float(np.std(values, ddof=1)))

    array = np.asarray(gains, dtype=float)
    mean = float(array.mean())
    deviation = float(array.std(ddof=1)) if len(array) > 1 else 0.0

    # An upper confidence bound, because the claim this quantity is used to make is a
    # bound: "no selection-only wrapper can clear the margin here". Comparing a *mean*
    # to a margin would assert that on a point estimate, which is the error this whole
    # subsystem exists to refuse -- and one this module committed until it was caught.
    #
    # One-sided at 5%, matching the per-side alpha of the two-sided 90% intervals the
    # verdicts use, so a ceiling and a verdict are quoted at the same confidence.
    if len(array) > 1 and deviation > 0.0:
        critical = float(stats.t.ppf(1.0 - _CEILING_ALPHA, len(array) - 1))
        upper = mean + critical * deviation / math.sqrt(len(array))
    else:
        # Every seed agreed exactly. There is no sampling variation to extrapolate
        # from, so the bound is the observation -- the same degenerate case the
        # statistics layer handles, and equally not a licence for extra confidence.
        upper = mean

    return {
        "ceiling": mean,
        "ceiling_sd": deviation,
        "ceiling_upper": float(upper),
        "metric_spread": float(np.mean(spreads)),
        "seeds": float(len(array)),
        "restarts": float(restarts),
    }


@dataclass
class _RestartScaffold:
    """Shared machinery: run ``restarts`` searches, then keep one of them.

    Subclasses differ only in which one they keep, which is what fixes the sign
    of their verdict. The budget is identical across all three, so the audit's
    control arm receives the same number of restarts in every case and the three
    calibrations are comparable to each other as well as to a real pipeline.
    """

    base: BaseSearcher
    restarts: int = 3
    name: str = "calibration"
    failures: list[str] = field(default_factory=list)

    def unwrap(self) -> BaseSearcher:
        return self.base

    def _seed_for(self, seed: int, restart: int) -> int:
        return paired_seed(seed + _CALIBRATION_OFFSET, restart)

    def _keep(self, outcomes: Sequence[SearchOutcome]) -> SearchOutcome:
        raise NotImplementedError

    def run(self, problem: Any, seed: int) -> SearchOutcome:
        outcomes = [
            self.base.search(problem, self._seed_for(seed, restart))
            for restart in range(self.restarts)
        ]
        kept = self._keep(outcomes)
        return SearchOutcome(
            metrics=kept.metrics,
            # The whole arm's spend, not the kept restart's: the ones discarded
            # were still paid for, and charging only the survivor would hand the
            # control arm fewer restarts than this scaffold actually consumed.
            evaluations_used=sum(outcome.evaluations_used for outcome in outcomes),
            representation=kept.representation,
            intermediate_representations=tuple(
                outcome.representation or "" for outcome in outcomes
            ),
        )


@dataclass
class NullScaffold(_RestartScaffold):
    """Null by construction: the same search, the same budget, the same rule.

    This draws ``restarts`` independent searches and selects among them with the
    base searcher's own ``select``, which is precisely what the control arm does.
    The two arms therefore sample the same distribution, and the true difference
    between them is exactly zero on every metric.

    The seeds are *not* shared with the control. A version that reused them would
    make the arms byte-identical and produce a ``NULL`` that tests only that the
    plumbing runs. The question worth asking is harder and more useful: can this
    design establish equivalence when the difference really is zero but the
    observations still disagree, which is the situation every real audit is in?

    Expected verdict: ``NULL``. An ``INCONCLUSIVE`` here is a finding about the
    audit, not about any scaffold -- it means the seed count and the margins
    cannot certify equivalence even when equivalence is true, and therefore that
    no ``INCONCLUSIVE`` this audit has ever reported carried information.
    """

    name: str = "calibration.null"

    def _keep(self, outcomes: Sequence[SearchOutcome]) -> SearchOutcome:
        return self.base.select(outcomes)


@dataclass
class WastefulScaffold(NullScaffold):
    """Harmful by construction: the same budget, then keep the worst of it.

    Inverting the selection rule is a deliberately crude way to spend compute
    badly, and crude is what a calibration wants -- the effect should be far
    outside the margin so that failing to detect it is unambiguous.

    Expected verdict: ``HARMFUL``. ``CONTRIBUTES`` here means an orientation is
    inverted somewhere, which is the failure that leaves every number plausible
    while reversing the conclusion.
    """

    name: str = "calibration.wasteful"

    def _keep(self, outcomes: Sequence[SearchOutcome]) -> SearchOutcome:
        best = self.base.select(outcomes)
        # Defined as "whatever select() does not pick" rather than by naming a
        # metric, so this stays correct for a pipeline whose selection rule this
        # module has never seen.
        worst = min(
            outcomes,
            key=lambda outcome: sum(1 for other in outcomes if self._beats(outcome, other)),
        )
        return worst if len(outcomes) > 1 else best

    def _beats(self, candidate: SearchOutcome, other: SearchOutcome) -> bool:
        """Whether ``candidate`` survives ``select`` against ``other`` alone.

        A pairwise probe of the pipeline's own rule. Ranking by how many
        head-to-head comparisons each outcome wins recovers the rule's ordering
        without this module needing to know which metric it reads or which
        direction that metric runs.
        """
        return self.base.select([candidate, other]) is candidate


@dataclass
class OracleScaffold(_RestartScaffold):
    """Contributing by construction: selection on information the control lacks.

    The control arm must select using the pipeline's own rule, which is computed
    from what a pipeline actually observes. This one selects on a named outcome
    metric instead -- the audit's own reported measure, computed on held-out
    data. That is cheating, and it is why the effect is real and one-directional.

    It exists to answer a question no ``NULL`` can: *if* a wrapper genuinely
    improved the outcome, at this budget and this seed count, would this audit
    say so? A calibration suite without this is a suite that has only ever been
    asked to confirm the absence of something.

    Expected verdict: ``CONTRIBUTES`` **on ``metric``, not overall.** That
    distinction was learned the hard way. Run against a real searcher this
    scaffold returned ``CONTRIBUTES`` on rmse at nine times the margin -- exactly
    as designed -- and ``HARMFUL`` overall, because selecting purely on held-out
    error picks bloated expressions and the complexity metric caught it. The
    overall rule reports harm over help by design, so a scaffold that cheats on
    one pre-registered metric while losing another is *correctly* harmful.

    Read the per-metric verdict on ``metric`` when calibrating. An
    ``INCONCLUSIVE`` there means the design lacks the power to see even a
    deliberately inflated effect, and every ``NULL`` in the record should be read
    in that light.

    The bloat is worth noticing rather than working around: it is a small piece
    of evidence for registering a complexity metric at all, and it agrees with
    the symbolic-regression model-selection literature, where description-length
    criteria beat plain held-out error.

    **Its headroom is a property of the problem, and must be checked first.**
    This cheats by exploiting the disagreement between the rule a pipeline
    selects with and the measure it is judged by. Where those two agree, the
    cheat is worth nothing and this scaffold is null despite being built to
    contribute -- so an ``INCONCLUSIVE`` here means *either* that the audit is
    underpowered *or* that there was no effect to find, and the two are not
    distinguishable from the verdict alone.

    That is not hypothetical. Measured on the real candidates of one benchmark,
    the gain from selecting on held-out error rather than on the training rule
    was 7.6x, 4.3x and 1.5x the pre-registered margin on three problems, and
    below 0.05x on four others -- the searcher's training score ranked
    candidates almost exactly as held-out error did, so there was nothing to
    exploit. Calibrating on one of those four would have reported a broken
    instrument when the instrument was fine.

    So estimate the headroom before spending a sweep: draw candidates the
    searcher actually produces, and compare the measure of the one the
    pipeline's rule would pick against the best available. If that gap does not
    clear the margin, this scaffold cannot return ``CONTRIBUTES`` on that
    problem and a positive control has to come from somewhere else -- from
    changing how the budget is *spent* rather than how the winner is picked.
    """

    name: str = "calibration.oracle"
    metric: str = "rmse"
    higher_is_better: bool = False

    def _keep(self, outcomes: Sequence[SearchOutcome]) -> SearchOutcome:
        missing = [o for o in outcomes if self.metric not in o.metrics]
        if missing:
            raise KeyError(
                f"{self.name} selects on {self.metric!r}, which the base searcher "
                f"does not report; it reports {sorted(missing[0].metrics)}"
            )
        chooser = max if self.higher_is_better else min
        return chooser(outcomes, key=lambda outcome: float(outcome.metrics[self.metric]))
