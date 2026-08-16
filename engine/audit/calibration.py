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

Nothing here names a domain. Each takes a ``BaseSearcher`` and a problem it
never inspects, so the same calibration runs against any plugin's primitive.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from engine.audit.arms import BaseSearcher, SearchOutcome, paired_seed

__all__ = ["NullScaffold", "OracleScaffold", "WastefulScaffold"]

# The control arm derives its restart seeds from the arm seed, by one of two
# rules depending on whether common random numbers are in force. A calibration
# scaffold must not collide with either, or the arms become literally identical
# and every difference is exactly zero -- which yields `NULL` from a comparison
# that never happened. Offsetting the arm seed into a disjoint region keeps the
# draws independent while leaving them identically distributed.
_CALIBRATION_OFFSET = 1_500_450_271


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

    Expected verdict: ``CONTRIBUTES`` on ``metric``. ``INCONCLUSIVE`` means the
    design lacks the power to see even a deliberately inflated effect, and every
    ``NULL`` in the record should be read in that light.
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
