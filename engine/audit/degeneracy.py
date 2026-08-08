"""Intra-run degeneracy: is the wrapper exploring, or repeating? (RFC-0001 section 4.3)

The statistical arms answer whether a wrapper *helped*. They cannot say why not.
This check answers the why, costs one run rather than thirty, and catches the
specific failure that motivated the whole audit: a loop whose iterations rebuild
the same search from the same state and therefore produce the same candidate
every time.

A wrapper can be null without being degenerate -- it may explore genuinely and
still not beat restarts. But a degenerate wrapper is null by construction, and
knowing that distinguishes "this idea does not help" from "this implementation
does not do the thing the idea describes". Those call for different fixes, and
conflating them wastes the audit's most useful signal.

Comparison is by string equality on opaque representations. The engine cannot
parse them and does not try, so two spellings of the same thing count as
distinct and this check *under*-reports degeneracy. That direction is deliberate:
a false accusation of degeneracy is far more damaging than a missed one.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class DegeneracyReport:
    """How much genuine variety a wrapper produced inside each single run."""

    assessed: bool
    runs: int = 0
    degenerate_runs: int = 0
    mean_distinct_ratio: float = 0.0
    min_proposals: int = 0

    @property
    def degenerate(self) -> bool:
        """True when every assessed run produced no internal variety at all.

        Requires *all* runs, not a majority. A wrapper that repeats itself on
        some seeds and explores on others has a different problem from one that
        never explores, and the stronger claim is the one worth making
        mechanically.
        """
        return self.assessed and self.runs > 0 and self.degenerate_runs == self.runs

    def summary(self) -> str:
        if not self.assessed:
            return "not assessed (pipeline exposed no intra-run proposals)"
        label = "DEGENERATE" if self.degenerate else "explores"
        return (
            f"{label}: {self.degenerate_runs}/{self.runs} runs produced a single "
            f"distinct proposal; mean distinct ratio {self.mean_distinct_ratio:.2f}"
        )


def assess_degeneracy(runs: Sequence[Sequence[str]]) -> DegeneracyReport:
    """Assess intra-run variety across runs, each a sequence of proposals.

    A run with fewer than two proposals carries no information about variety --
    a single proposal is trivially "all identical" -- so such runs are excluded
    rather than counted as degenerate. Counting them would let a wrapper that
    proposes once per run be labelled degenerate for a property it was never in a
    position to exhibit.
    """
    usable = [list(run) for run in runs if len(run) >= 2]
    if not usable:
        return DegeneracyReport(assessed=False)

    degenerate_runs = 0
    ratios: list[float] = []
    for proposals in usable:
        distinct = len(set(proposals))
        ratios.append(distinct / len(proposals))
        if distinct == 1:
            degenerate_runs += 1

    return DegeneracyReport(
        assessed=True,
        runs=len(usable),
        degenerate_runs=degenerate_runs,
        mean_distinct_ratio=sum(ratios) / len(ratios),
        min_proposals=min(len(p) for p in usable),
    )
