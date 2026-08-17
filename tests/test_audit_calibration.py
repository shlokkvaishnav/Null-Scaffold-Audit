"""Tests for the scaffolds whose verdicts are known before the audit runs.

These check the *calibration instruments themselves* -- that each one really is
null, wasteful, or oracular by construction -- and then check that the audit
returns the verdict each was built to earn.

The searcher here is a fake with tunable noise rather than a real symbolic
regressor, for two reasons. It runs in milliseconds, so these belong in the
ordinary suite. And it makes the ground truth exact: the true difference between
arms is known in advance, so a failure here is unambiguously the audit's, and
not a matter of what a search happened to find that day.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pytest

from engine.audit import (
    NullScaffold,
    OracleScaffold,
    SearchOutcome,
    Verdict,
    WastefulScaffold,
    audit,
)
from engine.audit.calibration import selection_ceiling

SEEDS = list(range(40))
MARGIN = 1.0
RESTARTS = 5

# How badly the proxy misleads, per instrument. These are properties of the
# *searcher* rather than of the audit -- the audit's design is identical across
# all three -- and each is set so its instrument's true effect clears the margin
# with room to spare, rather than sitting on the line where a test would pass or
# fail on Monte Carlo luck.
#
# Chosen by measurement, not by taste. At `ORACLE_PROXY_NOISE = 4.0` the oracle's
# true advantage is 0.775 against a margin of 1.0, and the audit correctly
# returns INCONCLUSIVE -- an effect genuinely smaller than the margin is not
# something the instrument should certify, and a "fix" that made it do so would
# be breaking exactly the property under test.
ORACLE_PROXY_NOISE = 12.0
WASTEFUL_PROXY_NOISE = 0.5
NULL_PROXY_NOISE = 1.0


@dataclass
class FakeSearcher:
    """A searcher whose quality is drawn from a fixed distribution.

    ``rmse`` is the outcome that gets measured; ``selection_score`` is the noisy
    proxy a pipeline actually selects on, standing in for the gap between what a
    searcher can see while running and what it is judged by afterwards.
    ``proxy_noise`` sets how wide that gap is -- which is exactly the quantity
    ``OracleScaffold`` exploits, and the one ``WastefulScaffold`` needs small.
    """

    restart_cost: int = 100
    proxy_noise: float = 1.0
    quality_sd: float = 2.0
    """How much the candidates themselves differ. Setting it to zero is the
    "restarts tie" case: nothing to select between, whatever the rule."""

    def search(self, problem: Any, seed: int) -> SearchOutcome:
        rng = np.random.default_rng(seed)
        quality = float(rng.normal(10.0, self.quality_sd))
        observed = quality + float(rng.normal(0.0, self.proxy_noise))
        return SearchOutcome(
            metrics={"rmse": quality, "selection_score": -observed},
            evaluations_used=self.restart_cost,
            representation=f"candidate-{seed}",
        )

    def select(self, outcomes: Sequence[SearchOutcome]) -> SearchOutcome:
        return max(outcomes, key=lambda outcome: outcome.metrics["selection_score"])


def run(scaffold: Any) -> Any:
    return audit(
        scaffold,
        problem=None,
        seeds=SEEDS,
        margins={"rmse": MARGIN},
        higher_is_better={"rmse": False},
    )


# --------------------------------------------------------------------------
# The instruments are what they claim to be
# --------------------------------------------------------------------------


def test_null_scaffold_does_not_reuse_the_control_arm_seeds() -> None:
    """The arms must be independent draws, not the same draw twice.

    Sharing the control's seeds would make every paired difference exactly zero
    and produce a ``NULL`` establishing only that the code runs. The calibration
    is worth something precisely because the arms disagree and equivalence still
    has to be established through that disagreement.
    """
    scaffold = NullScaffold(base=FakeSearcher(), restarts=3)
    outcome = scaffold.run(None, seed=0)
    control_representations = {f"candidate-{0 * 1_000_003 + r}" for r in range(3)}
    assert not set(outcome.intermediate_representations) & control_representations


def test_every_calibration_spends_the_same_budget() -> None:
    """All three must cost the same, or their verdicts are not comparable.

    A suite whose instruments cost different amounts would confound "the audit
    cannot see this effect" with "this arm was given fewer restarts", which is
    the confusion budget matching exists to remove.
    """
    base = FakeSearcher(restart_cost=100)
    spends = {
        scaffold.name: scaffold.run(None, seed=1).evaluations_used
        for scaffold in (
            NullScaffold(base=base, restarts=4),
            WastefulScaffold(base=base, restarts=4),
            OracleScaffold(base=base, restarts=4),
        )
    }
    assert set(spends.values()) == {400}


def test_wasteful_keeps_what_the_pipeline_rule_rejects() -> None:
    """The worst by the pipeline's own rule, found without naming a metric.

    ``WastefulScaffold`` ranks by pairwise probes of ``select`` rather than by
    reading a metric, so it stays correct against a pipeline whose selection
    rule this module has never seen. This pins that the probe recovers the
    rule's actual ordering.
    """
    base = FakeSearcher(proxy_noise=0.5)
    outcomes = [base.search(None, seed) for seed in (11, 12, 13)]
    worst = min(outcomes, key=lambda outcome: outcome.metrics["selection_score"])
    scaffold = WastefulScaffold(base=base, restarts=3)
    assert scaffold._keep(outcomes).representation == worst.representation


def test_oracle_keeps_the_best_by_the_measured_metric() -> None:
    base = FakeSearcher(proxy_noise=4.0)
    outcomes = [base.search(None, seed) for seed in (21, 22, 23)]
    best = min(outcomes, key=lambda outcome: outcome.metrics["rmse"])
    scaffold = OracleScaffold(base=base, restarts=3)
    assert scaffold._keep(outcomes).representation == best.representation


def test_oracle_refuses_a_metric_the_searcher_does_not_report() -> None:
    """Selecting on an absent metric must fail loudly, not fall back to a default.

    A silent fallback would turn the positive control into another null one, and
    the suite would then report that the audit cannot detect a contribution when
    in fact no contribution had been constructed.
    """
    scaffold = OracleScaffold(base=FakeSearcher(), restarts=2, metric="absent")
    with pytest.raises(KeyError, match="absent"):
        scaffold.run(None, seed=0)


def test_unwrap_returns_the_primitive_the_control_arm_will_use() -> None:
    base = FakeSearcher()
    assert NullScaffold(base=base).unwrap() is base


# --------------------------------------------------------------------------
# The ceiling on what selection alone can buy
# --------------------------------------------------------------------------


def test_ceiling_bounds_what_the_oracle_actually_achieves() -> None:
    """The property that makes it a bound rather than an estimate.

    No rule can select better than taking the best candidate by the measured
    metric, so the oracle -- which does exactly that -- must land at the ceiling
    and never above it. If this ever failed, the ceiling would be describing a
    different quantity from the one it is used to rule things out with.
    """
    base = FakeSearcher(proxy_noise=ORACLE_PROXY_NOISE)
    ceiling = selection_ceiling(
        base, None, SEEDS, metric="rmse", restarts=RESTARTS, higher_is_better=False
    )

    oracle = OracleScaffold(base=base, restarts=RESTARTS)
    null = NullScaffold(base=base, restarts=RESTARTS)
    achieved = float(
        np.mean(
            [
                null.run(None, seed).metrics["rmse"] - oracle.run(None, seed).metrics["rmse"]
                for seed in SEEDS
            ]
        )
    )
    assert achieved == pytest.approx(ceiling["ceiling"], rel=1e-9)


def test_ceiling_collapses_when_the_rule_already_agrees_with_the_metric() -> None:
    """A searcher whose selection rule is the metric leaves nothing to exploit.

    This is the case that matters in practice and the one that caught me out: on
    four of seven real benchmark problems the searcher's training score ranked
    candidates almost exactly as held-out error did, the ceiling was under a
    twentieth of the margin, and a scaffold *built* to contribute could not.
    Reading that as a broken audit would have been wrong.
    """
    agreeing = selection_ceiling(
        FakeSearcher(proxy_noise=0.0), None, SEEDS, metric="rmse", restarts=RESTARTS
    )
    disagreeing = selection_ceiling(
        FakeSearcher(proxy_noise=ORACLE_PROXY_NOISE),
        None,
        SEEDS,
        metric="rmse",
        restarts=RESTARTS,
    )
    assert agreeing["ceiling"] == pytest.approx(0.0, abs=1e-12)
    assert disagreeing["ceiling"] > MARGIN


def test_ceiling_is_never_negative() -> None:
    """Selecting better cannot hurt, so the bound is one-sided by construction."""
    for noise in (0.0, 1.0, 12.0):
        result = selection_ceiling(
            FakeSearcher(proxy_noise=noise), None, SEEDS, metric="rmse", restarts=3
        )
        assert result["ceiling"] >= 0.0


def test_ceiling_rejects_a_design_with_nothing_to_choose_between() -> None:
    with pytest.raises(ValueError, match="at least 2"):
        selection_ceiling(FakeSearcher(), None, SEEDS, metric="rmse", restarts=1)


def test_ceiling_refuses_an_unreported_metric() -> None:
    with pytest.raises(KeyError, match="absent"):
        selection_ceiling(FakeSearcher(), None, SEEDS, metric="absent", restarts=3)


# --------------------------------------------------------------------------
# The audit returns the verdict each instrument was built to earn
# --------------------------------------------------------------------------


def test_a_null_scaffold_is_certified_null() -> None:
    """The load-bearing test of the whole subsystem.

    If the audit cannot say ``NULL`` when the true difference is exactly zero
    and the budget is matched, then every ``INCONCLUSIVE`` it has reported was a
    property of the design rather than of the pipeline under test, and no
    ``NULL`` in the record means anything.
    """
    report = run(NullScaffold(base=FakeSearcher(proxy_noise=NULL_PROXY_NOISE), restarts=RESTARTS))
    assert report.verdict is Verdict.NULL
    assert report.per_metric["rmse"].power > 0.5


def test_a_wasteful_scaffold_is_caught_as_harmful() -> None:
    """Same budget, worse answer. Reading this as anything else inverts the tool."""
    report = run(
        WastefulScaffold(base=FakeSearcher(proxy_noise=WASTEFUL_PROXY_NOISE), restarts=RESTARTS)
    )
    assert report.verdict is Verdict.HARMFUL
    assert report.per_metric["rmse"].observed_difference < -MARGIN


def test_an_oracle_scaffold_is_credited_as_contributing() -> None:
    """The audit's only untested direction, and the one that makes NULL credible.

    An instrument that has never returned ``CONTRIBUTES`` cannot distinguish a
    pipeline that contributes nothing from a design that could not have noticed.
    """
    report = run(
        OracleScaffold(base=FakeSearcher(proxy_noise=ORACLE_PROXY_NOISE), restarts=RESTARTS)
    )
    assert report.verdict is Verdict.CONTRIBUTES
    assert report.per_metric["rmse"].observed_difference > MARGIN


def test_an_effect_smaller_than_the_margin_stays_inconclusive() -> None:
    """The margin has to mean something, or ``CONTRIBUTES`` is just "differs".

    At this proxy noise the oracle's true advantage is about 0.775 against a
    margin of 1.0 -- a real effect, but a practically uninteresting one by the
    standard registered in advance. ``INCONCLUSIVE`` is the right answer, and
    this test exists so that a later change which makes the positive control
    pass more easily cannot do so by quietly eroding the margin.
    """
    report = run(OracleScaffold(base=FakeSearcher(proxy_noise=4.0), restarts=3))
    assert report.verdict is Verdict.INCONCLUSIVE
    assert 0.0 < report.per_metric["rmse"].observed_difference < MARGIN


def test_the_three_verdicts_are_distinct() -> None:
    """Stated as one assertion because the suite's claim is joint.

    Any single verdict is reachable by an audit broken in the right direction --
    one that always says ``HARMFUL`` passes the wasteful test on its own. Only
    producing all three under the same audit design, the same seed count and the
    same budget shows the instrument discriminates rather than leans.
    """
    reports = [
        run(NullScaffold(base=FakeSearcher(proxy_noise=NULL_PROXY_NOISE), restarts=RESTARTS)),
        run(
            WastefulScaffold(base=FakeSearcher(proxy_noise=WASTEFUL_PROXY_NOISE), restarts=RESTARTS)
        ),
        run(OracleScaffold(base=FakeSearcher(proxy_noise=ORACLE_PROXY_NOISE), restarts=RESTARTS)),
    ]
    assert {report.verdict for report in reports} == {
        Verdict.NULL,
        Verdict.HARMFUL,
        Verdict.CONTRIBUTES,
    }
    # Same budget in all three, so no verdict is explained by unequal compute.
    assert len({report.arms.restarts_per_seed for report in reports}) == 1


def test_metric_spread_separates_a_tied_field_from_a_faithful_rule() -> None:
    """A low ceiling has two causes, and the ceiling alone cannot tell them apart.

    Either the restarts landed in the same place, so there was nothing to select
    between -- a fact about the problem -- or they differed a great deal and the
    pipeline's rule already picked the best, which is a fact about the selection
    signal and the far stronger finding. Reporting the within-seed spread beside
    the ceiling is what makes them distinguishable from the artifact alone.

    Both were observed on the real benchmark: `ideal_gas_law` had candidates whose
    error varied by twice the margin while its ceiling sat at three hundredths of
    it, and `ohms_law` had a spread of exactly zero.
    """
    faithful = selection_ceiling(
        FakeSearcher(proxy_noise=0.0, quality_sd=2.0),
        None,
        SEEDS,
        metric="rmse",
        restarts=RESTARTS,
    )
    tied = selection_ceiling(
        FakeSearcher(proxy_noise=0.0, quality_sd=0.0),
        None,
        SEEDS,
        metric="rmse",
        restarts=RESTARTS,
    )

    # Both have a zero ceiling -- the rule is perfect in both -- so the ceiling
    # cannot be what distinguishes them.
    assert faithful["ceiling"] == pytest.approx(0.0, abs=1e-12)
    assert tied["ceiling"] == pytest.approx(0.0, abs=1e-12)

    # The spread is what does.
    assert faithful["metric_spread"] > 1.0
    assert tied["metric_spread"] == pytest.approx(0.0, abs=1e-12)


def test_metric_spread_is_never_negative() -> None:
    for noise in (0.0, 1.0, 12.0):
        result = selection_ceiling(
            FakeSearcher(proxy_noise=noise), None, SEEDS, metric="rmse", restarts=3
        )
        assert result["metric_spread"] >= 0.0


def test_the_upper_bound_never_sits_below_the_mean() -> None:
    """It is a bound, so it has to bound. Cheap, and catches a sign slip."""
    for noise in (0.0, 1.0, 12.0):
        result = selection_ceiling(
            FakeSearcher(proxy_noise=noise), None, SEEDS, metric="rmse", restarts=3
        )
        assert result["ceiling_upper"] >= result["ceiling"]


def test_the_bound_collapses_to_the_estimate_when_every_seed_agrees() -> None:
    """No sampling variation means nothing to extrapolate from.

    Widening here would invent uncertainty; the honest bound is the observation.
    This is the same degenerate case the verdict statistics handle, and it is handled
    the same way -- as a limit on what can be claimed, not a licence for extra
    confidence.
    """
    result = selection_ceiling(
        FakeSearcher(proxy_noise=0.0, quality_sd=0.0), None, SEEDS, metric="rmse", restarts=3
    )
    assert result["ceiling_sd"] == pytest.approx(0.0, abs=1e-12)
    assert result["ceiling_upper"] == pytest.approx(result["ceiling"], abs=1e-12)


def test_fewer_seeds_give_a_looser_bound() -> None:
    """The bound has to pay for a small sample, or it is not doing its job.

    A closure claim from three seeds must be harder to make than one from forty. If
    this failed, the instrument would let a thin run assert the same guarantee as a
    thorough one.
    """
    searcher = FakeSearcher(proxy_noise=ORACLE_PROXY_NOISE)
    thin = selection_ceiling(searcher, None, SEEDS[:4], metric="rmse", restarts=3)
    thorough = selection_ceiling(searcher, None, SEEDS, metric="rmse", restarts=3)
    assert (thin["ceiling_upper"] - thin["ceiling"]) > (
        thorough["ceiling_upper"] - thorough["ceiling"]
    )
