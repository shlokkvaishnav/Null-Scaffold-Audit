"""Tests for the paired-binary branch of the equivalence statistics.

A metric that records a success or a failure -- whether the ground-truth
equation was recovered -- is not a measurement, and the machinery that serves
measurements does something specific and wrong on it. When no seed disagrees
between the arms, every bootstrap resample is identical, the interval collapses
to a single point, and a point sits inside any margin. The audit then reports
``NULL`` -- its one positive verdict, the whole reason this subsystem exists --
from a sample that observed nothing.

The tests below pin the replacement against three independent checks, because a
statistic nobody can verify is not an improvement on the one it replaced:

* it reduces to McNemar's statistic at a difference of zero,
* its intervals cover at their nominal rate under simulation, and
* it does not certify equivalence from zero discordant pairs.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from engine.audit import Verdict, equivalence_verdict
from engine.audit.statistics import _tango_interval, _tango_score

MARGIN = 0.10


def arms(n: int, better: int, worse: int) -> tuple[list[float], list[float]]:
    """Paired 0/1 arms with ``better`` and ``worse`` discordant pairs.

    The remaining pairs agree, and agree on success, so the concordant cells are
    non-empty -- an all-failure table is a different degenerate case.
    """
    if better + worse > n:
        raise ValueError("more discordant pairs than pairs")
    agree = n - better - worse
    treatment = [1.0] * better + [0.0] * worse + [1.0] * agree
    control = [0.0] * better + [1.0] * worse + [1.0] * agree
    return treatment, control


def verdict_for(treatment: list[float], control: list[float], **kwargs: Any) -> Any:
    params: dict[str, Any] = {
        "metric": "exact_recovery",
        "margin": MARGIN,
        "higher_is_better": True,
        "paired_binary": True,
    }
    params.update(kwargs)
    return equivalence_verdict(treatment, control, **params)


# --------------------------------------------------------------------------
# The statistic is what it claims to be
# --------------------------------------------------------------------------


@pytest.mark.parametrize(("n", "b", "c"), [(20, 5, 2), (50, 9, 14), (21, 1, 7), (30, 3, 3)])
def test_reduces_to_mcnemar_at_no_difference(n: int, b: int, c: int) -> None:
    """At a difference of zero the score statistic *is* McNemar's.

    This is the check that the constrained estimate inside ``_tango_score`` was
    derived correctly rather than transcribed from memory: at zero it must be
    the pooled discordant rate, and the whole expression must collapse to
    ``(b - c) / sqrt(b + c)``. Nothing else about the implementation has a
    closed form to compare against.
    """
    assert _tango_score(n, b, c, 0.0) == pytest.approx((b - c) / np.sqrt(b + c))


def test_interval_brackets_the_observed_difference() -> None:
    low, high = _tango_interval(40, 10, 2, 0.90)
    assert low < (10 - 2) / 40 < high


def test_interval_narrows_as_pairs_accumulate() -> None:
    """More evidence, tighter interval -- at a fixed observed difference."""
    narrow = _tango_interval(200, 20, 20, 0.90)
    wide = _tango_interval(20, 2, 2, 0.90)
    assert (narrow[1] - narrow[0]) < (wide[1] - wide[0])


def test_interval_covers_at_its_nominal_rate() -> None:
    """Simulated coverage of the 90% interval, over a spread of true differences.

    Stochastic and slow-ish, but it is the only check here that would catch an
    interval that is self-consistent and still wrong. The tolerance is wide
    enough that Monte Carlo error cannot fail it and narrow enough that a
    genuinely miscalibrated interval cannot pass.
    """
    rng = np.random.default_rng(7)
    for p11, p12, p21 in [(0.7, 0.15, 0.15), (0.5, 0.30, 0.10), (0.4, 0.05, 0.30)]:
        p22 = 1.0 - p11 - p12 - p21
        delta = p12 - p21
        covered = 0
        trials = 600
        for _ in range(trials):
            counts = rng.multinomial(30, [p11, p12, p21, p22])
            low, high = _tango_interval(30, int(counts[1]), int(counts[2]), 0.90)
            covered += low <= delta <= high
        assert 0.85 <= covered / trials <= 0.97


# --------------------------------------------------------------------------
# The defect this branch exists to fix
# --------------------------------------------------------------------------


def test_total_agreement_is_inconclusive_not_null() -> None:
    """Twenty agreeing pairs do not establish equivalence within ten points.

    This is the regression that motivated the change. The continuous branch
    returns a zero-width interval here and therefore ``NULL``, with ``power``
    reported as 1.00 -- certainty from a sample that saw no disagreement at all.
    The honest interval is about +/-0.119, which does not fit inside 0.10.
    """
    treatment, control = arms(20, better=0, worse=0)

    collapsed = verdict_for(treatment, control, paired_binary=False)
    assert collapsed.verdict is Verdict.NULL
    assert collapsed.ci_low == collapsed.ci_high == 0.0

    honest = verdict_for(treatment, control)
    assert honest.verdict is Verdict.INCONCLUSIVE
    assert honest.ci_low == pytest.approx(-0.119, abs=0.005)
    assert honest.ci_high == pytest.approx(0.119, abs=0.005)


def test_total_agreement_reports_the_seeds_it_would_have_needed() -> None:
    """``INCONCLUSIVE`` from no disagreement still carries an instruction.

    Power here must not be computed from the observed discordant rate, which is
    zero and would report perfect power. It uses the rule-of-three upper bound,
    so the verdict says how many seeds the question needs rather than claiming
    it was already answered.
    """
    verdict = verdict_for(*arms(20, better=0, worse=0))
    assert verdict.power < 0.5
    assert verdict.n_for_target_power is not None
    assert verdict.n_for_target_power > 20


def test_equivalence_is_reachable_with_enough_agreeing_pairs() -> None:
    """The branch must still be able to say ``NULL``, or it is merely pessimistic.

    A test that could only ever return ``INCONCLUSIVE`` would be safe and
    useless. At 400 pairs with a few disagreements each way the interval does
    fit inside the margin, and the positive finding stays available.
    """
    verdict = verdict_for(*arms(400, better=4, worse=4))
    assert verdict.verdict is Verdict.NULL
    assert verdict.p_value is not None
    assert verdict.p_value < 0.05


# --------------------------------------------------------------------------
# Orientation, direction, and declaration
# --------------------------------------------------------------------------


def test_a_clear_win_reads_as_contributes() -> None:
    verdict = verdict_for(*arms(100, better=40, worse=2))
    assert verdict.verdict is Verdict.CONTRIBUTES
    assert verdict.observed_difference > 0


def test_a_clear_loss_reads_as_harmful() -> None:
    verdict = verdict_for(*arms(100, better=2, worse=40))
    assert verdict.verdict is Verdict.HARMFUL
    assert verdict.observed_difference < 0


def test_orientation_swaps_the_two_difference_verdicts() -> None:
    """The same table read as a loss metric must invert, not merely shift."""
    treatment, control = arms(100, better=40, worse=2)
    assert verdict_for(treatment, control).verdict is Verdict.CONTRIBUTES
    assert verdict_for(treatment, control, higher_is_better=False).verdict is Verdict.HARMFUL


def test_p_value_agrees_with_the_verdict_it_accompanies() -> None:
    """A claim's p-value must clear alpha whenever the interval established it.

    The interval decides the verdict and the score test supplies the p-value, so
    they are separate computations that must not be allowed to disagree -- a
    ``CONTRIBUTES`` beside a p-value of 0.8 would make the record incoherent.
    """
    for treatment, control in (arms(100, 40, 2), arms(100, 2, 40), arms(400, 4, 4)):
        verdict = verdict_for(treatment, control)
        assert verdict.p_value is not None
        assert verdict.p_value < 0.05


def test_inconclusive_carries_no_p_value() -> None:
    """Nothing was claimed, so there is nothing for the correction to weigh."""
    assert verdict_for(*arms(20, better=0, worse=0)).p_value is None


def test_the_test_used_is_recorded_on_the_verdict() -> None:
    """A reader comparing two metrics must see that they were not tested alike."""
    assert "Tango" in verdict_for(*arms(40, 8, 4)).test
    assert "bootstrap" in verdict_for(*arms(40, 8, 4), paired_binary=False).test


def test_non_binary_values_are_rejected_rather_than_coerced() -> None:
    """Declaring a continuous metric binary is a design error, not a rounding job.

    Silently thresholding would produce a verdict that looks ordinary and
    answers a question nobody asked.
    """
    with pytest.raises(ValueError, match="paired-binary"):
        verdict_for([0.0, 0.4, 1.0], [0.0, 1.0, 1.0])
