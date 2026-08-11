"""Tests for the equivalence statistics (ENG-0001 section 6).

The module under test decides whether a wrapper contributed anything. Its
dangerous failure is not crashing -- it is returning a confident, plausible,
wrong verdict, which is the exact error RFC-0001 exists to prevent. Two
properties therefore get the most attention here:

* An underpowered comparison must never be certified ``NULL``. "We could not
  tell" and "there was no effect" are different claims.
* Orientation must not be silently inverted. A metric read the wrong way round
  swaps ``CONTRIBUTES`` and ``HARMFUL`` while leaving every number plausible.
"""

from __future__ import annotations

import numpy as np
import pytest

from engine.audit import MetricVerdict, Verdict, equivalence_verdict

MARGIN = 0.5


def arms(effect: float, spread: float, n: int, seed: int = 0) -> tuple[list[float], list[float]]:
    """Paired arms whose treatment sits ``effect`` above control on a loss metric.

    A positive ``effect`` means the treatment's loss is *higher*, which for a
    loss metric means it did worse.
    """
    rng = np.random.default_rng(seed)
    control = rng.normal(10.0, spread, n)
    treatment = control + effect + rng.normal(0.0, spread, n)
    return list(treatment), list(control)


def verdict_for(treatment: list[float], control: list[float], **kwargs: object) -> MetricVerdict:
    params: dict = {"metric": "loss", "margin": MARGIN}
    params.update(kwargs)
    return equivalence_verdict(treatment, control, **params)  # type: ignore[arg-type]


# --------------------------------------------------------------------------
# The four verdicts
# --------------------------------------------------------------------------


def test_identical_arms_are_null() -> None:
    values = [1.0, 2.0, 3.0, 4.0, 5.0]
    result = verdict_for(values, list(values))
    assert result.verdict is Verdict.NULL
    assert result.observed_difference == 0.0


def test_large_improvement_is_contributes() -> None:
    """Treatment loss far below control, on a metric where lower is better."""
    treatment, control = arms(effect=-5.0, spread=0.3, n=40)
    assert verdict_for(treatment, control).verdict is Verdict.CONTRIBUTES


def test_large_regression_is_harmful() -> None:
    treatment, control = arms(effect=5.0, spread=0.3, n=40)
    assert verdict_for(treatment, control).verdict is Verdict.HARMFUL


def test_small_sample_is_inconclusive_not_null() -> None:
    """The central guarantee: absence of evidence is not evidence of absence.

    A true zero effect measured with three noisy seeds must not be certified as
    equivalent. A difference test would report p > 0.05 here and invite exactly
    that conclusion.
    """
    treatment, control = arms(effect=0.0, spread=5.0, n=3)
    result = verdict_for(treatment, control)
    assert result.verdict is Verdict.INCONCLUSIVE
    assert result.verdict is not Verdict.NULL


def test_null_requires_ci_inside_margin() -> None:
    """An interval overlapping the margin is inconclusive, not equivalent.

    Constructed rather than sampled: the per-seed differences are large and
    alternate in sign, so the mean is exactly zero while the interval is far
    too wide to establish equivalence. A procedure keying on the point estimate
    would call this NULL.
    """
    control = [10.0] * 8
    treatment = [13.0, 7.0, 13.0, 7.0, 13.0, 7.0, 13.0, 7.0]

    result = verdict_for(treatment, control)

    assert result.observed_difference == pytest.approx(0.0)
    assert result.ci_low <= -MARGIN or result.ci_high >= MARGIN
    assert result.verdict is Verdict.INCONCLUSIVE


def test_well_powered_zero_effect_is_null() -> None:
    treatment, control = arms(effect=0.0, spread=0.2, n=50)
    result = verdict_for(treatment, control)
    assert result.verdict is Verdict.NULL
    assert -MARGIN < result.ci_low
    assert result.ci_high < MARGIN


# --------------------------------------------------------------------------
# Orientation
# --------------------------------------------------------------------------


def test_orientation_flips_contributes_and_harmful() -> None:
    """Same numbers, opposite metric direction, opposite verdict."""
    treatment, control = arms(effect=-5.0, spread=0.3, n=40)

    as_loss = verdict_for(treatment, control, higher_is_better=False)
    as_score = verdict_for(treatment, control, higher_is_better=True)

    assert as_loss.verdict is Verdict.CONTRIBUTES
    assert as_score.verdict is Verdict.HARMFUL
    assert as_loss.observed_difference == pytest.approx(-as_score.observed_difference)


def test_observed_difference_is_positive_when_treatment_wins() -> None:
    """Positive always means the treatment did better, whichever way the metric runs."""
    treatment, control = arms(effect=-3.0, spread=0.2, n=30)
    assert verdict_for(treatment, control, higher_is_better=False).observed_difference > 0


# --------------------------------------------------------------------------
# Power
# --------------------------------------------------------------------------


def test_power_reported_on_every_verdict() -> None:
    cases = [
        arms(effect=0.0, spread=0.2, n=50),
        arms(effect=-5.0, spread=0.3, n=40),
        arms(effect=5.0, spread=0.3, n=40),
        arms(effect=0.0, spread=5.0, n=3),
    ]
    seen = set()
    for treatment, control in cases:
        result = verdict_for(treatment, control)
        assert 0.0 <= result.power <= 1.0
        seen.add(result.verdict)
    assert len(seen) == 4, f"expected all four verdicts, saw {seen}"


def test_underpowered_comparison_reports_low_power() -> None:
    """Power is what distinguishes a trustworthy NULL from a lucky one."""
    noisy = verdict_for(*arms(effect=0.0, spread=5.0, n=3))
    tight = verdict_for(*arms(effect=0.0, spread=0.2, n=50))
    assert noisy.power < tight.power
    assert tight.power > 0.9


# --------------------------------------------------------------------------
# Determinism and provenance fields
# --------------------------------------------------------------------------


def test_deterministic_for_fixed_seed() -> None:
    """A verdict that moved between runs could not be published beside a number."""
    treatment, control = arms(effect=0.3, spread=1.0, n=20)
    assert verdict_for(treatment, control) == verdict_for(treatment, control)


def test_record_carries_the_evidence_for_its_claim() -> None:
    treatment, control = arms(effect=0.0, spread=0.2, n=50)
    result = verdict_for(treatment, control, metric="best_loss")
    assert result.metric == "best_loss"
    assert result.margin == MARGIN
    assert result.n == 50
    assert result.higher_is_better is False
    assert result.ci_low <= result.observed_difference <= result.ci_high


# --------------------------------------------------------------------------
# Rejected input
# --------------------------------------------------------------------------


def test_mismatched_lengths_raise() -> None:
    with pytest.raises(ValueError, match="paired by seed"):
        verdict_for([1.0, 2.0, 3.0], [1.0, 2.0])


def test_single_pair_raises() -> None:
    with pytest.raises(ValueError, match="at least 2 paired observations"):
        verdict_for([1.0], [1.0])


@pytest.mark.parametrize("margin", [0.0, -1.0])
def test_non_positive_margin_raises(margin: float) -> None:
    with pytest.raises(ValueError, match="margin must be positive"):
        verdict_for([1.0, 2.0], [1.0, 2.0], margin=margin)


@pytest.mark.parametrize("confidence", [0.0, 1.0, 1.5, -0.1])
def test_confidence_out_of_range_raises(confidence: float) -> None:
    with pytest.raises(ValueError, match="confidence must lie"):
        verdict_for([1.0, 2.0], [1.0, 2.0], confidence=confidence)


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_non_finite_input_raises(bad: float) -> None:
    with pytest.raises(ValueError, match=r"treatment\[1\] is not finite"):
        verdict_for([1.0, bad, 3.0], [1.0, 2.0, 3.0])


def test_non_finite_control_names_the_control_arm() -> None:
    with pytest.raises(ValueError, match=r"control\[0\] is not finite"):
        verdict_for([1.0, 2.0], [float("nan"), 2.0])


# --------------------------------------------------------------------------
# Degenerate but valid input
# --------------------------------------------------------------------------


def test_zero_variance_identical_arms() -> None:
    """Constant arms are degenerate for a bootstrap but are a legitimate result."""
    result = verdict_for([3.0, 3.0, 3.0, 3.0], [3.0, 3.0, 3.0, 3.0])
    assert result.verdict is Verdict.NULL
    assert result.ci_low == result.ci_high == 0.0
    assert result.power == 1.0


def test_zero_variance_constant_offset_beyond_margin() -> None:
    """Every seed differs by the same amount: a point interval outside the margin."""
    result = verdict_for([5.0, 5.0, 5.0, 5.0], [3.0, 3.0, 3.0, 3.0])
    assert result.verdict is Verdict.HARMFUL
    assert result.ci_low == result.ci_high == pytest.approx(-2.0)


# --------------------------------------------------------------------------
# Ground truth
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("effect", "expected"),
    [
        (0.0, Verdict.NULL),
        (-2 * MARGIN, Verdict.CONTRIBUTES),
        (2 * MARGIN, Verdict.HARMFUL),
    ],
)
def test_known_ground_truth_recovery(effect: float, expected: Verdict) -> None:
    """Constructed cases where the correct verdict is known in advance.

    ADR-0001's first revisit criterion requires exactly this: the audit must be
    shown to return the right answer on a contributing, a null, and a harmful
    wrapper before it can be considered for a blocking gate.
    """
    treatment, control = arms(effect=effect, spread=0.15, n=50)
    assert verdict_for(treatment, control).verdict is expected
