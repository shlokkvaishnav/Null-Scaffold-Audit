"""Tests for `boundary_clearance_ratio` (issue #23).

`power`/`n_for_target_power` answer an equivalence question and are the
right lens for `NULL`/`INCONCLUSIVE`. `boundary_clearance_ratio` answers
the different question those two verdicts don't need and `CONTRIBUTES`/
`HARMFUL` do: how decisively did the interval clear its margin, in units of
the interval's own width. Direct unit tests exercise the formula exactly
(mirroring `test_audit_paired_binary.py`'s precedent for testing a private
statistics helper directly); integration tests go through the public
`equivalence_verdict()`/`audit()` surface, including the calibration
scaffolds `engine/audit/calibration.py` was built to validate instruments
with -- exactly SPEC.md's "Experimental design".
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
    equivalence_verdict,
)
from engine.audit.statistics import _boundary_clearance_ratio

# --------------------------------------------------------------------------
# The formula itself, exactly
# --------------------------------------------------------------------------


def test_none_for_null() -> None:
    assert _boundary_clearance_ratio(-0.5, 0.5, margin=1.0, verdict=Verdict.NULL) is None


def test_none_for_inconclusive() -> None:
    assert _boundary_clearance_ratio(-1.5, 1.5, margin=1.0, verdict=Verdict.INCONCLUSIVE) is None


def test_none_on_a_zero_width_interval() -> None:
    """The denominator vanishes; there is no ratio to report, not a zero one."""
    assert _boundary_clearance_ratio(5.0, 5.0, margin=1.0, verdict=Verdict.CONTRIBUTES) is None


def test_contributes_uses_ci_low_as_the_deciding_bound() -> None:
    # ci_low=3, margin=1 -> clearance=2; width=(6-3)=3 -> ratio=2/3
    ratio = _boundary_clearance_ratio(3.0, 6.0, margin=1.0, verdict=Verdict.CONTRIBUTES)
    assert ratio == pytest.approx(2.0 / 3.0)


def test_harmful_uses_negated_ci_high_as_the_deciding_bound() -> None:
    # ci_high=-3, margin=1 -> bound=3, clearance=2; width=(-3-(-6))=3 -> ratio=2/3
    ratio = _boundary_clearance_ratio(-6.0, -3.0, margin=1.0, verdict=Verdict.HARMFUL)
    assert ratio == pytest.approx(2.0 / 3.0)


def test_ratio_is_symmetric_between_contributes_and_a_mirrored_harmful_case() -> None:
    """The formula must not silently favour one verdict's sign."""
    contributes = _boundary_clearance_ratio(3.0, 6.0, margin=1.0, verdict=Verdict.CONTRIBUTES)
    harmful = _boundary_clearance_ratio(-6.0, -3.0, margin=1.0, verdict=Verdict.HARMFUL)
    assert contributes == pytest.approx(harmful)


def test_ratio_is_near_zero_when_the_bound_barely_clears_the_margin() -> None:
    # ci_low=1.01, margin=1.0 -> clearance=0.01; width=2.0 -> ratio=0.005
    ratio = _boundary_clearance_ratio(1.01, 3.01, margin=1.0, verdict=Verdict.CONTRIBUTES)
    assert ratio is not None
    assert ratio == pytest.approx(0.005)
    assert 0.0 < ratio < 0.05


def test_ratio_grows_as_the_bound_moves_further_past_the_margin() -> None:
    near = _boundary_clearance_ratio(1.1, 3.1, margin=1.0, verdict=Verdict.CONTRIBUTES)
    far = _boundary_clearance_ratio(3.0, 5.0, margin=1.0, verdict=Verdict.CONTRIBUTES)
    assert near is not None
    assert far is not None
    assert far > near


# --------------------------------------------------------------------------
# Wired into equivalence_verdict()
# --------------------------------------------------------------------------


def test_populated_on_a_clear_contributes() -> None:
    rng = np.random.default_rng(0)
    treatment = list(10.0 + rng.normal(0, 0.2, size=40))
    control = list(np.zeros(40) + rng.normal(0, 0.2, size=40))
    result = equivalence_verdict(treatment, control, metric="m", margin=1.0, higher_is_better=True)
    assert result.verdict is Verdict.CONTRIBUTES
    assert result.boundary_clearance_ratio is not None
    assert result.boundary_clearance_ratio > 0.0


def test_populated_on_a_clear_harmful() -> None:
    rng = np.random.default_rng(1)
    treatment = list(np.zeros(40) + rng.normal(0, 0.2, size=40))
    control = list(10.0 + rng.normal(0, 0.2, size=40))
    result = equivalence_verdict(treatment, control, metric="m", margin=1.0, higher_is_better=True)
    assert result.verdict is Verdict.HARMFUL
    assert result.boundary_clearance_ratio is not None
    assert result.boundary_clearance_ratio > 0.0


def test_none_on_a_real_null_verdict() -> None:
    rng = np.random.default_rng(2)
    treatment = list(rng.normal(0, 1.0, size=40))
    control = list(rng.normal(0, 1.0, size=40))
    result = equivalence_verdict(treatment, control, metric="m", margin=5.0, higher_is_better=True)
    assert result.verdict is Verdict.NULL
    assert result.boundary_clearance_ratio is None


def test_none_on_a_real_inconclusive_verdict() -> None:
    rng = np.random.default_rng(3)
    treatment = list(rng.normal(0, 3.0, size=6))
    control = list(rng.normal(0, 3.0, size=6))
    result = equivalence_verdict(treatment, control, metric="m", margin=0.5, higher_is_better=True)
    assert result.verdict is Verdict.INCONCLUSIVE
    assert result.boundary_clearance_ratio is None


def test_a_low_but_nonzero_ratio_is_not_a_low_power_reading() -> None:
    """The exact confusion issue #23 documents: a decisive CONTRIBUTES can
    report low TOST-for-NULL `power` while its own boundary_clearance_ratio
    is unambiguously positive -- the two fields are not redundant."""
    rng = np.random.default_rng(4)
    treatment = list(10.0 + rng.normal(0, 0.5, size=30))
    control = list(np.zeros(30) + rng.normal(0, 0.5, size=30))
    result = equivalence_verdict(treatment, control, metric="m", margin=2.0, higher_is_better=True)
    assert result.verdict is Verdict.CONTRIBUTES
    assert result.boundary_clearance_ratio is not None
    assert result.boundary_clearance_ratio > 0.0
    # power here answers "would this establish equivalence", which a real
    # effect this size does not -- the two fields disagree by design.


# --------------------------------------------------------------------------
# Growing more decisive with n, on the calibration scaffolds themselves --
# mirrors SPEC.md's "Experimental design": a barely-resolved case at low n,
# a decisive one at higher n, on the same known-answer construction.
# --------------------------------------------------------------------------

MARGIN = 1.0


@dataclass
class FakeSearcher:
    """Deterministic-quality searcher with a noisy selection proxy, matching
    tests/test_audit_calibration.py's own fixture exactly."""

    restart_cost: int = 100
    proxy_noise: float = 1.0
    quality_sd: float = 2.0

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


def _run(scaffold: Any, seeds: list[int]) -> Any:
    return audit(
        scaffold,
        problem=None,
        seeds=seeds,
        margins={"rmse": MARGIN},
        higher_is_better={"rmse": False},
    )


def test_oracle_scaffold_ratio_grows_from_barely_resolved_to_decisive_with_n() -> None:
    """The empirical validation issue #23 requires: a metric that only ever
    says 'yes, decisive' on cases already known to be robust hasn't been
    tested. RESTARTS/seed counts below are chosen (not tuned after seeing
    results) to land the small-n case close to its margin -- a small
    oracle-selection sample gives the interval a lot of width to still
    clear -- and the large-n case comfortably past it."""
    base = FakeSearcher(proxy_noise=12.0)  # calibration.py's own ORACLE_PROXY_NOISE
    scaffold = OracleScaffold(base=base, restarts=3)

    small = _run(scaffold, seeds=list(range(20)))
    large = _run(scaffold, seeds=list(range(200)))

    assert small.verdict is Verdict.CONTRIBUTES
    assert large.verdict is Verdict.CONTRIBUTES
    small_ratio = small.per_metric["rmse"].boundary_clearance_ratio
    large_ratio = large.per_metric["rmse"].boundary_clearance_ratio
    assert small_ratio is not None and large_ratio is not None
    assert large_ratio > small_ratio


def test_wasteful_scaffold_ratio_is_present_and_positive_on_harmful() -> None:
    base = FakeSearcher(proxy_noise=0.5)  # calibration.py's own WASTEFUL_PROXY_NOISE
    scaffold = WastefulScaffold(base=base, restarts=3)
    report = _run(scaffold, seeds=list(range(40)))
    assert report.verdict is Verdict.HARMFUL
    ratio = report.per_metric["rmse"].boundary_clearance_ratio
    assert ratio is not None and ratio > 0.0


def test_null_scaffold_ratio_is_none() -> None:
    base = FakeSearcher(proxy_noise=1.0)
    scaffold = NullScaffold(base=base, restarts=3)
    report = _run(scaffold, seeds=list(range(40)))
    assert report.verdict is Verdict.NULL
    assert report.per_metric["rmse"].boundary_clearance_ratio is None


# --------------------------------------------------------------------------
# The invariant survives correction: Holm downgrade and the vacuous guard
# --------------------------------------------------------------------------


def test_ratio_is_cleared_when_holm_correction_downgrades_to_inconclusive() -> None:
    """A CONTRIBUTES/HARMFUL claim that does not survive Holm correction
    becomes INCONCLUSIVE -- its boundary_clearance_ratio must not survive
    with it, or the field's own documented invariant (None unless verdict
    is CONTRIBUTES/HARMFUL) would be false on exactly the rows a reader is
    most likely to inspect closely. Constructed directly against
    `_holm_correct` (mirroring test_audit_paired_binary.py's precedent for
    testing a private statistics/correction helper exactly) rather than via
    a full audit(), so the p-values that decide which claim survives are
    exact and not left to sampling luck."""
    from engine.audit.arms import _holm_correct
    from engine.audit.verdict import MetricVerdict

    weak = MetricVerdict(
        metric="weak",
        verdict=Verdict.CONTRIBUTES,
        observed_difference=1.01,
        ci_low=1.001,
        ci_high=1.02,
        margin=1.0,
        power=0.5,
        n=10,
        higher_is_better=True,
        p_value=0.04,  # survives alone at alpha=0.05, but not after Holm with `strong`
        boundary_clearance_ratio=_boundary_clearance_ratio(1.001, 1.02, 1.0, Verdict.CONTRIBUTES),
    )
    strong = MetricVerdict(
        metric="strong",
        verdict=Verdict.CONTRIBUTES,
        observed_difference=100.0,
        ci_low=90.0,
        ci_high=110.0,
        margin=1.0,
        power=0.99,
        n=10,
        higher_is_better=True,
        p_value=1e-12,
        boundary_clearance_ratio=_boundary_clearance_ratio(90.0, 110.0, 1.0, Verdict.CONTRIBUTES),
    )
    assert weak.boundary_clearance_ratio is not None  # sanity: was populated before correction

    corrected = _holm_correct({"weak": weak, "strong": strong}, alpha=0.025)

    assert corrected["strong"].verdict is Verdict.CONTRIBUTES
    assert corrected["strong"].boundary_clearance_ratio is not None
    assert corrected["weak"].verdict is Verdict.INCONCLUSIVE  # Holm-corrected away
    assert corrected["weak"].boundary_clearance_ratio is None


def test_ratio_is_cleared_by_the_vacuous_comparison_guard() -> None:
    """A verdict withdrawn to INCONCLUSIVE because both arms agreed on every
    seed must not keep a boundary_clearance_ratio from before the
    withdrawal -- same invariant, the other downgrade path."""
    from engine.audit.arms import ArmOutcomes, _guard_vacuous_comparison
    from engine.audit.verdict import MetricVerdict

    verdict = MetricVerdict(
        metric="m",
        verdict=Verdict.CONTRIBUTES,
        observed_difference=5.0,
        ci_low=4.0,
        ci_high=6.0,
        margin=1.0,
        power=0.9,
        n=8,
        higher_is_better=True,
        boundary_clearance_ratio=_boundary_clearance_ratio(4.0, 6.0, 1.0, Verdict.CONTRIBUTES),
    )
    identical = SearchOutcome(metrics={"m": 1.0}, evaluations_used=1, representation="same")
    arms = ArmOutcomes(
        treatment=[identical] * 3,
        control=[identical] * 3,
        seeds=[0, 1, 2],
        restarts_per_seed=1,
        treatment_evaluations=3,
        control_evaluations=3,
    )
    withdrawn, vacuous = _guard_vacuous_comparison({"m": verdict}, arms)
    assert vacuous
    assert withdrawn["m"].verdict is Verdict.INCONCLUSIVE
    assert withdrawn["m"].boundary_clearance_ratio is None
