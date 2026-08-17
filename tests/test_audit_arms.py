"""Tests for budget-matched two-arm execution.

Fakes stand in for real searchers throughout. The point of these tests is the
*accounting* -- that the control arm gets exactly the compute the treatment
spent, and not a unit more -- which is testable against known ground truth only
when the searcher's cost is something the test fixes rather than measures.

The dangerous failure here is a control arm that is quietly over- or
under-funded. Either produces intervals that look entirely reasonable and mean
the wrong thing.
"""

from __future__ import annotations

import pytest

from engine.audit import (
    Budget,
    NotSeparableError,
    SearchOutcome,
    Verdict,
    audit,
    run_arms,
)

MARGINS = {"loss": 0.5}


class FakeBase:
    """A primitive whose result is a deterministic function of its seed."""

    def __init__(self, restart_cost: int = 100, offset: float = 0.0) -> None:
        self.restart_cost = restart_cost
        self.offset = offset
        self.seeds_seen: list[int] = []

    def search(self, problem: object, seed: int) -> SearchOutcome:
        self.seeds_seen.append(seed)
        loss = self.offset + (seed % 7) * 0.1
        return SearchOutcome(
            metrics={"loss": loss, "selection_score": -loss},
            evaluations_used=self.restart_cost,
            representation=f"candidate-{seed % 7}",
        )

    def select(self, outcomes):
        return max(outcomes, key=lambda o: o.metrics["selection_score"])


class FakeScaffold:
    """A wrapper that spends `fits * restart_cost` and returns a fixed outcome."""

    def __init__(self, base: FakeBase, fits: int = 3, loss: float = 0.3, rep: str = "scaffolded"):
        self.name = "FakeScaffold"
        self._base = base
        self.fits = fits
        self.loss = loss
        self.rep = rep

    def unwrap(self) -> FakeBase:
        return self._base

    def run(self, problem: object, seed: int) -> SearchOutcome:
        return SearchOutcome(
            metrics={"loss": self.loss, "selection_score": -self.loss},
            evaluations_used=self.fits * self._base.restart_cost,
            representation=self.rep,
        )


class UnseparableScaffold:
    name = "Unseparable"

    def unwrap(self):
        raise NotSeparableError("end-to-end model has no inner searcher")

    def run(self, problem: object, seed: int) -> SearchOutcome:  # pragma: no cover - never reached
        raise AssertionError("run must not be called when unwrap fails")


# --------------------------------------------------------------------------
# Budget matching -- the property the audit's fairness rests on
# --------------------------------------------------------------------------


def test_control_budget_matches_treatment_exactly() -> None:
    base = FakeBase(restart_cost=100)
    arms = run_arms(FakeScaffold(base, fits=3), problem=None, seeds=[0, 1, 2])
    assert arms.treatment_evaluations == arms.control_evaluations


def test_restarts_are_treatment_spend_divided_by_restart_cost() -> None:
    base = FakeBase(restart_cost=250)
    arms = run_arms(FakeScaffold(base, fits=4), problem=None, seeds=[0, 1])
    assert arms.restarts_per_seed == 4


def test_restart_count_floors_rather_than_rounds_up() -> None:
    """A control arm may never receive more compute than the treatment spent."""
    base = FakeBase(restart_cost=100)
    arms = run_arms(FakeScaffold(base, fits=1), problem=None, seeds=[0])
    assert arms.restarts_per_seed == 1
    assert arms.control_evaluations <= arms.treatment_evaluations


def test_at_least_one_restart_even_when_treatment_underspends() -> None:
    base = FakeBase(restart_cost=1000)
    arms = run_arms(FakeScaffold(base, fits=0), problem=None, seeds=[0])
    assert arms.restarts_per_seed == 1


def test_control_arm_uses_distinct_seeds_per_restart() -> None:
    """Restarts must be independent; repeating a seed is the defect being audited."""
    base = FakeBase(restart_cost=100)
    run_arms(FakeScaffold(base, fits=5), problem=None, seeds=[0, 1])
    assert len(base.seeds_seen) == len(set(base.seeds_seen))


def test_control_arm_is_charged_for_losing_restarts() -> None:
    base = FakeBase(restart_cost=100)
    arms = run_arms(FakeScaffold(base, fits=3), problem=None, seeds=[0])
    assert arms.control[0].evaluations_used == 300


# --------------------------------------------------------------------------
# Separability
# --------------------------------------------------------------------------


def test_unseparable_pipeline_raises_before_running_either_arm() -> None:
    with pytest.raises(NotSeparableError):
        run_arms(UnseparableScaffold(), problem=None, seeds=[0, 1])


def test_no_seeds_raises() -> None:
    with pytest.raises(ValueError, match="at least one seed"):
        run_arms(FakeScaffold(FakeBase()), problem=None, seeds=[])


# --------------------------------------------------------------------------
# Reporting surface
# --------------------------------------------------------------------------


def test_identical_representation_rate_detects_agreement() -> None:
    base = FakeBase(restart_cost=100)
    # The scaffold returns exactly what a single restart at seed 0 returns.
    arms = run_arms(FakeScaffold(base, fits=1, rep="candidate-0"), problem=None, seeds=[0])
    assert arms.identical_representation_rate == 1.0


def test_identical_representation_rate_is_zero_when_arms_differ() -> None:
    base = FakeBase(restart_cost=100)
    arms = run_arms(FakeScaffold(base, fits=2, rep="something-else"), problem=None, seeds=[0, 1])
    assert arms.identical_representation_rate == 0.0


def test_map_fn_is_the_parallelism_seam() -> None:
    calls = {"n": 0}

    def counting_map(fn, items):
        calls["n"] += 1
        return [fn(i) for i in items]

    base = FakeBase(restart_cost=100)
    run_arms(FakeScaffold(base, fits=2), problem=None, seeds=[0, 1], map_fn=counting_map)
    assert calls["n"] == 2  # one pass per arm


# --------------------------------------------------------------------------
# Verdict aggregation
# --------------------------------------------------------------------------


def test_audit_requires_pre_registered_margins() -> None:
    with pytest.raises(ValueError, match="margins are required"):
        audit(FakeScaffold(FakeBase()), problem=None, seeds=[0, 1], margins={})


def test_audit_rejects_metric_absent_from_outcomes() -> None:
    with pytest.raises(KeyError, match="absent_metric"):
        audit(
            FakeScaffold(FakeBase()),
            problem=None,
            seeds=[0, 1],
            margins={"absent_metric": 1.0},
        )


def test_equal_arms_report_null() -> None:
    """Scaffold returns exactly what the control's single restart returns."""
    base = FakeBase(restart_cost=100, offset=0.0)
    scaffold = FakeScaffold(base, fits=1, loss=0.0, rep="candidate-0")
    report = audit(scaffold, problem=None, seeds=list(range(20)), margins=MARGINS)
    assert report.verdict is Verdict.NULL


def test_clearly_worse_scaffold_is_harmful() -> None:
    base = FakeBase(restart_cost=100, offset=0.0)
    scaffold = FakeScaffold(base, fits=1, loss=5.0)
    report = audit(scaffold, problem=None, seeds=list(range(20)), margins=MARGINS)
    assert report.verdict is Verdict.HARMFUL


def test_clearly_better_scaffold_contributes() -> None:
    base = FakeBase(restart_cost=100, offset=10.0)
    scaffold = FakeScaffold(base, fits=1, loss=0.0)
    report = audit(scaffold, problem=None, seeds=list(range(20)), margins=MARGINS)
    assert report.verdict is Verdict.CONTRIBUTES


def test_report_carries_its_limitations() -> None:
    """A report claiming no limitations is treated as incomplete."""
    base = FakeBase(restart_cost=100)
    report = audit(FakeScaffold(base, fits=2), problem=None, seeds=list(range(10)), margins=MARGINS)
    assert report.limitations
    assert any("Holm" in item for item in report.limitations)
    assert report.scaffold == "FakeScaffold"


# --------------------------------------------------------------------------
# Budget validation
# --------------------------------------------------------------------------


@pytest.mark.parametrize("evaluations", [0, -1])
def test_non_positive_budget_raises(evaluations: int) -> None:
    with pytest.raises(ValueError, match="budget must be positive"):
        Budget(evaluations=evaluations)


def test_budget_accepts_positive_value() -> None:
    assert Budget(evaluations=1).evaluations == 1


def test_mixed_result_is_reported_by_its_harm() -> None:
    """A wrapper better on one metric and worse on another must not read as a win.

    The first real sweep produced this exact shape. Reporting it as CONTRIBUTES
    buried a significant regression beneath a significant improvement, which is
    the one direction of error this tool may not make.
    """
    base = FakeBase(restart_cost=100, offset=0.0)

    class MixedScaffold(FakeScaffold):
        def run(self, problem: object, seed: int) -> SearchOutcome:
            return SearchOutcome(
                metrics={"loss": 5.0, "size": 0.0, "selection_score": 0.0},
                evaluations_used=self._base.restart_cost,
                representation="mixed",
            )

    def base_search(problem: object, seed: int) -> SearchOutcome:
        return SearchOutcome(
            metrics={"loss": 0.0, "size": 5.0, "selection_score": 0.0},
            evaluations_used=100,
            representation=f"c-{seed}",
        )

    base.search = base_search  # type: ignore[method-assign]
    report = audit(
        MixedScaffold(base, fits=1),
        problem=None,
        seeds=list(range(20)),
        margins={"loss": 0.5, "size": 0.5},
    )
    assert report.per_metric["loss"].verdict is Verdict.HARMFUL
    assert report.per_metric["size"].verdict is Verdict.CONTRIBUTES
    assert report.verdict is Verdict.HARMFUL


# --------------------------------------------------------------------------
# A comparison that never happened
# --------------------------------------------------------------------------


class MirrorScaffold:
    """A scaffold that reproduces the control arm exactly, seeds included.

    This is what common random numbers turned the real scaffold into: it derived
    iteration `i`'s seed by the same rule the control used for restart `i`, so
    both arms ran the same searches and selected among them with the same rule.
    """

    def __init__(self, base: FakeBase, fits: int = 3) -> None:
        self.name = "MirrorScaffold"
        self._base = base
        self.fits = fits

    def unwrap(self) -> FakeBase:
        return self._base

    def run(self, problem: object, seed: int) -> SearchOutcome:
        outcomes = [
            self._base.search(problem, seed * 1_000_003 + restart) for restart in range(self.fits)
        ]
        best = self._base.select(outcomes)
        return SearchOutcome(
            metrics=best.metrics,
            evaluations_used=sum(o.evaluations_used for o in outcomes),
            representation=best.representation,
        )


def test_arms_that_agree_on_every_seed_report_no_verdict() -> None:
    """Identical output on every seed is not equivalence -- it is not a comparison.

    Left alone this is the audit's strongest possible claim reached by its
    weakest possible evidence: every paired difference is exactly zero, the
    interval collapses to a point, a point sits inside any margin, and NULL falls
    out with power 1.00. A real sweep produced exactly this on eight problems out
    of eight.
    """
    base = FakeBase()
    report = audit(MirrorScaffold(base), problem=None, seeds=list(range(8)), margins=MARGINS)

    assert report.arms.identical_representation_rate == 1.0
    assert report.verdict is Verdict.INCONCLUSIVE
    assert all(v.verdict is Verdict.INCONCLUSIVE for v in report.per_metric.values())


def test_a_vacuous_comparison_says_so_in_its_limitations() -> None:
    """The reason must travel with the report, not be inferred from a rate."""
    report = audit(MirrorScaffold(FakeBase()), problem=None, seeds=list(range(8)), margins=MARGINS)
    assert any("VACUOUS" in limitation for limitation in report.limitations)
    assert all("withdrawn" in (v.test or "") for v in report.per_metric.values())


def test_withdrawing_a_verdict_keeps_the_evidence_behind_it() -> None:
    """Downgrading must be auditable, so the interval and margin are retained."""
    report = audit(MirrorScaffold(FakeBase()), problem=None, seeds=list(range(8)), margins=MARGINS)
    loss = report.per_metric["loss"]
    assert loss.margin == MARGINS["loss"]
    assert loss.n == 8
    assert loss.ci_low == loss.ci_high == 0.0


def test_partial_agreement_does_not_trigger_the_guard() -> None:
    """Only *total* agreement is vacuous; agreeing often is an ordinary finding.

    A scaffold that lands on the control's answer most of the time is telling us
    something real about itself, and suppressing that would throw away the
    finding the guard exists to protect.
    """
    base = FakeBase()
    report = audit(
        FakeScaffold(base, loss=0.3), problem=None, seeds=list(range(8)), margins=MARGINS
    )
    assert report.arms.identical_representation_rate < 1.0
    assert not any("VACUOUS" in limitation for limitation in report.limitations)
