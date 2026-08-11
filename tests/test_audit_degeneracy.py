"""Tests for the intra-run degeneracy pre-check (RFC-0001 section 4.3).

The check exists to catch a loop that repeats itself instead of exploring. Its
dangerous failure is the false positive: calling a wrapper degenerate when it is
merely unlucky would be an accusation about the implementation rather than a
measurement of it, so the tests below pin the conservative direction hard.
"""

from __future__ import annotations

from engine.audit import DegeneracyReport, SearchOutcome, Verdict, assess_degeneracy, audit


def test_identical_proposals_are_degenerate() -> None:
    """The defect that motivated the audit: every iteration returns the same thing."""
    report = assess_degeneracy([["x0*x1", "x0*x1", "x0*x1"]])
    assert report.degenerate
    assert report.degenerate_runs == 1
    assert report.mean_distinct_ratio == 1 / 3


def test_varied_proposals_are_not_degenerate() -> None:
    report = assess_degeneracy([["x0", "x1", "x0+x1"]])
    assert not report.degenerate
    assert report.mean_distinct_ratio == 1.0


def test_degeneracy_requires_every_run_to_repeat() -> None:
    """One exploring run is enough to withdraw the claim.

    "Repeats on some seeds" and "never explores" are different findings; only
    the second is worth asserting mechanically.
    """
    report = assess_degeneracy([["a", "a", "a"], ["a", "b", "c"]])
    assert not report.degenerate
    assert report.degenerate_runs == 1
    assert report.runs == 2


def test_single_proposal_runs_are_excluded_not_condemned() -> None:
    """A wrapper proposing once per run was never in a position to vary."""
    report = assess_degeneracy([["only"], ["one"]])
    assert not report.assessed
    assert not report.degenerate


def test_runs_with_too_few_proposals_are_dropped_from_the_denominator() -> None:
    report = assess_degeneracy([["only"], ["a", "a"]])
    assert report.assessed
    assert report.runs == 1
    assert report.degenerate


def test_no_runs_at_all_is_unassessed() -> None:
    report = assess_degeneracy([])
    assert not report.assessed
    assert not report.degenerate
    assert "not assessed" in report.summary()


def test_unassessed_report_never_claims_degeneracy() -> None:
    assert not DegeneracyReport(assessed=False).degenerate


def test_summary_names_the_finding() -> None:
    assert "DEGENERATE" in assess_degeneracy([["a", "a"]]).summary()
    assert "explores" in assess_degeneracy([["a", "b"]]).summary()


# --------------------------------------------------------------------------
# Wiring: the audit attaches degeneracy without folding it into the verdict
# --------------------------------------------------------------------------


class _Base:
    restart_cost = 100

    def search(self, problem: object, seed: int) -> SearchOutcome:
        return SearchOutcome(
            metrics={"loss": 0.0, "selection_score": 0.0},
            evaluations_used=100,
            representation="base",
        )

    def select(self, outcomes):
        return outcomes[0]


class _RepeatingScaffold:
    name = "Repeating"

    def unwrap(self) -> _Base:
        return _Base()

    def run(self, problem: object, seed: int) -> SearchOutcome:
        return SearchOutcome(
            metrics={"loss": 0.0, "selection_score": 0.0},
            evaluations_used=100,
            representation="same",
            intermediate_representations=("same", "same", "same"),
        )


def test_audit_reports_degeneracy_alongside_a_null_verdict() -> None:
    """NULL says the wrapper did not help; DEGENERATE says why. Both are reported."""
    report = audit(_RepeatingScaffold(), problem=None, seeds=list(range(20)), margins={"loss": 0.5})
    assert report.verdict is Verdict.NULL
    assert report.degeneracy.degenerate


def test_degeneracy_defaults_to_unassessed_for_opaque_pipelines() -> None:
    """A pipeline that exposes no internals is not penalised for it."""

    class Opaque(_RepeatingScaffold):
        def run(self, problem: object, seed: int) -> SearchOutcome:
            return SearchOutcome(
                metrics={"loss": 0.0, "selection_score": 0.0},
                evaluations_used=100,
                representation="opaque",
            )

    report = audit(Opaque(), problem=None, seeds=list(range(20)), margins={"loss": 0.5})
    assert not report.degeneracy.assessed
    assert not report.degeneracy.degenerate
