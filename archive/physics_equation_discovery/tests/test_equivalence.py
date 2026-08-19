"""Pins the rediscovery rule, which decides the project's headline metric.

`exact_recovery` is the number this work will be read for, and it is entirely
determined by what counts as equivalent. This project originally required exact
identity. It now follows SRBench's definition -- a candidate is a symbolic
solution when its difference from, or its ratio to, the ground truth simplifies
to a constant -- so recovery rates here mean the same thing as recovery rates in
the methods SRBench ranks.

That relaxation can only move the number upward, which is exactly why it needs
tests: a rule that silently counts more things as successes is the easiest way
to improve a result without improving anything. The strict figure is computed
alongside and asserted here too.
"""

from __future__ import annotations

import pytest

from engine.evaluation.equivalence import check_equivalence

VARIABLES = ["x0", "x1"]
RANGES = {"x0": [1.0, 5.0], "x1": [1.0, 5.0]}
TRUTH = "x0*x1"


def check(candidate: str) -> dict:
    return check_equivalence(
        candidate_equation=candidate,
        ground_truth_formula=TRUTH,
        variables=VARIABLES,
        test_ranges=RANGES,
        seed=0,
    )


def test_identical_expression_satisfies_both_rules() -> None:
    result = check("x0*x1")
    assert result["symbolic_match"]
    assert result["strict_match"]


def test_constant_multiple_counts_under_srbench_but_not_strictly() -> None:
    """The case the relaxation exists for.

    A symbolic regressor fits constants numerically, so recovering 2*x0*x1 for
    a truth of x0*x1 has found the form and missed a coefficient. SRBench counts
    the form; the strict rule does not.
    """
    result = check("2*x0*x1")
    assert result["symbolic_match"]
    assert not result["strict_match"]


def test_constant_offset_counts_under_srbench_but_not_strictly() -> None:
    result = check("x0*x1 + 3")
    assert result["symbolic_match"]
    assert not result["strict_match"]


def test_a_genuinely_different_expression_fails_both() -> None:
    """The relaxation must not become 'anything with the right variables'."""
    result = check("x0 + x1")
    assert not result["symbolic_match"]
    assert not result["strict_match"]


def test_a_collapsed_candidate_is_not_a_rediscovery() -> None:
    """Zero over anything is a constant ratio, and it has discovered nothing.

    Without the explicit guard this is the case that would silently score every
    degenerate run as a success.
    """
    result = check("0")
    assert not result["symbolic_match"]
    assert not result["strict_match"]


def test_a_constant_candidate_is_not_a_rediscovery() -> None:
    result = check("5")
    assert not result["symbolic_match"]
    assert not result["strict_match"]


def test_gplearn_prefix_notation_is_understood() -> None:
    """Both arms emit gplearn's printed form, so the rule must parse it."""
    result = check("mul(X0, X1)")
    assert result["symbolic_match"]


def test_unparseable_candidate_returns_false_rather_than_raising() -> None:
    """A check that could not run has not shown the candidate correct."""
    result = check("((((")
    assert not result["symbolic_match"]
    assert not result["strict_match"]


def test_every_documented_key_is_present() -> None:
    result = check("x0*x1")
    for key in ("symbolic_match", "strict_match", "numeric_match", "max_relative_error"):
        assert key in result


def test_strict_match_never_exceeds_srbench_match() -> None:
    """The strict rule is a subset of SRBench's. If it ever passed where
    SRBench's did not, one of the two is wrong."""
    for candidate in ("x0*x1", "2*x0*x1", "x0*x1 + 3", "x0 + x1", "0", "mul(X0, X1)"):
        result = check(candidate)
        assert not (result["strict_match"] and not result["symbolic_match"]), candidate


@pytest.mark.parametrize("candidate", ["x0*x1", "2*x0*x1", "x0 + x1"])
def test_result_is_stable_across_calls(candidate: str) -> None:
    """A verdict that moved between runs could not be published beside a number."""
    assert check(candidate) == check(candidate)
