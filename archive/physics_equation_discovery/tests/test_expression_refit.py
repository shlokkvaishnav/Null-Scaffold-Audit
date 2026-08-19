"""Tests for structure-preserving constant refitting.

This exists to test a hypothesis that predicts refitting *helps*, which makes an
over-eager implementation the dangerous one: a refit that quietly rewrites structure, or
that fits against the wrong split, would manufacture the improvement it was built to
measure. So the tests below care less about the fit being good than about it being
*bounded* -- never worse than the incumbent, never touching what it should not, never
raising on the sort of malformed candidate a search emits by the hundred.
"""

from __future__ import annotations

import numpy as np

from engine.expressions.expression_eval import safe_evaluate
from engine.expressions.refit import refit_constants


def data(n: int = 200, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Features and a target of ``3 * x0 * x1``, so the right constant is 3.0."""
    rng = np.random.default_rng(seed)
    features = rng.uniform(1.0, 5.0, size=(n, 2))
    return features, 3.0 * features[:, 0] * features[:, 1]


def train_error(equation: str, features: np.ndarray, targets: np.ndarray) -> float:
    predicted = safe_evaluate(equation, features)
    if predicted is None:
        return float("inf")
    predicted = np.where(np.isfinite(predicted), predicted, 1e30)
    return float(np.sum((predicted - targets) ** 2))


# --------------------------------------------------------------------------
# It does the thing
# --------------------------------------------------------------------------


def test_recovers_a_constant_the_search_got_wrong() -> None:
    """The motivating case: right structure, wrong constant, bad score.

    This is what "Good Structure, Bad Score" names. A genetic-programming searcher can
    find ``c * x0 * x1`` and still rank it poorly because ``c`` arrived as a random
    terminal and was never fitted.
    """
    features, targets = data()
    refitted = refit_constants("mul(2.7, mul(X0, X1))", features, targets)
    predicted = safe_evaluate(refitted, features)
    assert predicted is not None
    assert np.allclose(predicted, targets, rtol=1e-6)


def test_refitting_never_increases_training_error() -> None:
    """The load-bearing invariant, and a real one rather than a hope.

    The optimiser starts from the incumbent constants, so it cannot end above them
    without something having gone wrong. Checked across candidates of very different
    quality, including ones where the structure is hopeless and no constant saves it.
    """
    features, targets = data()
    candidates = [
        "mul(2.7, mul(X0, X1))",
        "add(0.5, mul(X0, X1))",
        "div(X0, 0.31)",
        "add(mul(0.2, X0), mul(0.9, X1))",
        "log(mul(0.05, X0))",
    ]
    for equation in candidates:
        refitted = refit_constants(equation, features, targets)
        assert train_error(refitted, features, targets) <= train_error(equation, features, targets)


# --------------------------------------------------------------------------
# It does not do anything else
# --------------------------------------------------------------------------


def test_a_candidate_with_no_literals_is_returned_untouched() -> None:
    """Nothing to fit means nothing to change, byte for byte.

    Rewriting it into sympy's infix form would still be *correct*, but it would churn
    the representation strings the audit compares between arms, and the
    identical-representation rate is what detects a vacuous comparison.
    """
    features, targets = data()
    assert refit_constants("mul(X0, X1)", features, targets) == "mul(X0, X1)"


def test_an_unparseable_candidate_does_not_raise() -> None:
    """A search emits these by the hundred; one must not end a multi-hour sweep."""
    features, targets = data()
    for junk in ("((((", "mul(X0,", "", "not an expression at all"):
        assert refit_constants(junk, features, targets) == junk


def test_a_candidate_naming_an_absent_column_is_returned_unchanged() -> None:
    """Two features, a candidate referring to a third: refuse rather than guess."""
    features, targets = data()
    assert refit_constants("mul(0.5, X7)", features, targets) == "mul(0.5, X7)"


def test_exponents_are_not_treated_as_parameters() -> None:
    """Fitting an exponent is not a refit, it is a change of structure.

    It also breaks evaluation for negative bases by going complex, which would look
    like the candidate failing rather than like the refit overreaching.
    """
    features, targets = data()
    refitted = refit_constants("mul(1.5, mul(X0, X0))", features, targets)
    assert "**2" in refitted or "X0*X0" in refitted


def test_a_candidate_with_too_many_constants_is_refused() -> None:
    """Past a point this stops preserving structure and becomes curve fitting.

    The literals have to be wrapped in ``sin`` to survive sympification. A nest of
    constant-only additions folds to a single number before this module ever sees it --
    ``add(2.8, add(2.6, ...))`` arrives as ``28.72``, one literal, comfortably under the
    cap. That is sympy doing the right thing, and it is why the count that matters is
    the one taken *after* parsing rather than the one a reader counts in the string.
    """
    features, targets = data()
    terms = [f"sin(mul({index / 10 + 0.1:.3f}, X0))" for index in range(13)]
    many = terms[0]
    for term in terms[1:]:
        many = f"add({term}, {many})"
    assert refit_constants(many, features, targets) == many


# --------------------------------------------------------------------------
# It fits the data it was given, and only that
# --------------------------------------------------------------------------


def test_the_fit_depends_only_on_the_data_passed_in() -> None:
    """Determinism, and the guard against a fit that peeked at another split.

    Same equation and same training arrays must give the same string no matter what
    else exists. If this ever failed, a refitted ceiling could not be compared against
    an unrefitted one, because the two would differ by something other than the refit.
    """
    features, targets = data()
    first = refit_constants("mul(2.7, mul(X0, X1))", features, targets)
    second = refit_constants("mul(2.7, mul(X0, X1))", features, targets)
    assert first == second


def test_fitting_on_a_different_split_gives_a_different_answer() -> None:
    """The complement of the test above: it really is using the arrays handed to it.

    A refit that ignored its inputs would pass every determinism check and be useless.
    Here the same structure is fitted against two different targets, and the recovered
    constants must follow the targets.
    """
    features, _ = data()
    three = refit_constants(
        "mul(2.7, mul(X0, X1))", features, 3.0 * features[:, 0] * features[:, 1]
    )
    seven = refit_constants(
        "mul(2.7, mul(X0, X1))", features, 7.0 * features[:, 0] * features[:, 1]
    )
    assert three != seven
    assert np.allclose(
        safe_evaluate(seven, features), 7.0 * features[:, 0] * features[:, 1], rtol=1e-6
    )


def test_a_hopeless_structure_stays_hopeless() -> None:
    """Refitting must not rescue a candidate whose structure cannot fit the target.

    If it could, the improvement being measured would be the optimiser's rather than
    the structure's, and the whole comparison would be meaningless.
    """
    features, targets = data()
    refitted = refit_constants("add(0.5, X0)", features, targets)
    predicted = safe_evaluate(refitted, features)
    assert predicted is not None
    assert not np.allclose(predicted, targets, rtol=0.1)
