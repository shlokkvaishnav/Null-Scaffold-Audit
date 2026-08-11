"""Tests for the plugin-side audit adapter.

These exist because of a specific failure: a multi-hour sweep died partway
through on a candidate that could not be evaluated on some test rows. The cost
of that bug was measured in compute time, not correctness, which is exactly the
kind of defect a test suite stops paying for twice.
"""

from __future__ import annotations

import numpy as np

from engine.audit import AuditProblem
from plugins.physics.audit_adapter import (
    _outcome_metrics,
    score_like_the_pipeline,
)


def make_problem() -> AuditProblem:
    rng = np.random.default_rng(0)
    x = rng.uniform(1.0, 5.0, size=(40, 2))
    y = x[:, 0] * x[:, 1]
    return AuditProblem(
        equation_id="synthetic_product",
        x_train=x[:32],
        y_train=y[:32],
        x_test=x[32:],
        y_test=y[32:],
        ground_truth={
            "formula": "x0*x1",
            "variables": ["x0", "x1"],
            "ranges": {"x0": (1.0, 5.0), "x1": (1.0, 5.0)},
        },
    )


def test_evaluable_candidate_reports_no_nonfinite() -> None:
    metrics = _outcome_metrics(make_problem(), "x0*x1", seed=0)
    assert metrics["nonfinite_fraction"] == 0.0
    assert np.isfinite(metrics["rmse"])


def test_non_evaluable_candidate_does_not_raise() -> None:
    """The regression: log of a negative produced NaN and killed the sweep.

    sklearn rejects NaN outright, so before the fix this raised rather than
    scoring. A symbolic search emits such expressions routinely.
    """
    metrics = _outcome_metrics(make_problem(), "log(x0 - 1000000.0)", seed=0)
    assert np.isfinite(metrics["rmse"])
    assert np.isfinite(metrics["mae"])


def test_unparseable_candidate_counts_as_not_recovered() -> None:
    """An equivalence check that cannot run has not shown the candidate correct."""
    metrics = _outcome_metrics(make_problem(), "((((", seed=0)
    assert metrics["exact_recovery"] == 0.0


def test_every_metric_the_audit_registers_margins_for_is_present() -> None:
    """The runner pre-registers margins by name; a missing key aborts the sweep."""
    metrics = _outcome_metrics(make_problem(), "x0*x1", seed=0)
    for required in ("rmse", "symbolic_complexity", "exact_recovery"):
        assert required in metrics


def test_scoring_helper_returns_a_finite_score_for_a_sane_candidate() -> None:
    problem = make_problem()
    score = score_like_the_pipeline(
        "x0*x1", {"features": problem.x_train, "targets": problem.y_train}
    )
    assert np.isfinite(score)


def test_scoring_helper_prefers_the_better_candidate() -> None:
    """Selection must order candidates, or the control arm picks arbitrarily."""
    problem = make_problem()
    observation = {"features": problem.x_train, "targets": problem.y_train}
    good = score_like_the_pipeline("x0*x1", observation)
    bad = score_like_the_pipeline("x0 + 1000000.0", observation)
    assert good > bad
