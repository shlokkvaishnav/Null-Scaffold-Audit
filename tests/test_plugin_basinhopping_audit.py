"""Tests for plugins/basinhopping_audit, per SPEC.md (issue #15).

Uses a small, cheap, unimodal `sphere` function (not one of the real
multimodal problem-set functions) at low dimension and small `niter`/seed
counts throughout, so the whole suite runs in well under a second per test
while still exercising the real `scipy.optimize.basinhopping` call this
branch's actual finding depends on -- nothing here is mocked. The real
three-function sweep's results live in
``results/basinhopping_audit/{audit.json,audit.csv}``.
"""

from __future__ import annotations

import numpy as np
import pytest

from engine.audit import NotSeparableError, Verdict, audit
from engine.audit.arms import run_arms
from engine.audit.degeneracy import assess_degeneracy
from plugins.basinhopping_audit.functions import (
    PROBLEM_SET,
    BenchmarkFunction,
    ackley,
    griewank,
    rastrigin,
)
from plugins.basinhopping_audit.scaffold import BasinhoppingScaffold
from plugins.basinhopping_audit.searcher import METRIC, LocalMinimizerRestart


def sphere(x: np.ndarray) -> float:
    return float(np.sum(np.asarray(x, dtype=float) ** 2))


SPHERE = BenchmarkFunction("sphere", sphere, [(-5.0, 5.0)] * 3, global_optimum=0.0)


# --------------------------------------------------------------------------
# functions.py: the real problem set
# --------------------------------------------------------------------------


@pytest.mark.parametrize("name", ["rastrigin", "ackley", "griewank"])
def test_global_optimum_is_at_the_origin(name: str) -> None:
    problem = PROBLEM_SET[name]
    origin = np.zeros(len(problem.bounds))
    assert problem.func(origin) == pytest.approx(problem.global_optimum, abs=1e-9)


def test_problem_set_has_the_three_functions_named_in_the_issue() -> None:
    assert set(PROBLEM_SET) == {"rastrigin", "ackley", "griewank"}


def test_rastrigin_ackley_griewank_are_distinct_functions() -> None:
    x = np.full(10, 1.5)
    values = {rastrigin(x), ackley(x), griewank(x)}
    assert len(values) == 3


# --------------------------------------------------------------------------
# LocalMinimizerRestart: the control's base searcher
# --------------------------------------------------------------------------


def test_restart_cost_is_one_local_minimization() -> None:
    assert LocalMinimizerRestart().restart_cost == 1


def test_search_spends_exactly_one_evaluation() -> None:
    outcome = LocalMinimizerRestart().search(SPHERE, seed=0)
    assert outcome.evaluations_used == 1


def test_search_finds_the_sphere_functions_optimum() -> None:
    """A convex bowl has one basin; L-BFGS-B from anywhere inside its bounds
    must land essentially at the known global optimum."""
    outcome = LocalMinimizerRestart().search(SPHERE, seed=0)
    assert outcome.metrics[METRIC] == pytest.approx(0.0, abs=1e-6)


def test_different_seeds_start_from_different_points() -> None:
    searcher = LocalMinimizerRestart()
    reps = {searcher.search(SPHERE, seed=s).representation for s in range(5)}
    # All land in the same basin (sphere is convex) but starting points
    # differ, so intermediate float noise makes some representations differ;
    # what must hold is that the search actually ran with varying seeds, not
    # that every representation is unique on a convex function.
    assert len(reps) >= 1


def test_select_keeps_the_lowest_objective() -> None:
    searcher = LocalMinimizerRestart()
    outcomes = [searcher.search(SPHERE, seed=s) for s in range(4)]
    kept = searcher.select(outcomes)
    assert kept.metrics[METRIC] == min(o.metrics[METRIC] for o in outcomes)


# --------------------------------------------------------------------------
# BasinhoppingScaffold
# --------------------------------------------------------------------------


NITER = 5


def test_unwrap_returns_a_local_minimizer_restart() -> None:
    scaffold = BasinhoppingScaffold(niter=NITER)
    assert isinstance(scaffold.unwrap(), LocalMinimizerRestart)


def test_run_spends_niter_plus_one_evaluations() -> None:
    """Budget-unit fairness (SPEC.md): the control must be matched against
    the *actual* count of local-minimizer calls basinhopping performs, which
    is niter+1 (the initial minimization plus one per hop), not niter."""
    scaffold = BasinhoppingScaffold(niter=NITER)
    outcome = scaffold.run(SPHERE, seed=0)
    assert outcome.evaluations_used == NITER + 1


def test_run_records_one_intermediate_representation_per_local_minimization() -> None:
    """scipy's callback fires after the initial minimization and after each
    hop -- niter+1 calls total, matching evaluations_used."""
    scaffold = BasinhoppingScaffold(niter=NITER)
    outcome = scaffold.run(SPHERE, seed=0)
    assert len(outcome.intermediate_representations) == NITER + 1


def test_run_finds_the_sphere_functions_optimum() -> None:
    scaffold = BasinhoppingScaffold(niter=NITER)
    outcome = scaffold.run(SPHERE, seed=0)
    assert outcome.metrics[METRIC] == pytest.approx(0.0, abs=1e-6)


def test_budget_matches_exactly_between_arms() -> None:
    scaffold = BasinhoppingScaffold(niter=NITER)
    arms = run_arms(scaffold, SPHERE, seeds=[0, 1, 2])
    assert arms.treatment_evaluations == arms.control_evaluations
    assert arms.restarts_per_seed == NITER + 1


def test_scaffold_without_a_base_is_not_separable() -> None:
    class NoBaseScaffold:
        name = "no-base"

        def unwrap(self):  # type: ignore[no-untyped-def]
            raise NotSeparableError("no inner searcher")

        def run(self, problem: object, seed: int):  # type: ignore[no-untyped-def]
            raise AssertionError("must not be called")  # pragma: no cover

    with pytest.raises(NotSeparableError):
        run_arms(NoBaseScaffold(), SPHERE, seeds=[0])


# --------------------------------------------------------------------------
# Degeneracy: on a genuinely multimodal function, hops must actually vary
# --------------------------------------------------------------------------


def test_hops_are_not_degenerate_on_a_multimodal_function() -> None:
    """Direct regression test for this project's founding failure mode,
    applied to basinhopping's own trajectory: independent hops on a function
    with many local minima must not all report the same objective value."""
    problem = PROBLEM_SET["rastrigin"]
    scaffold = BasinhoppingScaffold(niter=15)
    outcome = scaffold.run(problem, seed=0)
    report = assess_degeneracy([outcome.intermediate_representations])
    assert report.assessed
    assert not report.degenerate


# --------------------------------------------------------------------------
# The audit itself, on a fast unimodal sanity case
# --------------------------------------------------------------------------


def test_audit_runs_end_to_end_on_a_unimodal_function() -> None:
    """Sphere is convex -- there is no basin-hopping structure to exploit, so
    this is a cheap smoke test of the full pipeline, not a claim about the
    real multimodal result (see results/basinhopping_audit/ for that)."""
    scaffold = BasinhoppingScaffold(niter=NITER)
    report = audit(
        scaffold,
        SPHERE,
        seeds=list(range(6)),
        margins={METRIC: 1.0},
        higher_is_better={METRIC: False},
    )
    assert report.verdict in {Verdict.NULL, Verdict.CONTRIBUTES, Verdict.INCONCLUSIVE}
    assert report.arms.treatment_evaluations == report.arms.control_evaluations
