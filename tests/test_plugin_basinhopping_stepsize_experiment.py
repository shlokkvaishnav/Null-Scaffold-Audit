"""Tests for plugins/basinhopping_audit/run_stepsize_experiment.py (issue #17).

Checks the pre-registered derivation rule and seed disjointness from PR
#16's run -- the two properties STEPSIZE_SPEC.md's "Confounds considered"
section depends on -- without re-running the real, slow sweep (that lives
in results/basinhopping_audit_stepsize_scaling/).
"""

from __future__ import annotations

import json

import pytest

import plugins.basinhopping_audit.run_audit as original
import plugins.basinhopping_audit.run_stepsize_experiment as experiment
from plugins.basinhopping_audit.functions import PROBLEM_SET

# --------------------------------------------------------------------------
# The derivation rule: pre-registered, not tuned after seeing a result
# --------------------------------------------------------------------------


def test_ratio_is_derived_from_rastrigins_original_configuration() -> None:
    rastrigin_width = PROBLEM_SET["rastrigin"].bounds[0][1] - PROBLEM_SET["rastrigin"].bounds[0][0]
    assert experiment.STEPSIZE_RATIO == pytest.approx(0.5 / rastrigin_width)


def test_rastrigins_stepsize_is_unchanged_by_construction() -> None:
    assert experiment.domain_scaled_stepsize("rastrigin") == pytest.approx(0.5)


def test_stepsizes_match_the_issues_stated_derivation() -> None:
    """STEPSIZE_SPEC.md states these three values explicitly; a change here
    without a corresponding SPEC update would be exactly the after-the-fact
    retuning GIT_WORKFLOW.md's cherry-picking rule forbids."""
    assert experiment.domain_scaled_stepsize("rastrigin") == pytest.approx(0.5, abs=1e-9)
    assert experiment.domain_scaled_stepsize("ackley") == pytest.approx(3.2, abs=0.01)
    assert experiment.domain_scaled_stepsize("griewank") == pytest.approx(58.59, abs=0.01)


def test_stepsize_scales_monotonically_with_domain_width() -> None:
    """The whole point of the experiment: ratio held fixed, so stepsize must
    grow exactly in proportion to width, not by some other rule."""
    widths = {name: p.bounds[0][1] - p.bounds[0][0] for name, p in PROBLEM_SET.items()}
    stepsizes = {name: experiment.domain_scaled_stepsize(name) for name in PROBLEM_SET}
    ratios = {name: stepsizes[name] / widths[name] for name in PROBLEM_SET}
    assert ratios["rastrigin"] == pytest.approx(ratios["ackley"])
    assert ratios["ackley"] == pytest.approx(ratios["griewank"])


# --------------------------------------------------------------------------
# Seed disjointness from PR #16's own run
# --------------------------------------------------------------------------


def test_pilot_seeds_are_disjoint_from_the_original_runs_pilot_seeds() -> None:
    assert set(experiment.PILOT_SEEDS).isdisjoint(original.PILOT_SEEDS)


SAFE_BOUND = 5000
"""A generous ceiling on how large any single run's real sweep could
plausibly be, for disjointness checks -- `run_audit.py` no longer defines
`MAX_SEEDS` (removed by issue #25), so there is no fixed attribute to read
its historical maximum from; this bound is chosen far above any seed count
this project's feasibility probes have produced (the largest so far,
issue #19's Ackley re-run, needed 1971)."""


def test_real_sweep_seeds_are_disjoint_from_the_original_runs_seed_range() -> None:
    """PR #16's real sweep starts at `run_audit.REAL_SWEEP_SEED_OFFSET`."""
    this_experiment_seeds = range(
        experiment.REAL_SWEEP_SEED_OFFSET, experiment.REAL_SWEEP_SEED_OFFSET + SAFE_BOUND
    )
    original_possible_seeds = range(
        original.REAL_SWEEP_SEED_OFFSET, original.REAL_SWEEP_SEED_OFFSET + SAFE_BOUND
    )
    assert set(this_experiment_seeds).isdisjoint(original_possible_seeds)


def test_real_sweep_seeds_are_also_disjoint_from_this_experiments_own_pilot() -> None:
    this_experiment_seeds = range(
        experiment.REAL_SWEEP_SEED_OFFSET, experiment.REAL_SWEEP_SEED_OFFSET + SAFE_BOUND
    )
    assert set(this_experiment_seeds).isdisjoint(experiment.PILOT_SEEDS)


# --------------------------------------------------------------------------
# The real result, pinned against the committed artifact
# --------------------------------------------------------------------------


def test_real_result_artifact_uses_the_derived_stepsizes() -> None:
    path = experiment.REPO_ROOT / "results" / "basinhopping_audit_stepsize_scaling" / "audit.json"
    if not path.exists():
        pytest.skip("results/basinhopping_audit_stepsize_scaling/audit.json not present")

    payload = json.loads(path.read_text(encoding="utf-8"))
    for name in PROBLEM_SET:
        stepsize = payload["functions"][name]["config"]["stepsize"]
        assert stepsize == pytest.approx(experiment.domain_scaled_stepsize(name))
