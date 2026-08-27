"""Tests for plugins/basinhopping_audit/run_ackley_power_experiment.py (issue #19).

Checks the two properties ACKLEY_POWER_SPEC.md's "Confounds considered"
depends on -- seed disjointness across every prior basinhopping_audit run,
and an uncapped seed count -- without re-running the real, slow sweep (that
lives in results/basinhopping_audit_ackley_power/).
"""

from __future__ import annotations

import json

import pytest

import plugins.basinhopping_audit.run_ackley_power_experiment as experiment
import plugins.basinhopping_audit.run_audit as pr16
import plugins.basinhopping_audit.run_stepsize_experiment as pr18

# --------------------------------------------------------------------------
# Seed disjointness across every prior basinhopping_audit run
# --------------------------------------------------------------------------


def test_pilot_seeds_are_disjoint_from_pr16s_pilot() -> None:
    assert set(experiment.PILOT_SEEDS).isdisjoint(pr16.PILOT_SEEDS)


def test_pilot_seeds_are_disjoint_from_pr18s_pilot() -> None:
    assert set(experiment.PILOT_SEEDS).isdisjoint(pr18.PILOT_SEEDS)


def _real_sweep_range(offset: int, max_possible_n: int) -> range:
    return range(offset, offset + max_possible_n)


def test_real_sweep_range_is_disjoint_from_pr16s_real_sweep_range() -> None:
    # pr16 (run_audit.py) no longer defines MAX_SEEDS (removed by issue #25);
    # SAFE_BOUND is a generous ceiling far above any real seed count this
    # project's feasibility probes have produced.
    SAFE_BOUND = 5000
    this_experiment = _real_sweep_range(experiment.REAL_SWEEP_SEED_OFFSET, SAFE_BOUND)
    pr16_possible = _real_sweep_range(pr16.REAL_SWEEP_SEED_OFFSET, SAFE_BOUND)
    assert set(this_experiment).isdisjoint(pr16_possible)


def test_real_sweep_range_is_disjoint_from_pr18s_real_sweep_range() -> None:
    SAFE_BOUND = 5000  # pr18 (run_stepsize_experiment.py) no longer defines MAX_SEEDS either
    this_experiment = _real_sweep_range(experiment.REAL_SWEEP_SEED_OFFSET, SAFE_BOUND)
    pr18_possible = _real_sweep_range(pr18.REAL_SWEEP_SEED_OFFSET, SAFE_BOUND)
    assert set(this_experiment).isdisjoint(pr18_possible)


def test_real_sweep_range_is_disjoint_from_both_pilots() -> None:
    this_experiment = _real_sweep_range(experiment.REAL_SWEEP_SEED_OFFSET, 500)
    assert set(this_experiment).isdisjoint(pr16.PILOT_SEEDS)
    assert set(this_experiment).isdisjoint(pr18.PILOT_SEEDS)
    assert set(this_experiment).isdisjoint(experiment.PILOT_SEEDS)


# --------------------------------------------------------------------------
# Uncapped seed count -- the whole point of this branch
# --------------------------------------------------------------------------


def test_there_is_no_max_seeds_ceiling() -> None:
    """PR #18's MAX_SEEDS=60 cap is exactly the defect this branch exists to
    avoid repeating; this module must not define an analogous ceiling."""
    assert not hasattr(experiment, "MAX_SEEDS")


def test_stepsize_and_niter_are_unchanged_from_pr18() -> None:
    assert experiment.STEPSIZE == pytest.approx(3.2)
    assert experiment.NITER == 50


# --------------------------------------------------------------------------
# The real result, pinned against the committed artifact
# --------------------------------------------------------------------------


def test_real_result_used_the_pre_registered_stepsize_and_no_cap() -> None:
    path = experiment.REPO_ROOT / "results" / "basinhopping_audit_ackley_power" / "audit.json"
    if not path.exists():
        pytest.skip("results/basinhopping_audit_ackley_power/audit.json not present")

    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["config"]["stepsize"] == pytest.approx(3.2)
    assert payload["config"]["function"] == "ackley"
    # The real run must use exactly what the feasibility probe called for
    # (floored at MIN_SEEDS), never silently re-capped -- the exact defect
    # this issue exists to fix.
    needed = payload["config"]["required_n_for_80pct_power"]
    expected = max(needed, experiment.MIN_SEEDS) if needed is not None else experiment.MIN_SEEDS
    assert payload["config"]["seed_count"] == expected
