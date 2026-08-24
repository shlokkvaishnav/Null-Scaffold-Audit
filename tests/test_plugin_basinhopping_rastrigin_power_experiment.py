"""Tests for plugins/basinhopping_audit/run_rastrigin_power_experiment.py
(issue #21).

Checks the two properties RASTRIGIN_POWER_SPEC.md's "Confounds considered"
depends on -- seed disjointness across every prior basinhopping_audit run
(now four blocks, including issue #19/PR #20's), and an uncapped seed
count -- without re-running the real, slow sweep (that lives in
results/basinhopping_audit_rastrigin_power/).
"""

from __future__ import annotations

import json

import pytest

import plugins.basinhopping_audit.run_ackley_power_experiment as pr20
import plugins.basinhopping_audit.run_audit as pr16
import plugins.basinhopping_audit.run_rastrigin_power_experiment as experiment
import plugins.basinhopping_audit.run_stepsize_experiment as pr18

# --------------------------------------------------------------------------
# Seed disjointness across every prior basinhopping_audit run
# --------------------------------------------------------------------------


def test_pilot_seeds_are_disjoint_from_every_prior_pilot() -> None:
    assert set(experiment.PILOT_SEEDS).isdisjoint(pr16.PILOT_SEEDS)
    assert set(experiment.PILOT_SEEDS).isdisjoint(pr18.PILOT_SEEDS)
    assert set(experiment.PILOT_SEEDS).isdisjoint(pr20.PILOT_SEEDS)


def _real_sweep_range(offset: int, max_possible_n: int) -> range:
    return range(offset, offset + max_possible_n)


def test_real_sweep_range_is_disjoint_from_pr16s_real_sweep_range() -> None:
    # pr16 (run_audit.py) no longer defines MAX_SEEDS (removed by issue #25);
    # 5000 is a generous ceiling far above any real seed count this
    # project's feasibility probes have produced.
    this_experiment = _real_sweep_range(experiment.REAL_SWEEP_SEED_OFFSET, 5000)
    pr16_possible = _real_sweep_range(pr16.REAL_SWEEP_SEED_OFFSET, 5000)
    assert set(this_experiment).isdisjoint(pr16_possible)


def test_real_sweep_range_is_disjoint_from_pr18s_real_sweep_range() -> None:
    this_experiment = _real_sweep_range(experiment.REAL_SWEEP_SEED_OFFSET, 5000)
    pr18_possible = _real_sweep_range(pr18.REAL_SWEEP_SEED_OFFSET, 5000)
    assert set(this_experiment).isdisjoint(pr18_possible)


def test_real_sweep_range_is_disjoint_from_pr20s_real_sweep_range() -> None:
    """PR #20's real sweep ran at n=1971 -- pad generously past that in case
    a re-run of this test suite's assumptions ever needs to grow."""
    this_experiment = _real_sweep_range(experiment.REAL_SWEEP_SEED_OFFSET, 5000)
    pr20_possible = _real_sweep_range(pr20.REAL_SWEEP_SEED_OFFSET, 5000)
    assert set(this_experiment).isdisjoint(pr20_possible)


def test_real_sweep_range_is_disjoint_from_all_pilots() -> None:
    this_experiment = _real_sweep_range(experiment.REAL_SWEEP_SEED_OFFSET, 5000)
    assert set(this_experiment).isdisjoint(pr16.PILOT_SEEDS)
    assert set(this_experiment).isdisjoint(pr18.PILOT_SEEDS)
    assert set(this_experiment).isdisjoint(pr20.PILOT_SEEDS)
    assert set(this_experiment).isdisjoint(experiment.PILOT_SEEDS)


# --------------------------------------------------------------------------
# Uncapped seed count -- the whole point of this branch, and of issue #19
# --------------------------------------------------------------------------


def test_there_is_no_max_seeds_ceiling() -> None:
    """PR #18's MAX_SEEDS=60 cap is the defect this whole thread exists to
    avoid repeating; this module must not define an analogous ceiling."""
    assert not hasattr(experiment, "MAX_SEEDS")


def test_stepsize_and_niter_are_unchanged_from_pr18() -> None:
    assert experiment.STEPSIZE == pytest.approx(0.5)
    assert experiment.NITER == 50


# --------------------------------------------------------------------------
# The real result, pinned against the committed artifact
# --------------------------------------------------------------------------


def test_real_result_used_the_pre_registered_stepsize_and_no_cap() -> None:
    path = experiment.REPO_ROOT / "results" / "basinhopping_audit_rastrigin_power" / "audit.json"
    if not path.exists():
        pytest.skip("results/basinhopping_audit_rastrigin_power/audit.json not present")

    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["config"]["stepsize"] == pytest.approx(0.5)
    assert payload["config"]["function"] == "rastrigin"
    # The real run must use exactly what the feasibility probe called for
    # (floored at MIN_SEEDS), never silently re-capped.
    needed = payload["config"]["required_n_for_80pct_power"]
    expected = max(needed, experiment.MIN_SEEDS) if needed is not None else experiment.MIN_SEEDS
    assert payload["config"]["seed_count"] == expected
