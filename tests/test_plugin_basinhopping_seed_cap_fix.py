"""Tests for issue #25: removing the MAX_SEEDS cap from run_audit.py and
run_stepsize_experiment.py.

The confirmatory re-run itself (`results/basinhopping_audit_seed_cap_fix/`)
is expensive (hundreds of real basinhopping sweeps) and is not reproduced
by this suite -- these tests pin the invariants the fix must hold
(no ceiling, a floor only) and, where the confirmatory artifact is present,
check its verdicts against SEED_CAP_FIX_SPEC.md's Results table.
"""

from __future__ import annotations

import json

import pytest

from plugins.basinhopping_audit import run_audit, run_stepsize_experiment


@pytest.mark.parametrize("module", [run_audit, run_stepsize_experiment])
def test_no_max_seeds_ceiling(module: object) -> None:
    assert not hasattr(module, "MAX_SEEDS")


@pytest.mark.parametrize("module", [run_audit, run_stepsize_experiment])
def test_min_seeds_floor_still_present(module: object) -> None:
    assert module.MIN_SEEDS == 20  # type: ignore[attr-defined]


def test_run_audit_and_run_stepsize_experiment_seed_blocks_are_disjoint() -> None:
    assert set(run_audit.PILOT_SEEDS).isdisjoint(run_stepsize_experiment.PILOT_SEEDS)
    bound = 5000
    audit_real = range(run_audit.REAL_SWEEP_SEED_OFFSET, run_audit.REAL_SWEEP_SEED_OFFSET + bound)
    stepsize_real = range(
        run_stepsize_experiment.REAL_SWEEP_SEED_OFFSET,
        run_stepsize_experiment.REAL_SWEEP_SEED_OFFSET + bound,
    )
    assert set(audit_real).isdisjoint(stepsize_real)


# --------------------------------------------------------------------------
# The confirmatory re-run, pinned against its committed artifacts
# --------------------------------------------------------------------------

EXPECTED = {
    "run_audit": {
        "rastrigin": "CONTRIBUTES",
        "ackley": "HARMFUL",
        "griewank": "HARMFUL",  # see DECISION_LOG.md -- not to be read as established
    },
    "run_stepsize_experiment": {
        "rastrigin": "CONTRIBUTES",
        "ackley": "HARMFUL",
        "griewank": "NULL",
    },
}


@pytest.mark.parametrize("script", ["run_audit", "run_stepsize_experiment"])
def test_confirmatory_rerun_matches_seed_cap_fix_spec(script: str) -> None:
    path = (
        run_audit.REPO_ROOT / "results" / "basinhopping_audit_seed_cap_fix" / script / "audit.json"
    )
    if not path.exists():
        pytest.skip(f"results/basinhopping_audit_seed_cap_fix/{script}/audit.json not present")

    payload = json.loads(path.read_text(encoding="utf-8"))
    for function, expected_verdict in EXPECTED[script].items():
        assert payload["functions"][function]["verdict"] == expected_verdict


def test_confirmatory_rerun_did_not_overwrite_the_historical_artifacts() -> None:
    """PR #16's and PR #18's own committed results must be untouched by this
    branch -- the confirmatory re-run writes to a new directory instead."""
    for path, needed_n, seed_count in (
        (run_audit.REPO_ROOT / "results" / "basinhopping_audit" / "audit.json", 291, 60),
        (
            run_audit.REPO_ROOT / "results" / "basinhopping_audit_stepsize_scaling" / "audit.json",
            181,
            60,
        ),
    ):
        if not path.exists():
            pytest.skip(f"{path} not present")
        payload = json.loads(path.read_text(encoding="utf-8"))
        rastrigin_config = payload["functions"]["rastrigin"]["config"]
        assert rastrigin_config["required_n_for_80pct_power"] == needed_n
        assert rastrigin_config["seed_count"] == seed_count  # still capped, as originally published
