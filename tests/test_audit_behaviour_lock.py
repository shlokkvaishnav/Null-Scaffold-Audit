"""Pin the audit's output so the domain-independence refactor cannot change it silently.

The refactor this guards moves ~4,000 lines out of `physics_discovery/` and into
`engine/`, `algorithms/`, `validators/` and `plugins/`. One of those moves is
dangerous in a way the type checker cannot see.

`HypothesisScorer` lazily constructs an `EquationValidator` and uses it to
penalise constraint violations. That scorer is the selection rule for *both*
audit arms. Once it moves into the domain-independent core it can no longer
import a validator, so it must receive one by injection -- and if a caller
forgets to pass one, violation penalties quietly stop counting. Both arms shift,
the verdicts move, every number in the sweep changes, and nothing raises.

So the contract is: a relocation may change where code lives and what it is
called, and may not change what the audit reports. This test states that
mechanically. It runs the real pipeline -- real gplearn fits, the real scaffold,
the real scorer -- at a deliberately tiny budget, and compares everything the
audit produces against a recorded fixture.

Regenerate the fixture with::

    uv run python tests/test_audit_behaviour_lock.py

Regenerating is *not* a routine step. If this test fails during the refactor the
default reading is that the refactor changed behaviour, and the fixture is
correct. Only regenerate when a behaviour change is intended and argued for, and
say so in the commit that does it.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
# The runner lives in scripts/, which is not a package. It is imported rather
# than duplicated so the lock covers the margins the real sweeps use: a fixture
# built against a private copy of the margins would keep passing while the
# runner's own margins drifted.
for _path in (REPO_ROOT, REPO_ROOT / "scripts"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from run_null_scaffold_audit import (
    HIGHER_IS_BETTER,
    default_registry,
    margins_for,
)

from engine.audit import audit
from plugins.physics.audit_adapter import DiscoveryAgentScaffold

FIXTURE = Path(__file__).parent / "fixtures" / "audit_behaviour_lock.json"

# Frozen configuration. These are not tuning knobs -- changing any of them
# invalidates the fixture, so they are constants rather than parameters.
#
# Two equations rather than eight, four seeds rather than twenty, and a
# population 10x smaller than a real sweep: the lock has to run on every test
# invocation, so it buys determinism at the smallest budget that still exercises
# both arms, the selection rule, the degeneracy check and the statistics.
PROBLEM_SOURCE = "physics"
EQUATIONS = ("kinetic_energy", "coulomb_force")
SEEDS = 4
N_SAMPLES = 200
MAX_ITERS = 3
POPULATION_SIZE = 50
GENERATIONS = 5

# gplearn under a fixed random_state and the BCa bootstrap under its fixed seed
# are both bit-reproducible -- verified before this fixture was recorded. The
# tolerance therefore exists to absorb float reassociation from moving code
# between modules, not genuine drift: observed differences in this audit are of
# order 1e-2, so 1e-12 cannot hide a real change.
TOLERANCE = 1e-12


def run_locked_audit() -> dict[str, Any]:
    """Run the frozen configuration and return a JSON-shaped snapshot of everything."""
    scaffold = DiscoveryAgentScaffold(
        max_iters=MAX_ITERS,
        population_size=POPULATION_SIZE,
        generations=GENERATIONS,
    )

    # Resolved through the registry, exactly as the runner does. The lock is
    # on the audit's output, not on any particular way of reaching it -- but
    # the data must be identical, so this asserts the source by name rather
    # than accepting whichever one happens to be registered first.
    source = default_registry().build_problem_source(PROBLEM_SOURCE)

    snapshot: dict[str, Any] = {}
    for equation_id in EQUATIONS:
        problem = source.build_problem(equation_id, n_samples=N_SAMPLES, seed=0)
        report = audit(
            scaffold,
            problem,
            list(range(SEEDS)),
            margins=margins_for(problem),
            higher_is_better=HIGHER_IS_BETTER,
        )
        arms = report.arms
        snapshot[equation_id] = {
            "verdict": report.verdict.value,
            "restarts_per_seed": arms.restarts_per_seed,
            "treatment_evaluations": arms.treatment_evaluations,
            "control_evaluations": arms.control_evaluations,
            "identical_representation_rate": arms.identical_representation_rate,
            "degenerate": report.degeneracy.degenerate,
            "mean_distinct_ratio": report.degeneracy.mean_distinct_ratio,
            # The equation strings are the strongest signal in this fixture. A
            # change anywhere in the search, the scorer, or the selection rule
            # shows up here first, and shows up as something a reader can
            # actually inspect rather than as a shifted decimal.
            "treatment_representations": [o.representation for o in arms.treatment],
            "control_representations": [o.representation for o in arms.control],
            "per_metric": {
                metric: {
                    "verdict": verdict.verdict.value,
                    "observed_difference": verdict.observed_difference,
                    "ci_low": verdict.ci_low,
                    "ci_high": verdict.ci_high,
                    "margin": verdict.margin,
                    "power": verdict.power,
                    "n": verdict.n,
                    # The multiplicity correction is part of the claim, so it is
                    # locked with it. Without these, a change to the correction
                    # could silently promote or demote a verdict and the lock
                    # would notice only if the enum happened to move.
                    "p_value": verdict.p_value,
                    "adjusted_p_value": verdict.adjusted_p_value,
                    "correction": verdict.correction,
                }
                for metric, verdict in report.per_metric.items()
            },
        }
    return snapshot


@pytest.fixture(scope="module")
def recorded() -> dict[str, Any]:
    if not FIXTURE.exists():
        pytest.fail(
            f"{FIXTURE} is missing. Record it with:\n"
            f"    uv run python tests/test_audit_behaviour_lock.py"
        )
    return json.loads(FIXTURE.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def observed() -> dict[str, Any]:
    """One real audit run, shared by every assertion below."""
    return run_locked_audit()


@pytest.mark.parametrize("equation_id", EQUATIONS)
def test_verdicts_are_unchanged(
    equation_id: str, observed: dict[str, Any], recorded: dict[str, Any]
) -> None:
    """The verdict is the audit's output. If it moved, the refactor changed the finding."""
    now, before = observed[equation_id], recorded[equation_id]
    assert now["verdict"] == before["verdict"]
    assert {m: v["verdict"] for m, v in now["per_metric"].items()} == {
        m: v["verdict"] for m, v in before["per_metric"].items()
    }


@pytest.mark.parametrize("equation_id", EQUATIONS)
def test_representations_are_unchanged(
    equation_id: str, observed: dict[str, Any], recorded: dict[str, Any]
) -> None:
    """Both arms must still find the same equations, seed for seed.

    This catches a changed selection rule that happens to leave the verdict
    alone -- the case a verdict-only lock would pass.
    """
    now, before = observed[equation_id], recorded[equation_id]
    assert now["treatment_representations"] == before["treatment_representations"]
    assert now["control_representations"] == before["control_representations"]


@pytest.mark.parametrize("equation_id", EQUATIONS)
def test_statistics_are_unchanged(
    equation_id: str, observed: dict[str, Any], recorded: dict[str, Any]
) -> None:
    now, before = observed[equation_id], recorded[equation_id]
    for metric, expected in before["per_metric"].items():
        actual = now["per_metric"][metric]
        for field in ("observed_difference", "ci_low", "ci_high", "margin", "power"):
            assert actual[field] == pytest.approx(expected[field], abs=TOLERANCE), (
                f"{equation_id}.{metric}.{field} moved"
            )
        assert actual["n"] == expected["n"]


@pytest.mark.parametrize("equation_id", EQUATIONS)
def test_budget_matching_is_unchanged(
    equation_id: str, observed: dict[str, Any], recorded: dict[str, Any]
) -> None:
    """A shifted restart count would silently re-weight the comparison itself."""
    now, before = observed[equation_id], recorded[equation_id]
    for field in ("restarts_per_seed", "treatment_evaluations", "control_evaluations"):
        assert now[field] == before[field], f"{equation_id}.{field} moved"


@pytest.mark.parametrize("equation_id", EQUATIONS)
def test_degeneracy_is_unchanged(
    equation_id: str, observed: dict[str, Any], recorded: dict[str, Any]
) -> None:
    now, before = observed[equation_id], recorded[equation_id]
    assert now["degenerate"] == before["degenerate"]
    assert now["mean_distinct_ratio"] == pytest.approx(before["mean_distinct_ratio"], abs=TOLERANCE)
    assert now["identical_representation_rate"] == pytest.approx(
        before["identical_representation_rate"], abs=TOLERANCE
    )


if __name__ == "__main__":
    FIXTURE.parent.mkdir(parents=True, exist_ok=True)
    FIXTURE.write_text(json.dumps(run_locked_audit(), indent=2), encoding="utf-8")
    print(f"recorded {FIXTURE}")
