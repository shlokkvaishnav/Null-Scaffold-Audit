"""Tests for plugins/nas_search, per SPEC.md (issue #11).

A fake NATS-Bench-shaped problem stands in for the real one throughout, for
the same reason ``tests/test_audit_calibration.py`` uses a fake searcher: it
runs in milliseconds and makes the ground truth exact, so a failure here is
unambiguously this plugin's and not a matter of what a real sweep happened to
find that day. The real NATS-Bench data file is ~1.1GB, is not vendored in
this repository, and is not required to run this suite -- see
``plugins/nas_search/__init__.py``'s ``NATS_BENCH_FILE_ENV`` and
``results/nas_search_self_audit/`` for the real-data sweep this branch
actually ran.
"""

from __future__ import annotations

from typing import Any

import pytest

from engine.audit import NotSeparableError, Verdict, audit
from engine.audit.arms import run_arms
from plugins.nas_search import IdentityRestartScaffold, RandomSearch
from plugins.nas_search.problem import NatsBenchTopologyProblemSource
from plugins.nas_search.searcher import METRIC


class FakeApi:
    """Deterministic accuracy(index) = index % 100, so the true best in any
    sampled set is known in advance without touching real benchmark data."""

    def __init__(self, num_archs: int) -> None:
        self.num_archs = num_archs

    def __len__(self) -> int:
        return self.num_archs

    def get_more_info(
        self, index: int, dataset: str, hp: str = "12", is_random: bool = True
    ) -> dict[str, Any]:
        del dataset, hp, is_random
        return {"valid-accuracy": float(index % 100)}


class FakeProblem:
    """Duck-types `plugins.nas_search.problem.NatsBenchProblem`."""

    def __init__(self, num_archs: int = 1000) -> None:
        self.num_archs = num_archs
        self.api = FakeApi(num_archs)

    def valid_accuracy(self, index: int) -> float:
        return self.api.get_more_info(index, "cifar10-valid", hp="200", is_random=False)[
            "valid-accuracy"
        ]


# --------------------------------------------------------------------------
# RandomSearch
# --------------------------------------------------------------------------


def test_restart_cost_is_the_configured_budget() -> None:
    assert RandomSearch(budget=37).restart_cost == 37


def test_search_spends_exactly_the_budget() -> None:
    outcome = RandomSearch(budget=25).search(FakeProblem(), seed=0)
    assert outcome.evaluations_used == 25


def test_search_picks_the_true_best_of_its_sample() -> None:
    """The chosen candidate must be the best-scoring one it actually sampled,
    not merely a plausible one -- exact because FakeProblem's accuracy
    function is exact."""
    problem = FakeProblem(num_archs=1000)
    outcome = RandomSearch(budget=50).search(problem, seed=0)
    assert outcome.representation is not None
    assert outcome.metrics[METRIC] == problem.valid_accuracy(int(outcome.representation))


def test_search_samples_without_replacement() -> None:
    """A budget larger than the population would otherwise force duplicate
    indices into one sample, which `np.random.choice(replace=False)` raises
    on -- catching that regression here rather than at a much larger budget
    in a real sweep."""
    problem = FakeProblem(num_archs=20)
    outcome = RandomSearch(budget=20).search(problem, seed=0)
    assert outcome.evaluations_used == 20


def test_different_seeds_sample_different_architectures() -> None:
    """The regression this whole issue exists to catch: a seed that does not
    actually vary the sample would make every 'independent' restart return
    the same architecture (this project's own founding incident, reproduced
    in a new domain)."""
    problem = FakeProblem(num_archs=1000)
    searcher = RandomSearch(budget=50)
    reps = {searcher.search(problem, seed=s).representation for s in range(10)}
    assert len(reps) > 1


def test_select_keeps_the_best_of_several_restarts() -> None:
    problem = FakeProblem(num_archs=1000)
    searcher = RandomSearch(budget=50)
    outcomes = [searcher.search(problem, seed=s) for s in range(5)]
    kept = searcher.select(outcomes)
    assert kept.metrics[METRIC] == max(o.metrics[METRIC] for o in outcomes)


# --------------------------------------------------------------------------
# IdentityRestartScaffold
# --------------------------------------------------------------------------


def test_unwrap_returns_the_same_base_instance() -> None:
    base = RandomSearch(budget=10)
    scaffold = IdentityRestartScaffold(base=base, restarts=3)
    assert scaffold.unwrap() is base


def test_scaffold_run_spends_restarts_times_budget() -> None:
    scaffold = IdentityRestartScaffold(base=RandomSearch(budget=10), restarts=3)
    outcome = scaffold.run(FakeProblem(), seed=0)
    assert outcome.evaluations_used == 30


def test_scaffold_does_not_share_seeds_with_a_bare_restart() -> None:
    """Guards the confound SPEC.md calls out explicitly: the scaffold's
    internal restart seeds must not collide with what `run_arms`'s control
    arm would derive for the same trial index, or the comparison degenerates
    into the arms being literally the same run."""
    from engine.audit.arms import paired_seed
    from engine.audit.calibration import _CALIBRATION_OFFSET

    base = RandomSearch(budget=10)
    scaffold = IdentityRestartScaffold(base=base, restarts=1)
    assert scaffold._seed_for(0, 0) != 0  # not the bare control seed
    assert scaffold._seed_for(0, 0) == paired_seed(0 + _CALIBRATION_OFFSET, 0)


# --------------------------------------------------------------------------
# The audit itself: this issue's actual hypothesis
# --------------------------------------------------------------------------


MARGINS = {METRIC: 5.0}


def test_identity_restart_scaffold_audits_as_null() -> None:
    """This is the issue's hypothesis, reproduced on a fake but exact problem:
    a scaffold that is null by construction must be reported NULL."""
    base = RandomSearch(budget=50)
    scaffold = IdentityRestartScaffold(base=base, restarts=3)
    report = audit(
        scaffold,
        FakeProblem(num_archs=2000),
        seeds=list(range(30)),
        margins=MARGINS,
        higher_is_better={METRIC: True},
    )
    assert report.verdict is Verdict.NULL
    assert not report.degeneracy.degenerate


def test_identity_restart_scaffold_is_not_degenerate() -> None:
    base = RandomSearch(budget=50)
    scaffold = IdentityRestartScaffold(base=base, restarts=3)
    arms = run_arms(scaffold, FakeProblem(num_archs=2000), seeds=list(range(10)))
    from engine.audit.degeneracy import assess_degeneracy

    report = assess_degeneracy([o.intermediate_representations for o in arms.treatment])
    assert report.assessed
    assert not report.degenerate


def test_scaffold_without_a_base_is_not_separable() -> None:
    class NoBaseScaffold:
        name = "no-base"

        def unwrap(self) -> RandomSearch:
            raise NotSeparableError("no inner searcher")

        def run(self, problem: Any, seed: int) -> Any:  # pragma: no cover
            raise AssertionError("must not be called")

    with pytest.raises(NotSeparableError):
        run_arms(NoBaseScaffold(), FakeProblem(), seeds=[0])


# --------------------------------------------------------------------------
# NatsBenchTopologyProblemSource -- no real data file required
# --------------------------------------------------------------------------


def test_problem_source_lists_the_single_problem() -> None:
    source = NatsBenchTopologyProblemSource("/does/not/exist")
    assert source.list_problems() == ["cifar10-topology"]


def test_problem_source_rejects_unknown_problem_id_without_touching_the_data_file() -> None:
    """The unknown-id check must happen before the (expensive, optional)
    NATS-Bench file is loaded, so this is checkable without nats-bench
    installed or the 1.1GB data file present."""
    source = NatsBenchTopologyProblemSource("/does/not/exist")
    with pytest.raises(KeyError):
        source.build_problem("not-a-real-problem", n_samples=0, seed=0)


def test_registering_without_the_env_var_is_a_noop() -> None:
    import os

    from engine.registry import PluginRegistry
    from plugins.nas_search import NATS_BENCH_FILE_ENV, register

    previous = os.environ.pop(NATS_BENCH_FILE_ENV, None)
    try:
        registry = PluginRegistry()
        register(registry)
        assert registry.list_problem_sources() == []
    finally:
        if previous is not None:
            os.environ[NATS_BENCH_FILE_ENV] = previous


def test_registering_with_the_env_var_set_registers_nas_search() -> None:
    import os

    from engine.registry import PluginRegistry
    from plugins.nas_search import NATS_BENCH_FILE_ENV, register

    previous = os.environ.get(NATS_BENCH_FILE_ENV)
    os.environ[NATS_BENCH_FILE_ENV] = "/does/not/exist"
    try:
        registry = PluginRegistry()
        register(registry)
        assert registry.list_problem_sources() == ["nas_search"]
    finally:
        if previous is None:
            del os.environ[NATS_BENCH_FILE_ENV]
        else:
            os.environ[NATS_BENCH_FILE_ENV] = previous
