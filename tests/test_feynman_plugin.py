"""Integration tests for equation_discovery.plugins.feynman: the first real
plugin behind the engine.plugin interfaces, driven through the orchestrator.

Requires sklearn.metrics (used by equation_discovery.evaluation.metrics) to be
importable; skipped otherwise rather than erroring collection.
"""

from __future__ import annotations

import pytest

pytest.importorskip("sklearn.metrics")

from engine.orchestrator import DiscoveryOrchestrator, ExperimentConfig
from engine.registry import PluginRegistry
from equation_discovery.plugins.feynman import (
    FeynmanDomainPlugin,
    GBMBaselineAlgorithm,
    SymbolicRegressionAlgorithm,
    register,
)

EQUATION_ID = "coulomb_force"


@pytest.fixture
def registry() -> PluginRegistry:
    reg = PluginRegistry()
    register(reg)
    return reg


def test_register_populates_expected_names(registry: PluginRegistry) -> None:
    assert registry.list_domains() == ["feynman_physics"]
    assert registry.list_algorithms() == [
        "discovery_agent",
        "gbm_baseline",
        "neural_moe",
        "symbolic_regression",
    ]


def test_domain_plugin_loads_dataset_with_ground_truth() -> None:
    domain = FeynmanDomainPlugin()
    dataset = domain.load_dataset(equation_id=EQUATION_ID, n_samples=50, seed=0)

    assert dataset.X.shape == (50, 4)
    assert dataset.y.shape == (50,)
    assert dataset.feature_names == ["q1", "q2", "eps0", "r"]
    assert dataset.metadata["ground_truth"]["id"] == EQUATION_ID


def test_domain_plugin_validate_flags_division_by_zero() -> None:
    domain = FeynmanDomainPlugin()
    assert domain.validate(None) == {}
    assert domain.validate("q1 / 0") != {}
    assert domain.validate("q1 * q2") == {}


def test_orchestrator_runs_gbm_baseline_end_to_end(registry: PluginRegistry) -> None:
    orchestrator = DiscoveryOrchestrator(registry)
    result = orchestrator.run(
        ExperimentConfig(
            domain="feynman_physics",
            algorithm="gbm_baseline",
            domain_kwargs={"equation_id": EQUATION_ID, "n_samples": 60, "seed": 0},
            algorithm_kwargs={"random_state": 0},
        )
    )

    assert result.domain == "feynman_physics"
    assert result.algorithm == "gbm_baseline"
    assert result.equation is None
    assert "rmse" in result.metrics
    assert result.constraints == {}


def test_orchestrator_runs_symbolic_regression_end_to_end(registry: PluginRegistry) -> None:
    orchestrator = DiscoveryOrchestrator(registry)
    result = orchestrator.run(
        ExperimentConfig(
            domain="feynman_physics",
            algorithm="symbolic_regression",
            domain_kwargs={"equation_id": EQUATION_ID, "n_samples": 60, "seed": 0},
            algorithm_kwargs={
                "backend": "gplearn",
                "population_size": 20,
                "generations": 2,
                "random_state": 0,
            },
        )
    )

    assert result.algorithm == "symbolic_regression"
    assert result.equation is not None
    assert "rmse" in result.metrics
    assert "symbolic_complexity" in result.metrics
