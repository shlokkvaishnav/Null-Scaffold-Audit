"""Integration tests for physics_discovery.plugins.feynman: the first real
plugin behind the engine.plugin interfaces, driven through the orchestrator.

The plugin transitively imports several compiled sklearn extensions
(sklearn.metrics, sklearn.ensemble, ...) -- in some locked-down environments
(e.g. a Windows Application Control / AppLocker policy) one or more of these
DLLs are blocked at import time. Skip this module rather than error
collection when that happens; verify via Docker (Linux, unaffected) instead.
"""

from __future__ import annotations

import pytest

from engine.orchestrator import DiscoveryOrchestrator, ExperimentConfig
from engine.registry import PluginRegistry

try:
    from physics_discovery.plugins.feynman import FeynmanDomainPlugin, register
except ImportError as exc:
    pytest.skip(
        f"physics_discovery.plugins.feynman not importable here: {exc}", allow_module_level=True
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
