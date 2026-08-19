"""Integration tests for plugins.synthetic.plugin: the second
domain plugin, used to stress-test DomainPlugin generality (no equation_id,
no known ground truth) against the algorithm plugins registered by
plugins.physics.plugin.

Same environment caveat as tests/test_feynman_plugin.py: skipped if a
transitively-imported compiled sklearn extension is blocked here.
"""

from __future__ import annotations

import pytest

from engine.orchestrator import DiscoveryOrchestrator, ExperimentConfig
from engine.registry import PluginRegistry

try:
    from plugins.physics import plugin as feynman
    from plugins.synthetic import plugin as synthetic
except ImportError as exc:
    pytest.skip(
        f"plugins.physics/plugins.synthetic not importable here: {exc}", allow_module_level=True
    )


@pytest.fixture
def registry() -> PluginRegistry:
    reg = PluginRegistry()
    feynman.register(reg)
    synthetic.register(reg)
    return reg


def test_register_adds_domain_without_duplicating_algorithms(registry: PluginRegistry) -> None:
    assert registry.list_domains() == ["feynman_physics", "synthetic_regression"]
    assert registry.list_algorithms() == [
        "discovery_agent",
        "gbm_baseline",
        "neural_moe",
        "symbolic_regression",
    ]


def test_domain_plugin_loads_dataset_with_no_ground_truth() -> None:
    domain = synthetic.SyntheticRegressionDomainPlugin()
    dataset = domain.load_dataset(seed=0, n_samples=80, n_features=6)

    assert dataset.X.shape == (80, 6)
    assert dataset.y.shape == (80,)
    assert dataset.feature_names == ["x0", "x1", "x2", "x3", "x4", "x5"]
    assert "ground_truth" not in dataset.metadata
    assert "generating_formula" in dataset.metadata


def test_orchestrator_runs_gbm_baseline_against_synthetic_domain(registry: PluginRegistry) -> None:
    orchestrator = DiscoveryOrchestrator(registry)
    result = orchestrator.run(
        ExperimentConfig(
            domain="synthetic_regression",
            algorithm="gbm_baseline",
            domain_kwargs={"seed": 0, "n_samples": 100},
            algorithm_kwargs={"random_state": 0},
        )
    )

    assert result.domain == "synthetic_regression"
    assert result.algorithm == "gbm_baseline"
    assert result.equation is None
    assert "rmse" in result.metrics


def test_orchestrator_runs_symbolic_regression_against_synthetic_domain(
    registry: PluginRegistry,
) -> None:
    orchestrator = DiscoveryOrchestrator(registry)
    result = orchestrator.run(
        ExperimentConfig(
            domain="synthetic_regression",
            algorithm="symbolic_regression",
            domain_kwargs={"seed": 0, "n_samples": 100},
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
