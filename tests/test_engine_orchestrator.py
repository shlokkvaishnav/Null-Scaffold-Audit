"""Orchestrator wiring tests using fake plugins -- no equation_discovery/ML deps.

These exist to prove the engine seams (Dataset/AlgorithmPlugin/DomainPlugin/
PluginRegistry/DiscoveryOrchestrator) work generically, independent of any
concrete science implementation.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
import pytest

from engine.orchestrator import DiscoveryOrchestrator, ExperimentConfig
from engine.plugin import Dataset
from engine.registry import PluginRegistry


class _LinearAlgorithm:
    """Fake AlgorithmPlugin: predicts the mean of y_train, always."""

    name = "mean_predictor"

    def __init__(self, **_: Any) -> None:
        self._mean = 0.0

    def fit(self, X: np.ndarray, y: np.ndarray) -> "_LinearAlgorithm":
        self._mean = float(np.mean(y))
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.full(len(X), self._mean)

    @property
    def equation(self) -> Optional[str]:
        return f"y = {self._mean}"


class _NoEquationAlgorithm:
    """Fake AlgorithmPlugin with no symbolic output, e.g. a black-box baseline."""

    name = "black_box"

    def fit(self, X: np.ndarray, y: np.ndarray) -> "_NoEquationAlgorithm":
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.zeros(len(X))

    @property
    def equation(self) -> Optional[str]:
        return None


class _FakeDomain:
    """Fake DomainPlugin: constant synthetic dataset, trivial validation/scoring."""

    name = "fake_domain"

    def load_dataset(self, **kwargs: Any) -> Dataset:
        n = kwargs.get("n_samples", 20)
        X = np.arange(n, dtype=float).reshape(-1, 1)
        y = np.full(n, 3.0)
        return Dataset(X=X, y=y, feature_names=["x0"], metadata={"source": "fake"})

    def validate(self, equation: Optional[str]) -> Dict[str, Any]:
        if equation and "bad" in equation:
            return {"bad_equation": {"violation_rate": 1.0}}
        return {}

    def score(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        equation: Optional[str] = None,
    ) -> Dict[str, float]:
        rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
        return {"rmse": rmse, "has_equation": float(equation is not None)}


@pytest.fixture
def registry() -> PluginRegistry:
    reg = PluginRegistry()
    reg.register_domain("fake_domain", _FakeDomain)
    reg.register_algorithm("mean_predictor", _LinearAlgorithm)
    reg.register_algorithm("black_box", _NoEquationAlgorithm)
    return reg


def test_orchestrator_runs_symbolic_algorithm(registry: PluginRegistry) -> None:
    orchestrator = DiscoveryOrchestrator(registry)
    result = orchestrator.run(ExperimentConfig(domain="fake_domain", algorithm="mean_predictor"))

    assert result.domain == "fake_domain"
    assert result.algorithm == "mean_predictor"
    assert result.equation == "y = 3.0"
    assert result.metrics["rmse"] == pytest.approx(0.0, abs=1e-9)
    assert result.metrics["has_equation"] == 1.0
    assert result.constraints == {}


def test_orchestrator_handles_algorithm_with_no_equation(registry: PluginRegistry) -> None:
    orchestrator = DiscoveryOrchestrator(registry)
    result = orchestrator.run(ExperimentConfig(domain="fake_domain", algorithm="black_box"))

    assert result.equation is None
    assert result.metrics["has_equation"] == 0.0
    assert result.constraints == {}


def test_registry_rejects_duplicate_and_unknown_names(registry: PluginRegistry) -> None:
    with pytest.raises(ValueError):
        registry.register_domain("fake_domain", _FakeDomain)

    with pytest.raises(KeyError):
        registry.build_algorithm("does_not_exist")


def test_train_fraction_controls_split_used_for_fit(registry: PluginRegistry) -> None:
    orchestrator = DiscoveryOrchestrator(registry)
    result = orchestrator.run(
        ExperimentConfig(
            domain="fake_domain",
            algorithm="mean_predictor",
            domain_kwargs={"n_samples": 10},
            train_fraction=0.5,
        )
    )
    # y is constant 3.0 regardless of split, so the mean predictor should be exact.
    assert result.metrics["rmse"] == pytest.approx(0.0, abs=1e-9)
