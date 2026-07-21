"""The first SDE plugin: wraps the existing Feynman/physics rediscovery pipeline
behind the engine.plugin interfaces.

This module adapts pre-existing science code (generators, data loading,
metrics, validation) to `engine.plugin.AlgorithmPlugin`/`DomainPlugin` -- it
does not reimplement any of it. The adapters exist so the orchestrator can
drive this pipeline the same way it would drive any future plugin; the
underlying classes (`SymbolicHypothesisGenerator`, `BaselineModel`,
`Ensemble`, `DiscoveryAgent`) are unchanged and still importable/usable
directly (e.g. by `physics_discovery.evaluation.benchmark_runner`, which
predates this plugin layer and is untouched by it).
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np

from engine.plugin import Dataset
from engine.registry import PluginRegistry
from physics_discovery.core.agent import DiscoveryAgent
from physics_discovery.data.feynman_loader import generate_feynman_dataset
from physics_discovery.evaluation.metrics import compute_fit_metrics
from physics_discovery.generators.baselines import BaselineModel
from physics_discovery.generators.ensemble import Ensemble
from physics_discovery.generators.symbolic import SymbolicHypothesisGenerator
from physics_discovery.validation.equation_validity import EquationValidator


class SymbolicRegressionAlgorithm:
    """AlgorithmPlugin adapter around SymbolicHypothesisGenerator."""

    name = "symbolic_regression"

    def __init__(self, **config: Any) -> None:
        self._model = SymbolicHypothesisGenerator(config)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "SymbolicRegressionAlgorithm":
        self._model.fit(X, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self._model.predict(X)

    @property
    def equation(self) -> Optional[str]:
        return self._model.equation


class GBMBaselineAlgorithm:
    """AlgorithmPlugin adapter around BaselineModel (no symbolic output)."""

    name = "gbm_baseline"

    def __init__(self, **config: Any) -> None:
        config.setdefault("model_type", "lightgbm")
        self._model = BaselineModel(config)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "GBMBaselineAlgorithm":
        self._model.fit(X, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self._model.predict(X)

    @property
    def equation(self) -> Optional[str]:
        return None


class NeuralEnsembleAlgorithm:
    """AlgorithmPlugin adapter around Ensemble (neural+tree MoE baseline, no symbolic output)."""

    name = "neural_moe"

    def __init__(self, **config: Any) -> None:
        self._model = Ensemble(config)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "NeuralEnsembleAlgorithm":
        self._model.fit(X, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self._model.predict(X)

    @property
    def equation(self) -> Optional[str]:
        return None


class DiscoveryAgentAlgorithm:
    """AlgorithmPlugin adapter around the full DiscoveryAgent loop.

    Mirrors physics_discovery.evaluation.benchmark_runner._run_discovery_agent:
    num_regimes=1 because a Feynman equation is one global closed form, not a
    regime-switching system.
    """

    name = "discovery_agent"

    def __init__(
        self,
        backend: str = "gplearn",
        random_state: int = 0,
        max_iters: int = 3,
        **extra_agent_config: Any,
    ) -> None:
        agent_cfg: Dict[str, Any] = {
            "agent": {
                "num_regimes": 1,
                "use_verification": True,
                "use_memory": True,
                "use_belief": True,
                "use_reasoning": True,
                "reasoning_mode": backend,
                "random_state": random_state,
                **extra_agent_config,
            }
        }
        self._agent = DiscoveryAgent(agent_cfg)
        self._max_iters = max_iters
        self._best_hypothesis = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "DiscoveryAgentAlgorithm":
        obs = {"features": X, "targets": y}
        for _ in range(self._max_iters):
            self._agent.step(obs)

        if not self._agent.memory or not self._agent.memory.hypotheses:
            raise RuntimeError("DiscoveryAgent produced no hypotheses.")

        self._best_hypothesis = max(
            self._agent.memory.hypotheses, key=lambda h: getattr(h, "score", float("-inf"))
        )
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self._best_hypothesis is None:
            raise RuntimeError("DiscoveryAgentAlgorithm must be fit before predict.")
        return self._best_hypothesis.evaluate(X)

    @property
    def equation(self) -> Optional[str]:
        if self._best_hypothesis is None:
            return None
        return str(self._best_hypothesis.equation)


class FeynmanDomainPlugin:
    """DomainPlugin adapter around the Feynman dataset loader, validator, and metrics."""

    name = "feynman_physics"

    def __init__(self) -> None:
        self._validator = EquationValidator()

    def load_dataset(
        self,
        equation_id: str,
        n_samples: int = 500,
        noise_std: float = 0.0,
        seed: int = 0,
    ) -> Dataset:
        X, y, ground_truth = generate_feynman_dataset(
            equation_id, n_samples=n_samples, noise_std=noise_std, seed=seed
        )
        return Dataset(
            X=X,
            y=y,
            feature_names=list(ground_truth["variables"]),
            metadata={"ground_truth": ground_truth},
        )

    def validate(self, equation: Optional[str]) -> Dict[str, Any]:
        if not equation:
            return {}
        return self._validator.check_constraints(equation)

    def score(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        equation: Optional[str] = None,
    ) -> Dict[str, float]:
        return compute_fit_metrics(y_true, y_pred, equation=equation)


def register(registry: PluginRegistry) -> None:
    """Register every plugin this module provides into the given registry."""
    registry.register_domain(FeynmanDomainPlugin.name, FeynmanDomainPlugin)
    registry.register_algorithm(SymbolicRegressionAlgorithm.name, SymbolicRegressionAlgorithm)
    registry.register_algorithm(GBMBaselineAlgorithm.name, GBMBaselineAlgorithm)
    registry.register_algorithm(NeuralEnsembleAlgorithm.name, NeuralEnsembleAlgorithm)
    registry.register_algorithm(DiscoveryAgentAlgorithm.name, DiscoveryAgentAlgorithm)
