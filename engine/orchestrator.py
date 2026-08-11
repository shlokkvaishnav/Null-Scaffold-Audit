"""Domain-agnostic workflow sequencing: load -> split -> fit -> predict -> validate -> score.

The orchestrator never performs scientific computation itself -- every step
that touches data semantics (loading, constraint checking, scoring) is
delegated to whichever DomainPlugin/AlgorithmPlugin the registry resolves for
a given ExperimentConfig.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from engine.plugin import Dataset
from engine.registry import PluginRegistry


@dataclass
class ExperimentConfig:
    """Minimal experiment description: which plugins to run and how to split data."""

    domain: str
    algorithm: str
    domain_kwargs: dict[str, Any] = field(default_factory=dict)
    algorithm_kwargs: dict[str, Any] = field(default_factory=dict)
    train_fraction: float = 0.8
    seed: int = 0


@dataclass
class RunResult:
    """Everything produced by one orchestrator run, ready for reporting/ranking."""

    domain: str
    algorithm: str
    equation: str | None
    metrics: dict[str, float]
    constraints: dict[str, Any]


def _train_test_split(dataset: Dataset, train_fraction: float):
    split = int(train_fraction * len(dataset.y))
    x_train, y_train = dataset.X[:split], dataset.y[:split]
    x_test, y_test = dataset.X[split:], dataset.y[split:]
    return x_train, y_train, x_test, y_test


class DiscoveryOrchestrator:
    """Runs one experiment config against a plugin registry."""

    def __init__(self, registry: PluginRegistry) -> None:
        self._registry = registry

    def run(self, config: ExperimentConfig) -> RunResult:
        domain = self._registry.build_domain(config.domain)
        dataset = domain.load_dataset(**config.domain_kwargs)

        x_train, y_train, x_test, y_test = _train_test_split(dataset, config.train_fraction)

        algorithm = self._registry.build_algorithm(config.algorithm, **config.algorithm_kwargs)
        algorithm.fit(x_train, y_train)
        y_pred = np.asarray(algorithm.predict(x_test))

        equation = algorithm.equation
        constraints = domain.validate(equation)
        metrics = domain.score(y_test, y_pred, equation)

        return RunResult(
            domain=domain.name,
            algorithm=algorithm.name,
            equation=equation,
            metrics=metrics,
            constraints=constraints,
        )
