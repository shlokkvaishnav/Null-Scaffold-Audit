"""In-process registry mapping plugin names to factories.

Deliberately simple: explicit registration calls, no dynamic/sandboxed
loading or entry-point discovery yet. Those can be layered on top of this
same registration surface later without changing callers.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from engine.audit.problem import AuditProblemSource
from engine.plugin import AlgorithmPlugin, DomainPlugin

AlgorithmFactory = Callable[..., AlgorithmPlugin]
DomainFactory = Callable[..., DomainPlugin]
ProblemSourceFactory = Callable[..., AuditProblemSource]


class PluginRegistry:
    """Holds algorithm, domain and audit problem-source factories, keyed by name."""

    def __init__(self) -> None:
        self._algorithms: dict[str, AlgorithmFactory] = {}
        self._domains: dict[str, DomainFactory] = {}
        # Kept separate from _domains because being auditable is an extra
        # capability rather than part of being a domain. A plugin may register
        # either, or both.
        self._problem_sources: dict[str, ProblemSourceFactory] = {}

    def register_algorithm(self, name: str, factory: AlgorithmFactory) -> None:
        if name in self._algorithms:
            raise ValueError(f"Algorithm plugin already registered: {name!r}")
        self._algorithms[name] = factory

    def register_domain(self, name: str, factory: DomainFactory) -> None:
        if name in self._domains:
            raise ValueError(f"Domain plugin already registered: {name!r}")
        self._domains[name] = factory

    def build_algorithm(self, name: str, **kwargs: Any) -> AlgorithmPlugin:
        if name not in self._algorithms:
            raise KeyError(
                f"Unknown algorithm plugin: {name!r}. Registered: {sorted(self._algorithms)}"
            )
        return self._algorithms[name](**kwargs)

    def build_domain(self, name: str, **kwargs: Any) -> DomainPlugin:
        if name not in self._domains:
            raise KeyError(f"Unknown domain plugin: {name!r}. Registered: {sorted(self._domains)}")
        return self._domains[name](**kwargs)

    def register_problem_source(self, name: str, factory: ProblemSourceFactory) -> None:
        if name in self._problem_sources:
            raise ValueError(f"Audit problem source already registered: {name!r}")
        self._problem_sources[name] = factory

    def build_problem_source(self, name: str, **kwargs: Any) -> AuditProblemSource:
        if name not in self._problem_sources:
            raise KeyError(
                f"Unknown audit problem source: {name!r}. "
                f"Registered: {sorted(self._problem_sources)}"
            )
        return self._problem_sources[name](**kwargs)

    def list_algorithms(self) -> list[str]:
        return sorted(self._algorithms)

    def list_domains(self) -> list[str]:
        return sorted(self._domains)

    def list_problem_sources(self) -> list[str]:
        return sorted(self._problem_sources)
