"""In-process registry mapping plugin names to factories.

Deliberately simple: explicit registration calls, no dynamic/sandboxed
loading or entry-point discovery yet. Those can be layered on top of this
same registration surface later without changing callers.
"""

from __future__ import annotations

from typing import Any, Callable, Dict

from engine.plugin import AlgorithmPlugin, DomainPlugin

AlgorithmFactory = Callable[..., AlgorithmPlugin]
DomainFactory = Callable[..., DomainPlugin]


class PluginRegistry:
    """Holds algorithm and domain plugin factories, keyed by name."""

    def __init__(self) -> None:
        self._algorithms: Dict[str, AlgorithmFactory] = {}
        self._domains: Dict[str, DomainFactory] = {}

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

    def list_algorithms(self) -> list[str]:
        return sorted(self._algorithms)

    def list_domains(self) -> list[str]:
        return sorted(self._domains)
