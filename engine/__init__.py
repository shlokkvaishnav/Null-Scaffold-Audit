"""SDE core engine: domain-agnostic orchestration over algorithm/domain plugins.

This package owns workflow sequencing only. It must never import from any
domain-specific package -- science and
domain logic live behind the `AlgorithmPlugin`/`DomainPlugin` interfaces in
`engine.plugin`, registered into an `engine.registry.PluginRegistry` and
driven by `engine.orchestrator.DiscoveryOrchestrator`.
"""

from __future__ import annotations

from engine.discovery import discover_plugins
from engine.orchestrator import DiscoveryOrchestrator, ExperimentConfig, RunResult
from engine.plugin import AlgorithmPlugin, Dataset, DomainPlugin
from engine.registry import PluginRegistry

__all__ = [
    "AlgorithmPlugin",
    "Dataset",
    "DiscoveryOrchestrator",
    "DomainPlugin",
    "ExperimentConfig",
    "PluginRegistry",
    "RunResult",
    "discover_plugins",
]
