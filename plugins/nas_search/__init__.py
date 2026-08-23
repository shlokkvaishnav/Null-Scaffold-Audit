"""NATS-Bench-backed NAS domain plugin. See SPEC.md for scope.

Only the pieces required by ``research/nas-random-search-self-audit``
(issue #11): a ``RandomSearch`` base searcher, an ``IdentityRestartScaffold``
null-calibration wrapper, and a problem source over NATS-Bench's tabular
topology search space. Registering a real ``DomainPlugin``/``AlgorithmPlugin``
or a NAS controller (README item 3) is out of scope here.
"""

from __future__ import annotations

import os

from engine.registry import PluginRegistry
from plugins.nas_search.problem import NatsBenchTopologyProblemSource
from plugins.nas_search.scaffold import IdentityRestartScaffold
from plugins.nas_search.searcher import RandomSearch

__all__ = ["IdentityRestartScaffold", "NatsBenchTopologyProblemSource", "RandomSearch", "register"]

# The topology-space "simple" file is ~1.1GB and is not vendored in this
# repository (it is downloaded once, out of band -- see SPEC.md). Pointing
# this env var at the extracted directory is what makes the domain
# registrable on a given machine.
NATS_BENCH_FILE_ENV = "NATS_BENCH_TSS_SIMPLE_PATH"


def register(registry: PluginRegistry) -> None:
    """Register the NATS-Bench problem source, if its data file is configured.

    A no-op, not a raise, when ``NATS_BENCH_FILE_ENV`` is unset:
    ``engine.discovery.discover_plugins`` already treats a raising
    ``register()`` as "skip with a warning", but a domain that is simply not
    configured on this machine (e.g. CI, which does not have the 1.1GB file)
    is a normal state, not a broken plugin.
    """
    path = os.environ.get(NATS_BENCH_FILE_ENV)
    if not path:
        return
    registry.register_problem_source(
        "nas_search", lambda **kwargs: NatsBenchTopologyProblemSource(path, **kwargs)
    )
