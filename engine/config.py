"""Pydantic schema for an orchestrator run, validated instead of loaded as bare YAML.

`RunConfig` describes one orchestrator run -- a single domain+algorithm plugin
pair -- which is what `engine.orchestrator.DiscoveryOrchestrator.run` needs. It
names no plugin and fixes no experiment protocol: `domain` and `algorithm` are
opaque registry keys, and the kwargs are passed through unread.

A paper-benchmark config schema used to live here too. It was moved out to the
plugin package that owns that protocol, because it fixed one plugin's seed
list, metric set, and iteration budget -- things the engine may not know
(Article 5). See the commit that removed it for the destination; naming the
package here would reintroduce the violation the move was made to fix.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field

from engine.orchestrator import ExperimentConfig


class RunConfig(BaseModel):
    """One orchestrator run: which plugins to use and how to split data."""

    domain: str
    algorithm: str
    domain_kwargs: dict[str, Any] = Field(default_factory=dict)
    algorithm_kwargs: dict[str, Any] = Field(default_factory=dict)
    train_fraction: float = 0.8
    seed: int = 0

    def to_experiment_config(self) -> ExperimentConfig:
        return ExperimentConfig(
            domain=self.domain,
            algorithm=self.algorithm,
            domain_kwargs=self.domain_kwargs,
            algorithm_kwargs=self.algorithm_kwargs,
            train_fraction=self.train_fraction,
            seed=self.seed,
        )


def load_run_config(path: Path | str) -> RunConfig:
    with Path(path).open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    return RunConfig.model_validate(raw)
