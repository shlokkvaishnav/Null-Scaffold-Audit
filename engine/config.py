"""Pydantic schemas for experiment configs, validated instead of loaded as bare YAML.

Two distinct config shapes exist in this repo and are formalized separately
rather than merged, because they drive different things:

- `RunConfig` describes one orchestrator run (a single domain+algorithm
  plugin pair) -- what `engine.orchestrator.DiscoveryOrchestrator.run` needs.
- `BaselineExperimentConfig` mirrors the existing `configs/paper/*.yaml`
  shape (multi-model, multi-seed comparison with ablations and significance
  testing) that `physics_discovery.experiments.contract` and
  `scripts/reproduce_benchmarks.py` already enforce/consume. This model adds
  static validation on top of that same contract -- it does not replace it.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml
from pydantic import BaseModel, Field, model_validator

from engine.orchestrator import ExperimentConfig
from physics_discovery.experiments.contract import (
    ExperimentContractError,
    validate_baseline_contract,
)


class RunConfig(BaseModel):
    """One orchestrator run: which plugins to use and how to split data."""

    domain: str
    algorithm: str
    domain_kwargs: Dict[str, Any] = Field(default_factory=dict)
    algorithm_kwargs: Dict[str, Any] = Field(default_factory=dict)
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


class DatasetSplit(BaseModel):
    strategy: str
    train: str
    validation: str
    test: str


class SeedPolicy(BaseModel):
    deterministic: bool
    seeds: List[int]


class Budget(BaseModel):
    max_iters: int
    candidate_bank_size: int
    regimes: int


class Reporting(BaseModel):
    run_columns: List[str]
    aggregate: Dict[str, str]
    artifact_format: str


class Significance(BaseModel):
    confidence_level: float
    test: str


class BaselineExperimentConfig(BaseModel):
    """Mirrors configs/paper/*.yaml. Rejects configs that violate the shared
    baseline experiment contract in physics_discovery.experiments.contract."""

    experiment_name: str
    dataset_split: DatasetSplit
    seed_policy: SeedPolicy
    models: List[str]
    metrics: List[str]
    reporting: Reporting
    budget: Budget
    ablations: List[str] = Field(default_factory=list)
    significance: Optional[Significance] = None

    @model_validator(mode="after")
    def _enforce_shared_contract(self) -> "BaselineExperimentConfig":
        try:
            validate_baseline_contract(
                self.model_dump(mode="json"), runner_name=self.experiment_name
            )
        except ExperimentContractError as exc:
            raise ValueError(str(exc)) from exc
        return self


def load_run_config(path: Path | str) -> RunConfig:
    with Path(path).open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    return RunConfig.model_validate(raw)


def load_baseline_experiment_config(path: Path | str) -> BaselineExperimentConfig:
    with Path(path).open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    return BaselineExperimentConfig.model_validate(raw)
