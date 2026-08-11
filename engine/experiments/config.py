"""Pydantic schema over the `configs/paper/*.yaml` benchmark-config shape.

This lived in `engine/config.py` until it was found to violate Constitution
Article 5: it encodes *this plugin's* experiment protocol -- a fixed seed list,
a fixed metric set, a fixed iteration budget -- and it imported
`engine.experiments.contract` to enforce it. An engine that ships a
schema for one plugin's experiment protocol is not domain independent, whatever
the schema is named.

The fix is removal rather than indirection. Injecting the contract as a callable
would have satisfied the import checker while leaving the engine holding a type
whose every field exists to mirror one plugin's YAML. See Article 15: when the
core seems to need a special case, the special case does not belong in the core.

The model adds static validation on top of the same contract that
`scripts/reproduce_benchmarks.py` enforces at runtime; it does not replace it.
"""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, Field, model_validator

from engine.experiments.contract import (
    ExperimentContractError,
    validate_baseline_contract,
)


class DatasetSplit(BaseModel):
    strategy: str
    train: str
    validation: str
    test: str


class SeedPolicy(BaseModel):
    deterministic: bool
    seeds: list[int]


class Budget(BaseModel):
    max_iters: int
    candidate_bank_size: int
    regimes: int


class Reporting(BaseModel):
    run_columns: list[str]
    aggregate: dict[str, str]
    artifact_format: str


class Significance(BaseModel):
    confidence_level: float
    test: str


class BaselineExperimentConfig(BaseModel):
    """Mirrors configs/paper/*.yaml. Rejects configs that violate the shared
    baseline experiment contract in engine.experiments.contract."""

    experiment_name: str
    dataset_split: DatasetSplit
    seed_policy: SeedPolicy
    models: list[str]
    metrics: list[str]
    reporting: Reporting
    budget: Budget
    ablations: list[str] = Field(default_factory=list)
    significance: Significance | None = None

    @model_validator(mode="after")
    def _enforce_shared_contract(self) -> BaselineExperimentConfig:
        try:
            validate_baseline_contract(
                self.model_dump(mode="json"), runner_name=self.experiment_name
            )
        except ExperimentContractError as exc:
            raise ValueError(str(exc)) from exc
        return self


def load_baseline_experiment_config(path: Path | str) -> BaselineExperimentConfig:
    with Path(path).open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    return BaselineExperimentConfig.model_validate(raw)
