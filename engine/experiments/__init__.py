"""Experiment description and the contract a reported run must satisfy.

`contract` states what a run has to record to count as evidence -- seeds,
splits, budget, environment. `config` is the schema describing one such run
before it happens.

Neither knows what is being discovered, which is why they sit here rather than
beside a dataset loader.
"""

from __future__ import annotations

from engine.experiments.config import (
    BaselineExperimentConfig,
    Budget,
    DatasetSplit,
    Reporting,
    SeedPolicy,
    Significance,
    load_baseline_experiment_config,
)

__all__ = [
    "BaselineExperimentConfig",
    "Budget",
    "DatasetSplit",
    "Reporting",
    "SeedPolicy",
    "Significance",
    "load_baseline_experiment_config",
]
