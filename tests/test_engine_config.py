"""Tests for the two config schemas: engine.config (RunConfig) and the paper
benchmark-config shape, validated against the shared baseline contract.

The two are tested together because the same CLI loads both, but they live in
different packages on purpose: the baseline shape encodes one plugin's
experiment protocol and so may not sit in the engine and so is domain-specific."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from engine.config import load_run_config
from engine.experiments.config import load_baseline_experiment_config

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_load_run_config_from_yaml() -> None:
    config = load_run_config(REPO_ROOT / "configs" / "run" / "coulomb_symbolic.yaml")

    assert config.domain == "feynman_physics"
    assert config.algorithm == "symbolic_regression"
    assert config.domain_kwargs["equation_id"] == "coulomb_force"
    assert config.algorithm_kwargs["backend"] == "gplearn"

    experiment_config = config.to_experiment_config()
    assert experiment_config.domain == "feynman_physics"
    assert experiment_config.algorithm == "symbolic_regression"


@pytest.mark.parametrize("filename", ["benchmark_minimal.yaml", "benchmark_full.yaml"])
def test_existing_paper_configs_pass_validation(filename: str) -> None:
    config = load_baseline_experiment_config(REPO_ROOT / "configs" / "paper" / filename)
    assert config.dataset_split.strategy == "seeded_random_split"
    assert config.budget.max_iters == 25
    assert config.seed_policy.seeds[0] == 7


def test_baseline_config_rejects_contract_violation(tmp_path: Path) -> None:
    bad_yaml = tmp_path / "bad.yaml"
    bad_yaml.write_text(
        """
experiment_name: bad_experiment
dataset_split:
  strategy: seeded_random_split
  train: train
  validation: validation
  test: test
seed_policy:
  deterministic: true
  seeds: [1, 2, 3]
models: [linear]
metrics: [rmse]
reporting:
  run_columns: [seed, model, split]
  aggregate: {mean: float, ci95: float}
  artifact_format: json+csv
budget:
  max_iters: 999
  candidate_bank_size: 20
  regimes: 4
"""
    )
    with pytest.raises(ValidationError):
        load_baseline_experiment_config(bad_yaml)
