"""Shared experiment contract for baseline comparability."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, Mapping

BASELINE_EXPERIMENT_CONTRACT: Dict[str, Any] = {
    "dataset_split": {
        "strategy": "canonical_time_split",
        "train": "train",
        "validation": "validation",
        "test": "test",
    },
    "seed_policy": {
        "deterministic": True,
        "seeds": [7, 11, 23, 47, 101, 131, 181, 223, 269, 307],
    },
    "budget": {
        "max_iters": 25,
        "candidate_bank_size": 20,
        "regimes": 4,
    },
    "metrics": [
        "rmse",
        "mae",
        "calibration_error",
        "symbolic_complexity",
        "runtime_seconds",
    ],
    "reporting": {
        "run_columns": ["seed", "model", "split"],
        "aggregate": {
            "mean": "float",
            "ci95": "float",
        },
        "artifact_format": "json+csv",
    },
}


class ExperimentContractError(ValueError):
    """Raised when a baseline run config violates the shared contract."""


def _expect_equal(config: Mapping[str, Any], key: str, expected: Any) -> None:
    actual = config.get(key)
    if actual != expected:
        raise ExperimentContractError(
            f"Contract mismatch for '{key}': expected {expected!r}, got {actual!r}"
        )


def validate_baseline_contract(config: Mapping[str, Any], *, runner_name: str = "baseline") -> None:
    """Validate that a baseline runner config matches the shared experiment contract."""
    for key, expected in BASELINE_EXPERIMENT_CONTRACT.items():
        _expect_equal(config, key, expected)

    if not config.get("models"):
        raise ExperimentContractError(
            f"Runner '{runner_name}' has no models configured; cannot execute baseline contract."
        )


def apply_contract(config: Mapping[str, Any]) -> Dict[str, Any]:
    """Return a copy of config with the canonical baseline contract enforced."""
    merged = deepcopy(dict(config))
    for key, value in BASELINE_EXPERIMENT_CONTRACT.items():
        merged[key] = deepcopy(value)
    return merged
