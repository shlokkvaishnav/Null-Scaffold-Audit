"""Utilities for reproducibility manifests and deterministic runtime enforcement."""

from __future__ import annotations

import json
import os
import platform
import subprocess
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Any, Dict, Iterable

import numpy as np


def _as_plain_dict(config: Any) -> Dict[str, Any]:
    """Convert OmegaConf-like objects to plain dictionaries."""
    if isinstance(config, dict):
        return config

    try:
        from omegaconf import OmegaConf  # type: ignore

        return OmegaConf.to_container(config, resolve=True)  # type: ignore[return-value]
    except Exception:
        pass

    if hasattr(config, "items"):
        return dict(config.items())

    raise TypeError("Configuration must be a mapping-like object.")


def _require(config: Dict[str, Any], path: str) -> Any:
    current: Any = config
    for key in path.split("."):
        if not isinstance(current, dict) or key not in current:
            raise ValueError(f"Missing required deterministic config field: {path}")
        current = current[key]
    return current


def validate_determinism_config(config: Any) -> Dict[str, Any]:
    """Validate required deterministic fields and return a plain dict config."""
    cfg = _as_plain_dict(config)

    deterministic = _require(cfg, "seed_policy.deterministic")
    seeds = _require(cfg, "seed_policy.seeds")
    py_hash_seed = _require(cfg, "deterministic_runtime.python_hash_seed")
    use_det_alg = _require(cfg, "deterministic_runtime.torch_use_deterministic_algorithms")
    cudnn_det = _require(cfg, "deterministic_runtime.cudnn_deterministic")
    cudnn_bench = _require(cfg, "deterministic_runtime.cudnn_benchmark")

    if deterministic is not True:
        raise ValueError("seed_policy.deterministic must be true for reproducible runs.")
    if not isinstance(seeds, list) or len(seeds) == 0:
        raise ValueError("seed_policy.seeds must be a non-empty list.")
    if not isinstance(py_hash_seed, int):
        raise ValueError("deterministic_runtime.python_hash_seed must be an integer.")
    for name, value in {
        "deterministic_runtime.torch_use_deterministic_algorithms": use_det_alg,
        "deterministic_runtime.cudnn_deterministic": cudnn_det,
        "deterministic_runtime.cudnn_benchmark": cudnn_bench,
    }.items():
        if not isinstance(value, bool):
            raise ValueError(f"{name} must be a boolean.")

    return cfg


def enforce_deterministic_runtime(config: Dict[str, Any]) -> None:
    """Apply deterministic runtime settings before heavy computation starts."""
    runtime = config["deterministic_runtime"]
    os.environ["PYTHONHASHSEED"] = str(runtime["python_hash_seed"])
    if "cublas_workspace_config" in runtime:
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = str(runtime["cublas_workspace_config"])

    np.random.seed(int(config["seed_policy"]["seeds"][0]))

    try:
        import torch

        torch.use_deterministic_algorithms(bool(runtime["torch_use_deterministic_algorithms"]))
        torch.backends.cudnn.deterministic = bool(runtime["cudnn_deterministic"])
        torch.backends.cudnn.benchmark = bool(runtime["cudnn_benchmark"])
    except Exception:
        # Torch is optional for some script paths.
        pass


def get_git_commit_hash() -> str:
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "HEAD"], text=True)
            .strip()
        )
    except Exception:
        return "unknown"


def get_package_versions(packages: Iterable[str]) -> Dict[str, str]:
    versions: Dict[str, str] = {}
    for package in packages:
        try:
            versions[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            versions[package] = "not-installed"
    return versions


def get_platform_hardware() -> Dict[str, Any]:
    details: Dict[str, Any] = {
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "machine": platform.machine(),
        "processor": platform.processor(),
    }

    try:
        import torch

        details["torch_version"] = torch.__version__
        details["cuda_available"] = bool(torch.cuda.is_available())
        details["cuda_device_count"] = int(torch.cuda.device_count())
        if torch.cuda.is_available() and torch.cuda.device_count() > 0:
            details["cuda_devices"] = [
                torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())
            ]
    except Exception:
        details["torch_version"] = "not-installed"
        details["cuda_available"] = False
        details["cuda_device_count"] = 0

    return details


def build_run_manifest(config: Dict[str, Any]) -> Dict[str, Any]:
    tracked_packages = [
        "numpy",
        "scipy",
        "scikit-learn",
        "sympy",
        "torch",
        "hydra-core",
        "pysr",
        "xgboost",
        "lightgbm",
    ]
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": get_git_commit_hash(),
        "seed_policy": config["seed_policy"],
        "deterministic_runtime": config["deterministic_runtime"],
        "package_versions": get_package_versions(tracked_packages),
        "platform_hardware": get_platform_hardware(),
    }


def log_run_manifest(config: Dict[str, Any]) -> Dict[str, Any]:
    manifest = build_run_manifest(config)
    print("[REPRODUCIBILITY_MANIFEST]")
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return manifest


def write_manifest(output_dir: Path, manifest: Dict[str, Any], stem: str) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{stem}_manifest.json"
    with path.open("w") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
    return path
