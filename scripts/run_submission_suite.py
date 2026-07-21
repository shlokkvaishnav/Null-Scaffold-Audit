"""Submission benchmark suite runner.

Runs train+eval across all benchmark model variants declared in configs/paper/*.yaml.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import yaml

from physics_discovery.core.agent import DiscoveryAgent
from physics_discovery.data.synthetic import SplitData as DatasetSplit
from physics_discovery.data.synthetic import generate_synthetic_regression
from physics_discovery.evaluation.metrics import compute_fit_metrics, confidence_interval, paired_significance
from physics_discovery.generators.baselines import BaselineModel
from physics_discovery.generators.ensemble import Ensemble
from physics_discovery.generators.symbolic import SymbolicHypothesisGenerator


def _hardware_metadata() -> Dict[str, Any]:
    metadata = {
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
    }
    try:
        import torch

        metadata["torch_version"] = torch.__version__
        metadata["cuda_available"] = bool(torch.cuda.is_available())
        metadata["cuda_device_count"] = int(torch.cuda.device_count()) if torch.cuda.is_available() else 0
    except Exception:
        metadata["torch_version"] = None
        metadata["cuda_available"] = False
        metadata["cuda_device_count"] = 0
    return metadata


def _run_pysr_global(data: DatasetSplit, seed: int) -> Tuple[Any, str]:
    model = SymbolicHypothesisGenerator(
        {
            "backend": "gplearn",
            "generations": 25,
            "population_size": 400,
            "random_state": seed,
        }
    )
    model.fit(data.x_train, data.y_train)
    return model.predict(data.x_test), model.equation


def _run_neural_moe(data: DatasetSplit, seed: int) -> Tuple[Any, str]:
    model = Ensemble({"random_state": seed, "n_estimators": 250, "max_iter": 400})
    model.fit(data.x_train, data.y_train)
    return model.predict(data.x_test), "neural_moe_ensemble"


def _run_baseline(data: DatasetSplit, seed: int, model_type: str) -> Tuple[Any, str]:
    model = BaselineModel({"model_type": model_type, "random_state": seed})
    model.fit(data.x_train, data.y_train)
    return model.predict(data.x_test), model_type


def _run_discovery_agent(data: DatasetSplit, seed: int, variant: str, budget: Dict[str, Any]) -> Tuple[Any, str]:
    agent_cfg: Dict[str, Any] = {
        "agent": {
            "num_regimes": int(budget.get("regimes", 3)),
            "use_verification": True,
            "use_memory": True,
            "use_belief": True,
            "use_reasoning": True,
            "reasoning_mode": "gplearn",
        }
    }

    if variant == "no_scoring":
        agent_cfg["agent"]["use_verification"] = False
    elif variant == "no_confidence_tracking":
        agent_cfg["agent"]["use_belief"] = False
    elif variant == "no_validation":
        agent_cfg["agent"]["use_verification"] = False
    elif variant != "discovery_agent_full":
        raise NotImplementedError(f"Missing discovery-agent wiring for variant: {variant}")

    agent = DiscoveryAgent(agent_cfg)
    max_iters = int(budget.get("max_iters", 10))
    obs = {"features": data.x_train, "targets": data.y_train}
    for _ in range(max_iters):
        agent.step(obs)

    if not agent.memory or not agent.memory.hypotheses:
        raise RuntimeError(f"Variant '{variant}' produced no hypotheses; implementation wiring appears broken.")

    best = max(agent.memory.hypotheses, key=lambda h: getattr(h, "score", float("-inf")))
    y_pred = best.evaluate(data.x_test)
    return y_pred, str(best.equation)


def _runner_for_variant(variant: str):
    if variant == "pysr_global":
        return _run_pysr_global
    if variant == "neural_moe":
        return _run_neural_moe
    if variant in {"lightgbm", "xgboost"}:
        return lambda data, seed, _variant=variant: _run_baseline(data, seed, model_type=_variant)
    if variant in {"discovery_agent_full", "no_scoring", "no_confidence_tracking", "no_validation"}:
        return None
    raise NotImplementedError(f"Missing implementation wiring for model variant: {variant}")


def _write_csv(rows: List[Dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def run_config(config_path: Path, output_root: Path) -> Dict[str, str]:
    with config_path.open() as handle:
        config = yaml.safe_load(handle)

    if "models" not in config:
        return {}

    experiment = config["experiment_name"]
    seeds = config["seed_policy"]["seeds"]
    metrics = config["metrics"]
    confidence_level = float(config.get("significance", {}).get("confidence_level", 0.95))
    variants = list(dict.fromkeys(config.get("models", []) + config.get("ablations", [])))

    for variant in variants:
        if variant.startswith("discovery_agent") or variant.startswith("no_"):
            continue
        _runner_for_variant(variant)

    output_dir = output_root / experiment
    output_dir.mkdir(parents=True, exist_ok=True)

    hardware = _hardware_metadata()
    per_seed_rows: List[Dict[str, Any]] = []

    for variant in variants:
        for seed in seeds:
            data = generate_synthetic_regression(seed=seed)
            started = time.perf_counter()

            if variant.startswith("discovery_agent") or variant.startswith("no_"):
                y_pred, equation = _run_discovery_agent(
                    data, seed=seed, variant=variant, budget=config.get("budget", {})
                )
            else:
                runner = _runner_for_variant(variant)
                y_pred, equation = runner(data, seed)

            duration = time.perf_counter() - started
            model_metrics = compute_fit_metrics(data.y_test, y_pred, equation=equation)
            model_metrics["runtime_seconds"] = float(duration)

            row = {
                "experiment": experiment,
                "config_path": str(config_path),
                "model": variant,
                "seed": int(seed),
                "equation": equation,
                **{m: float(model_metrics[m]) for m in model_metrics},
                "hardware": json.dumps(hardware, sort_keys=True),
            }
            per_seed_rows.append(row)

    summary_rows: List[Dict[str, Any]] = []
    for variant in variants:
        subset = [r for r in per_seed_rows if r["model"] == variant]
        for metric in metrics + ["runtime_seconds"]:
            values = [float(r[metric]) for r in subset]
            mean, ci_lo, ci_hi = confidence_interval(values, confidence_level)
            summary_rows.append(
                {
                    "experiment": experiment,
                    "model": variant,
                    "metric": metric,
                    "mean": mean,
                    "ci_low": ci_lo,
                    "ci_high": ci_hi,
                    "n": len(values),
                    "confidence_level": confidence_level,
                }
            )

    significance_rows = paired_significance(
        per_seed_rows,
        metrics=[m for m in metrics if m in {"rmse", "mae", "calibration_error", "symbolic_complexity", "runtime_seconds"}],
        baseline="discovery_agent_full" if "discovery_agent_full" in variants else variants[0],
    )

    per_seed_json = output_dir / "per_seed_metrics.json"
    per_seed_csv = output_dir / "per_seed_metrics.csv"
    summary_json = output_dir / "aggregate_summary.json"
    summary_csv = output_dir / "aggregate_summary.csv"
    significance_json = output_dir / "significance_tests.json"
    significance_csv = output_dir / "significance_tests.csv"
    runtime_json = output_dir / "runtime_hardware_metadata.json"

    per_seed_json.write_text(json.dumps(per_seed_rows, indent=2))
    _write_csv(per_seed_rows, per_seed_csv)
    summary_json.write_text(json.dumps(summary_rows, indent=2))
    _write_csv(summary_rows, summary_csv)
    significance_json.write_text(json.dumps(significance_rows, indent=2))
    _write_csv(significance_rows, significance_csv)

    runtime_payload = {
        "experiment": experiment,
        "hardware": hardware,
        "runtime_seconds_total": float(sum(r["runtime_seconds"] for r in per_seed_rows)),
        "runtime_seconds_by_model": {
            model: float(sum(r["runtime_seconds"] for r in per_seed_rows if r["model"] == model))
            for model in variants
        },
    }
    runtime_json.write_text(json.dumps(runtime_payload, indent=2))

    return {
        "config": str(config_path),
        "per_seed_json": str(per_seed_json),
        "per_seed_csv": str(per_seed_csv),
        "summary_json": str(summary_json),
        "summary_csv": str(summary_csv),
        "significance_json": str(significance_json),
        "significance_csv": str(significance_csv),
        "runtime_json": str(runtime_json),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run benchmark submission suite for all paper configs.")
    parser.add_argument("--config-glob", default="configs/paper/*.yaml", help="Glob for benchmark config files.")
    parser.add_argument("--output-root", default="results/submission_suite", help="Directory for run artifacts.")
    args = parser.parse_args()

    config_paths = sorted(Path(".").glob(args.config_glob))
    if not config_paths:
        raise FileNotFoundError(f"No configs found for glob: {args.config_glob}")

    outputs = []
    for config_path in config_paths:
        result = run_config(config_path=config_path, output_root=Path(args.output_root))
        if result:
            outputs.append(result)

    index_path = Path(args.output_root) / "run_index.json"
    index_path.parent.mkdir(parents=True, exist_ok=True)
    index_path.write_text(json.dumps(outputs, indent=2))
    print(json.dumps({"runs": outputs, "index": str(index_path)}, indent=2))


if __name__ == "__main__":
    main()
