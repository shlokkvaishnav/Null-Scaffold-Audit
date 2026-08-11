"""Feynman rediscovery benchmark orchestrator.

For each curated Feynman-style equation, generates synthetic data, fits a
small set of hypothesis generators/baselines, scores each against the
ground-truth equation (fit metrics + symbolic/numeric equivalence), and
writes JSON+CSV result artifacts.

Usage:
    python -m plugins.physics.benchmark_runner --subset smoke
    python -m plugins.physics.benchmark_runner --subset all --backend gplearn

Three models are compared per equation:
- "symbolic_regression": the raw symbolic-regression generator, called directly.
- "discovery_agent": the full DiscoveryAgent loop (observe->retrieve->reason->
  verify->learn), using a single-regime config profile appropriate for a
  dataset with one global equation (no regime-switching structure) -- the
  same agent_cfg pattern `scripts/run_submission_suite.py::_run_discovery_agent`
  uses for its own multi-regime variants, simplified to num_regimes=1.
- "gbm_baseline": a gradient-boosted-tree baseline with no symbolic structure.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
from typing import Any

from algorithms.baselines import BaselineModel
from algorithms.symbolic import SymbolicHypothesisGenerator
from engine.evaluation.equivalence import check_equivalence
from engine.evaluation.metrics import compute_fit_metrics
from plugins.physics.feynman_loader import generate_feynman_dataset, list_feynman_equations
from plugins.physics.scaffold.agent import DiscoveryAgent

SMOKE_SUBSET_SIZE = 8


def _select_equations(subset: str) -> list[dict[str, Any]]:
    equations = list_feynman_equations()
    if subset == "smoke":
        return equations[:SMOKE_SUBSET_SIZE]
    if subset == "all":
        return equations
    raise ValueError(f"Unknown subset: {subset!r}. Expected 'smoke' or 'all'.")


def _train_test_split(X, y, train_fraction: float = 0.8):
    split = int(train_fraction * len(y))
    return X[:split], y[:split], X[split:], y[split:]


def _run_symbolic(x_train, y_train, x_test, backend: str, seed: int):
    model = SymbolicHypothesisGenerator({"backend": backend, "random_state": seed})
    model.fit(x_train, y_train)
    return model.predict(x_test), model.equation


def _run_gbm_baseline(x_train, y_train, x_test, seed: int):
    model = BaselineModel({"model_type": "lightgbm", "random_state": seed})
    model.fit(x_train, y_train)
    return model.predict(x_test), "gbm_baseline"


def _run_discovery_agent(x_train, y_train, x_test, backend: str, seed: int, max_iters: int = 3):
    """Run the full DiscoveryAgent loop on a single-global-equation dataset.

    num_regimes=1 since a Feynman equation is one global closed form, not a
    regime-switching system -- the multi-regime path exists for datasets
    with piecewise structure, which isn't this benchmark's shape.
    """
    agent_cfg: dict[str, Any] = {
        "agent": {
            "num_regimes": 1,
            "use_verification": True,
            "use_memory": True,
            "use_belief": True,
            "use_reasoning": True,
            "reasoning_mode": backend,
            "random_state": seed,
        }
    }
    agent = DiscoveryAgent(agent_cfg)
    obs = {"features": x_train, "targets": y_train}
    for _ in range(max_iters):
        agent.step(obs)

    if not agent.memory or not agent.memory.hypotheses:
        raise RuntimeError("DiscoveryAgent produced no hypotheses.")

    best = max(agent.memory.hypotheses, key=lambda h: getattr(h, "score", float("-inf")))
    return best.evaluate(x_test), str(best.equation)


def run_benchmark(
    subset: str = "smoke",
    backend: str = "gplearn",
    seed: int = 0,
    n_samples: int = 500,
) -> list[dict[str, Any]]:
    """Run the Feynman rediscovery benchmark and return a results table (list of dicts)."""
    equations = _select_equations(subset)
    results: list[dict[str, Any]] = []

    runners = {
        "symbolic_regression": lambda x_tr, y_tr, x_te: _run_symbolic(x_tr, y_tr, x_te, backend, seed),
        "discovery_agent": lambda x_tr, y_tr, x_te: _run_discovery_agent(x_tr, y_tr, x_te, backend, seed),
        "gbm_baseline": lambda x_tr, y_tr, x_te: _run_gbm_baseline(x_tr, y_tr, x_te, seed),
    }

    for entry in equations:
        equation_id = entry["id"]
        X, y, ground_truth = generate_feynman_dataset(equation_id, n_samples=n_samples, seed=seed)
        x_train, y_train, x_test, y_test = _train_test_split(X, y)

        for model_name, runner in runners.items():
            started = time.perf_counter()
            try:
                y_pred, candidate_equation = runner(x_train, y_train, x_test)
                runtime_seconds = time.perf_counter() - started
                fit_metrics = compute_fit_metrics(y_test, y_pred, equation=candidate_equation)
                equivalence = check_equivalence(
                    candidate_equation=candidate_equation,
                    ground_truth_formula=ground_truth["formula"],
                    variables=ground_truth["variables"],
                    test_ranges=ground_truth["ranges"],
                    seed=seed,
                )
                row = {
                    "equation_id": equation_id,
                    "equation_name": ground_truth["name"],
                    "model": model_name,
                    "candidate_equation": candidate_equation,
                    "rmse": fit_metrics["rmse"],
                    "mae": fit_metrics["mae"],
                    "calibration_error": fit_metrics["calibration_error"],
                    "symbolic_complexity": fit_metrics["symbolic_complexity"],
                    "symbolic_match": equivalence["symbolic_match"],
                    "numeric_match": equivalence["numeric_match"],
                    "max_relative_error": equivalence["max_relative_error"],
                    "runtime_seconds": runtime_seconds,
                    "error": None,
                }
            except Exception as exc:  # noqa: BLE001 - benchmark must keep going on model failure
                runtime_seconds = time.perf_counter() - started
                row = {
                    "equation_id": equation_id,
                    "equation_name": ground_truth["name"],
                    "model": model_name,
                    "candidate_equation": None,
                    "rmse": float("nan"),
                    "mae": float("nan"),
                    "calibration_error": float("nan"),
                    "symbolic_complexity": float("nan"),
                    "symbolic_match": False,
                    "numeric_match": False,
                    "max_relative_error": float("inf"),
                    "runtime_seconds": runtime_seconds,
                    "error": str(exc),
                }
            results.append(row)

    return results


def _write_artifacts(results: list[dict[str, Any]], output_root: Path) -> dict[str, str]:
    output_root.mkdir(parents=True, exist_ok=True)
    json_path = output_root / "results.json"
    csv_path = output_root / "results.csv"

    json_path.write_text(json.dumps(results, indent=2))

    if results:
        with csv_path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(results[0].keys()))
            writer.writeheader()
            writer.writerows(results)

    return {"json": str(json_path), "csv": str(csv_path)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Feynman equation-rediscovery benchmark.")
    parser.add_argument("--subset", choices=["smoke", "all"], default="smoke")
    parser.add_argument("--backend", choices=["gplearn", "pysr"], default="gplearn")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-samples", type=int, default=500)
    parser.add_argument(
        "--output-root",
        default="results/feynman_benchmark",
        help="Directory to write results.json/results.csv into.",
    )
    args = parser.parse_args()

    results = run_benchmark(
        subset=args.subset,
        backend=args.backend,
        seed=args.seed,
        n_samples=args.n_samples,
    )
    artifacts = _write_artifacts(results, Path(args.output_root))

    n_symbolic_matches = sum(1 for r in results if r["symbolic_match"])
    n_numeric_matches = sum(1 for r in results if r["numeric_match"])
    print(
        json.dumps(
            {
                "subset": args.subset,
                "backend": args.backend,
                "n_rows": len(results),
                "n_symbolic_matches": n_symbolic_matches,
                "n_numeric_matches": n_numeric_matches,
                "artifacts": artifacts,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
