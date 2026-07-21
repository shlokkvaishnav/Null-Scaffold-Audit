"""In-memory dataset/job stores and the background discovery task.

Deliberately simple: a process-local dict for datasets and a process-local
dict for jobs. No Celery/Redis/DB -- FastAPI's ``BackgroundTasks`` plus an
in-memory store is sufficient for this project's scope (single-process
demo/portfolio service, not a production multi-worker deployment).
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Tuple

import numpy as np

from equation_discovery.core.agent import DiscoveryAgent
from equation_discovery.evaluation.metrics import compute_fit_metrics

# dataset_id -> (X, y, feature_names)
DATASETS: Dict[str, Tuple[np.ndarray, np.ndarray, List[str]]] = {}

# job_id -> {"status", "created_at", "equation", "rmse", "confidence", "error"}
JOBS: Dict[str, Dict[str, Any]] = {}


def register_dataset(x: np.ndarray, y: np.ndarray, feature_names: List[str]) -> str:
    """Store a loaded dataset in memory and return its generated id."""
    dataset_id = str(uuid.uuid4())
    DATASETS[dataset_id] = (x, y, feature_names)
    return dataset_id


def create_job() -> str:
    """Register a new pending job and return its id."""
    job_id = str(uuid.uuid4())
    JOBS[job_id] = {
        "status": "pending",
        "created_at": datetime.now(timezone.utc),
        "equation": None,
        "rmse": None,
        "confidence": None,
        "error": None,
    }
    return job_id


def _train_test_split(x: np.ndarray, y: np.ndarray, train_fraction: float = 0.8):
    split = max(1, int(train_fraction * len(y)))
    split = min(split, len(y) - 1) if len(y) > 1 else len(y)
    return x[:split], y[:split], x[split:], y[split:]


def run_discovery_job(
    job_id: str,
    dataset_id: str,
    backend: str = "gplearn",
    max_iters: int | None = None,
) -> None:
    """Fit a symbolic equation to a stored dataset and record the result.

    Intended to be scheduled via ``BackgroundTasks.add_task``. Runs
    synchronously within the background task (no separate worker process).
    """
    job = JOBS.get(job_id)
    if job is None:
        return

    job["status"] = "running"

    try:
        if dataset_id not in DATASETS:
            raise KeyError(f"Unknown dataset_id: {dataset_id!r}")

        x, y, _feature_names = DATASETS[dataset_id]
        x_train, y_train, x_test, y_test = _train_test_split(x, y)
        if len(x_test) == 0:
            x_test, y_test = x_train, y_train

        # Runs the full DiscoveryAgent loop (observe->retrieve->reason->
        # verify->learn), not just a bare symbolic-regression fit -- this is
        # the actual product surface for the agent, matching the benchmark's
        # "discovery_agent" runner (evaluation/benchmark_runner.py).
        # num_regimes=1: an uploaded dataset is treated as one global
        # equation, not a regime-switching system.
        agent_cfg: Dict[str, Any] = {
            "agent": {
                "num_regimes": 1,
                "use_verification": True,
                "use_memory": True,
                "use_belief": True,
                "use_reasoning": True,
                "reasoning_mode": backend,
            }
        }
        if max_iters is not None:
            agent_cfg["agent"]["generations"] = max_iters
            agent_cfg["agent"]["niterations"] = max_iters

        agent = DiscoveryAgent(agent_cfg)
        obs = {"features": x_train, "targets": y_train}
        for _ in range(3):
            agent.step(obs)

        if not agent.memory or not agent.memory.hypotheses:
            raise RuntimeError("DiscoveryAgent produced no hypotheses.")

        best = max(agent.memory.hypotheses, key=lambda h: getattr(h, "score", float("-inf")))
        equation = str(best.equation)
        y_pred = best.evaluate(x_test)

        fit_metrics = compute_fit_metrics(y_test, y_pred, equation=equation)
        rmse = fit_metrics["rmse"]

        # Confidence proxy: maps RMSE (relative to the target's own spread)
        # into (0, 1], so a near-perfect fit -> confidence near 1 and a
        # poor fit -> confidence near 0. Not a calibrated probability, just
        # a simple monotonic proxy for "how good is this fit".
        target_scale = float(np.std(y_test)) + 1e-8
        normalized_rmse = rmse / target_scale
        confidence = 1.0 / (1.0 + normalized_rmse)

        job["status"] = "done"
        job["equation"] = equation
        job["rmse"] = rmse
        job["confidence"] = confidence
    except Exception as exc:  # noqa: BLE001 - job must record failure, not raise
        job["status"] = "failed"
        job["error"] = str(exc)
