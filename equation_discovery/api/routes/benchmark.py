"""Feynman benchmark endpoint.

Only the "smoke" subset is exposed synchronously here, since it runs in
well under a minute. The full ~30-120 equation subset is intentionally
NOT exposed via this endpoint -- it is slow (many symbolic-regression
fits) and would block a request/worker for minutes. Run it instead via the
CLI:

    python -m equation_discovery.evaluation.benchmark_runner --subset all
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query

from equation_discovery.evaluation.benchmark_runner import run_benchmark

router = APIRouter(prefix="/benchmark", tags=["benchmark"])


@router.get("/feynman")
async def run_feynman_benchmark(
    subset: str = Query(
        default="smoke",
        description=(
            "Benchmark subset to run. Only 'smoke' is allowed here since it "
            "is fast; use the benchmark_runner CLI for 'all'."
        ),
    ),
    backend: str = Query(default="gplearn"),
    seed: int = Query(default=0),
) -> dict:
    """Synchronously run the smoke-subset Feynman rediscovery benchmark and return results."""
    if subset != "smoke":
        raise HTTPException(
            status_code=400,
            detail=(
                "Only subset='smoke' is supported via this endpoint (fast, ~8 equations). "
                "Run 'python -m equation_discovery.evaluation.benchmark_runner --subset all' "
                "for the full benchmark."
            ),
        )

    results = run_benchmark(subset=subset, backend=backend, seed=seed)
    n_symbolic_matches = sum(1 for r in results if r["symbolic_match"])
    n_numeric_matches = sum(1 for r in results if r["numeric_match"])
    return {
        "subset": subset,
        "backend": backend,
        "n_rows": len(results),
        "n_symbolic_matches": n_symbolic_matches,
        "n_numeric_matches": n_numeric_matches,
        "results": results,
    }
