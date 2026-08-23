"""The multimodal test-function problem set, per SPEC.md.

Three classic, well-established global-optimization test functions, all with
a known global minimum of 0.0 at the origin, in a fixed dimension. Rosenbrock
was in the issue's list of candidates but is dropped: standard Rosenbrock is
not genuinely multimodal (its higher-dimensional local minima only appear in
specific, non-standard variants), and SPEC.md's own criterion for picking
functions is "having multiple local minima where basin-hopping's design
intent is meant to matter" -- Rastrigin, Ackley, and Griewank all
unambiguously satisfy that; Rosenbrock's inclusion would not have.

Bounds are each function's standard literature bounds, not tuned per problem.
Dimension is fixed at 10 for all three (SPEC.md: "a fixed, low dimension...
chosen for cheap evaluation and to avoid the curse-of-dimensionality regime
where any local-search-based method degrades"), decided before any sweep ran.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

DIMENSION = 10


@dataclass(frozen=True)
class BenchmarkFunction:
    """One benchmark function: opaque to `engine/audit` beyond its bounds and callable."""

    name: str
    func: Callable[[np.ndarray], float]
    bounds: list[tuple[float, float]]
    global_optimum: float = 0.0


def rastrigin(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    return float(10 * len(x) + np.sum(x**2 - 10 * np.cos(2 * np.pi * x)))


def ackley(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    a, b, c = 20.0, 0.2, 2 * np.pi
    return float(
        -a * np.exp(-b * np.sqrt(np.mean(x**2))) - np.exp(np.mean(np.cos(c * x))) + a + np.e
    )


def griewank(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    i = np.arange(1, len(x) + 1)
    return float(1 + np.sum(x**2) / 4000 - np.prod(np.cos(x / np.sqrt(i))))


PROBLEM_SET: dict[str, BenchmarkFunction] = {
    "rastrigin": BenchmarkFunction("rastrigin", rastrigin, [(-5.12, 5.12)] * DIMENSION),
    "ackley": BenchmarkFunction("ackley", ackley, [(-32.768, 32.768)] * DIMENSION),
    "griewank": BenchmarkFunction("griewank", griewank, [(-600.0, 600.0)] * DIMENSION),
}
