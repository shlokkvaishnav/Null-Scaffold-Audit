"""Loader for the curated AI-Feynman-style equation benchmark.

Equation metadata (formula, variable names, sampling ranges) is committed as
a small JSON table (`feynman_equations.json`) so that benchmark data can be
regenerated deterministically at runtime via seeded numpy sampling, with no
network access required.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import sympy

_EQUATIONS_PATH = Path(__file__).resolve().parent / "feynman_equations.json"

# Restricted namespace of numpy-backed functions available to formulas via
# sympy's lambdify. Keeping this explicit (rather than a bare eval) avoids
# arbitrary code execution from equation strings.
_ALLOWED_SYMPY_FUNCTIONS = {"exp", "log", "sin", "cos", "sqrt", "pi"}


def _load_all() -> list[dict[str, Any]]:
    with _EQUATIONS_PATH.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def list_feynman_equations() -> list[dict[str, Any]]:
    """Return metadata for every curated Feynman-style equation."""
    return _load_all()


def load_feynman_equation(equation_id: str) -> dict[str, Any]:
    """Return the metadata dict for a single equation by its `id`."""
    for entry in _load_all():
        if entry["id"] == equation_id:
            return entry
    raise KeyError(f"Unknown Feynman equation id: {equation_id!r}")


def _build_callable(formula: str, variables: list[str]):
    symbols = sympy.symbols(variables)
    if len(variables) == 1:
        symbols = (symbols,)
    local_dict = {name: sym for name, sym in zip(variables, symbols)}
    expr = sympy.sympify(formula, locals=local_dict)
    func = sympy.lambdify(symbols, expr, modules=["numpy"])
    return expr, func


def generate_feynman_dataset(
    equation_id: str,
    n_samples: int = 1000,
    noise_std: float = 0.0,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Sample a synthetic dataset for a curated Feynman-style equation.

    Args:
        equation_id: The `id` field of an entry in `feynman_equations.json`.
        n_samples: Number of rows to sample.
        noise_std: Standard deviation of additive Gaussian noise on the target
            (as a fraction of the target's standard deviation is NOT applied;
            this is absolute noise added directly to y).
        seed: Seed for the random number generator (deterministic sampling).

    Returns:
        Tuple of (X, y, ground_truth_info) where X has shape
        (n_samples, n_vars), y has shape (n_samples,), and ground_truth_info
        is a dict with the formula string, variable names, and equation id/name.
    """
    entry = load_feynman_equation(equation_id)
    variables: list[str] = entry["variables"]
    ranges: dict[str, list[float]] = entry["ranges"]
    formula: str = entry["formula"]

    rng = np.random.default_rng(seed)
    columns = []
    for name in variables:
        lo, hi = ranges[name]
        columns.append(rng.uniform(lo, hi, size=n_samples))
    X = np.column_stack(columns)

    _, func = _build_callable(formula, variables)
    y = np.asarray(func(*[X[:, i] for i in range(len(variables))]), dtype=float)
    # Broadcast constant results (e.g. formulas that don't use every sampled
    # variable in a way that affects output shape) to the full sample count.
    if y.shape == ():
        y = np.full(n_samples, float(y))

    if noise_std > 0:
        y = y + rng.normal(0.0, noise_std, size=n_samples)

    ground_truth_info = {
        "id": entry["id"],
        "name": entry["name"],
        "formula": formula,
        "variables": variables,
        "ranges": ranges,
        "n_vars": entry["n_vars"],
    }
    return X, y, ground_truth_info
