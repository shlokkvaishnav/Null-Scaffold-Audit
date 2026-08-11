"""Equivalence checking between a discovered candidate equation and ground truth.

A symbolic-regression model rarely returns an expression that's textually
identical to the ground-truth formula (different variable names, reordered
terms, small numeric-fit constants rather than exact ones). This module
determines whether a candidate equation is *equivalent* to a ground-truth
formula via two complementary checks:

1. Symbolic equivalence: simplify(candidate - ground_truth) == 0 after
   positionally mapping generic variable names (x0, x1, ...) onto the
   ground truth's variable list if needed.
2. Numeric equivalence: evaluate both expressions at many seeded random
   points within the documented variable ranges and check they agree within
   a relative + absolute tolerance. This is the more robust/primary signal
   since fitted symbolic-regression models are numeric approximations.
"""

from __future__ import annotations

import re
from typing import Any

import numpy as np
import sympy

from physics_discovery.core.expression_eval import GPLEARN_FUNCTIONS as _GPLEARN_FUNCTIONS


def _remap_generic_variable_names(expr_str: str, variables: list[str]) -> str:
    """Rewrite generic x0, x1, ... variable names to the ground-truth's variable names.

    Symbolic regression backends (gplearn, etc.) commonly emit variables named
    X0, X1, ... or x0, x1, .... This maps them positionally onto `variables`
    so the candidate can be compared against a formula written in the
    ground-truth's own variable names.
    """
    remapped = expr_str
    # Replace longer indices first (x10 before x1) to avoid partial overlaps.
    indices = sorted(range(len(variables)), key=lambda i: -i)
    for i in indices:
        if i >= len(variables):
            continue
        pattern = re.compile(rf"\b[Xx]{i}\b")
        remapped = pattern.sub(f"__VAR_{i}__", remapped)
    for i in indices:
        remapped = remapped.replace(f"__VAR_{i}__", variables[i])
    return remapped


def _safe_sympify(expr_str: str, variables: list[str]):
    symbols = {name: sympy.Symbol(name) for name in variables}
    return sympy.sympify(expr_str, locals={**_GPLEARN_FUNCTIONS, **symbols})


def _symbolic_equivalence(candidate: str, ground_truth_formula: str, variables: list[str]) -> bool:
    try:
        candidate_remapped = _remap_generic_variable_names(candidate, variables)
        candidate_expr = _safe_sympify(candidate_remapped, variables)
        truth_expr = _safe_sympify(ground_truth_formula, variables)
        diff = sympy.simplify(candidate_expr - truth_expr)
        return bool(diff == 0)
    # Blanket by necessity: sympify/simplify over a search-generated string
    # raise no single documented type -- SympifyError, TypeError,
    # AttributeError and RecursionError all occur. An equivalence check that
    # could not run has not shown the candidate correct, so False is the
    # conservative answer for every one of them.
    except Exception:  # noqa: BLE001
        return False


def _numeric_equivalence(
    candidate: str,
    ground_truth_formula: str,
    variables: list[str],
    test_ranges: dict[str, list[float]],
    n_check_points: int,
    seed: int,
    rtol: float,
    atol: float,
) -> dict[str, Any]:
    try:
        candidate_remapped = _remap_generic_variable_names(candidate, variables)
        symbols = sympy.symbols(variables)
        if len(variables) == 1:
            symbols = (symbols,)
        local_dict = {name: sym for name, sym in zip(variables, symbols)}
        local_dict_with_functions = {**_GPLEARN_FUNCTIONS, **local_dict}

        candidate_expr = sympy.sympify(candidate_remapped, locals=local_dict_with_functions)
        truth_expr = sympy.sympify(ground_truth_formula, locals=local_dict_with_functions)

        candidate_fn = sympy.lambdify(symbols, candidate_expr, modules=["numpy"])
        truth_fn = sympy.lambdify(symbols, truth_expr, modules=["numpy"])

        rng = np.random.default_rng(seed)
        columns = []
        for name in variables:
            lo, hi = test_ranges[name]
            columns.append(rng.uniform(lo, hi, size=n_check_points))
        X = np.column_stack(columns)

        with np.errstate(all="ignore"):
            candidate_vals = np.asarray(candidate_fn(*[X[:, i] for i in range(len(variables))]), dtype=float)
            truth_vals = np.asarray(truth_fn(*[X[:, i] for i in range(len(variables))]), dtype=float)

        if candidate_vals.shape == ():
            candidate_vals = np.full(n_check_points, float(candidate_vals))
        if truth_vals.shape == ():
            truth_vals = np.full(n_check_points, float(truth_vals))

        finite_mask = np.isfinite(candidate_vals) & np.isfinite(truth_vals)
        if not np.any(finite_mask):
            return {"numeric_match": False, "max_relative_error": float("inf")}

        candidate_vals = candidate_vals[finite_mask]
        truth_vals = truth_vals[finite_mask]

        match = bool(np.allclose(candidate_vals, truth_vals, rtol=rtol, atol=atol))
        rel_error = np.abs(candidate_vals - truth_vals) / (np.abs(truth_vals) + atol)
        max_rel_error = float(np.max(rel_error)) if rel_error.size else float("inf")

        return {"numeric_match": match, "max_relative_error": max_rel_error}
    # Same reasoning as the symbolic path: evaluating two arbitrary expressions
    # over sampled ranges can fail in numpy or in sympy, and a check that could
    # not run is reported as "no match" rather than crashing a sweep.
    except Exception:  # noqa: BLE001
        return {"numeric_match": False, "max_relative_error": float("inf")}


def check_equivalence(
    candidate_equation: str,
    ground_truth_formula: str,
    variables: list[str],
    test_ranges: dict[str, list[float]],
    n_check_points: int = 200,
    seed: int = 0,
    rtol: float = 1e-3,
    atol: float = 1e-6,
) -> dict[str, Any]:
    """Check whether a discovered candidate equation is equivalent to ground truth.

    Args:
        candidate_equation: Candidate equation string (may use generic
            variable names like x0, x1, ... or the ground truth's own names).
        ground_truth_formula: The ground-truth formula string, written in
            terms of `variables`.
        variables: Ordered list of ground-truth variable names. If the
            candidate uses generic names (x0, x1, ...), they are mapped
            positionally onto this list.
        test_ranges: Dict mapping each variable name to a [min, max] range
            used for numeric sampling.
        n_check_points: Number of random points to sample for the numeric check.
        seed: Seed for the random number generator used in numeric sampling.
        rtol: Relative tolerance for the numeric equivalence check.
        atol: Absolute tolerance for the numeric equivalence check.

    Returns:
        Dict with keys: symbolic_match (bool), numeric_match (bool),
        max_relative_error (float).
    """
    symbolic_match = _symbolic_equivalence(candidate_equation, ground_truth_formula, variables)
    numeric_result = _numeric_equivalence(
        candidate_equation,
        ground_truth_formula,
        variables,
        test_ranges,
        n_check_points,
        seed,
        rtol,
        atol,
    )

    return {
        "symbolic_match": symbolic_match,
        "numeric_match": numeric_result["numeric_match"],
        "max_relative_error": numeric_result["max_relative_error"],
    }
