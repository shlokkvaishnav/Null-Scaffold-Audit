"""Equivalence checking between a discovered candidate equation and ground truth.

A symbolic-regression model rarely returns an expression that's textually
identical to the ground-truth formula (different variable names, reordered
terms, small numeric-fit constants rather than exact ones). This module
determines whether a candidate equation is *equivalent* to a ground-truth
formula via two complementary checks:

1. Symbolic equivalence, following SRBench: the candidate counts as a
   rediscovery when simplify(candidate - ground_truth) or
   simplify(candidate / ground_truth) is a constant, after positionally
   mapping generic variable names (x0, x1, ...) onto the ground truth's
   variable list if needed. Strict identity is reported separately as
   `strict_match`.

   The relaxation is not leniency for its own sake. A symbolic regressor
   fits its constants numerically, so a run that returns 2.0001*x where the
   truth is 2*x has recovered the law and missed a decimal; scoring that as
   failure measures the optimiser rather than the discovery. Matching
   SRBench's rule is also what makes a recovery rate here comparable to a
   published one -- under a stricter rule, ours would read as worse than
   other methods for reasons that have nothing to do with the method.
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

from engine.expressions.expression_eval import GPLEARN_FUNCTIONS as _GPLEARN_FUNCTIONS

# How flat a sampled difference or ratio must be to count as "possibly
# constant", relative to the target's own spread. Deliberately loose: this only
# decides whether to pay for the exact symbolic test, and being too strict here
# would discard real rediscoveries, whereas being too loose only costs time.
_CONSTANCY_TOLERANCE = 1e-4


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


def _is_constant(expr: Any, variables: list[str]) -> bool:
    """True when `expr` contains none of the problem's variables and is finite."""
    if expr is None:
        return False
    free: set[Any] = getattr(expr, "free_symbols", set())
    if free & {sympy.Symbol(name) for name in variables}:
        return False
    # zoo/nan/oo come out of dividing by something that simplifies to zero.
    # They are variable-free, and they are not constants in any useful sense.
    return bool(expr.is_finite is not False) and not expr.has(sympy.nan, sympy.zoo, sympy.oo)


def _symbolic_equivalence(
    candidate: str, ground_truth_formula: str, variables: list[str]
) -> tuple[bool, bool]:
    """Return (srbench_equivalent, strictly_identical).

    The first follows SRBench's definition of a symbolic solution: a candidate
    counts as rediscovery when its difference from, or its ratio to, the ground
    truth simplifies to a constant. That is deliberately more permissive than
    exact identity, and it is the right permissiveness -- a symbolic regressor
    fits its constants numerically, so a run recovering `2.0001*x` where the
    truth is `2*x` has found the law and missed a decimal. Scoring that as
    failure measures the optimiser, not the discovery.

    The second is the strict test this project used before adopting SRBench's.
    It is kept because it is strictly stronger and costs nothing once the
    expressions are parsed. Reporting both means changing the definition cannot
    quietly inflate a headline number -- the stricter figure is still there.
    """
    try:
        candidate_remapped = _remap_generic_variable_names(candidate, variables)
        candidate_expr = _safe_sympify(candidate_remapped, variables)
        truth_expr = _safe_sympify(ground_truth_formula, variables)

        difference = sympy.simplify(candidate_expr - truth_expr)
        strict = bool(difference == 0)
        if _is_constant(difference, variables):
            return True, strict

        ratio = sympy.simplify(candidate_expr / truth_expr)
        # A zero ratio means the candidate collapsed to nothing. That is not a
        # rediscovery of anything, however constant it is.
        if _is_constant(ratio, variables) and ratio != 0:
            return True, strict

        return False, strict
    # Blanket by necessity: sympify/simplify over a search-generated string
    # raise no single documented type -- SympifyError, TypeError,
    # AttributeError and RecursionError all occur. An equivalence check that
    # could not run has not shown the candidate correct, so False is the
    # conservative answer for every one of them.
    except Exception:  # noqa: BLE001
        return False, False


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

        # Numeric evidence for the two things the symbolic check will ask, used
        # to decide whether paying for it is worth it. Both are vectorised over
        # points already sampled, so they are effectively free, and a genuinely
        # equivalent pair must satisfy one of them -- SRBench's rule *is* that
        # the difference or the ratio is a constant.
        difference = candidate_vals - truth_vals
        difference_constant = bool(np.std(difference) <= _CONSTANCY_TOLERANCE * max(
            float(np.std(truth_vals)), 1.0
        ))

        with np.errstate(all="ignore"):
            ratio = candidate_vals / truth_vals
        finite_ratio = ratio[np.isfinite(ratio)]
        ratio_constant = bool(
            finite_ratio.size >= 2
            and np.std(finite_ratio)
            <= _CONSTANCY_TOLERANCE * max(abs(float(np.mean(finite_ratio))), 1.0)
        )

        return {
            "numeric_match": match,
            "max_relative_error": max_rel_error,
            "evaluated": True,
            "difference_constant": difference_constant,
            "ratio_constant": ratio_constant,
        }
    # Same reasoning as the symbolic path: evaluating two arbitrary expressions
    # over sampled ranges can fail in numpy or in sympy, and a check that could
    # not run is reported as "no match" rather than crashing a sweep.
    except Exception:  # noqa: BLE001
        # `evaluated: False` matters: it means the gate below has no evidence,
        # so the symbolic check runs anyway. Treating "could not evaluate" as
        # "not equivalent" would let a parse failure silently suppress a real
        # rediscovery.
        return {
            "numeric_match": False,
            "max_relative_error": float("inf"),
            "evaluated": False,
            "difference_constant": False,
            "ratio_constant": False,
        }


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

    # Cheap test gates the expensive one. sympy.simplify costs ~0.7s per
    # candidate and accounted for 60% of an audit's runtime -- more than the
    # symbolic regression it was auditing -- while the numeric check is
    # vectorised numpy over points already sampled.
    #
    # This cannot suppress a real rediscovery. SRBench's rule is that the
    # difference or the ratio is a constant, so any equivalent pair is constant
    # in one of them numerically too. Only candidates that are constant in
    # neither are dropped, and those cannot pass the symbolic test either. When
    # numeric evaluation fails outright there is no evidence, so the symbolic
    # check still runs.
    worth_simplifying = (
        not numeric_result["evaluated"]
        or numeric_result["difference_constant"]
        or numeric_result["ratio_constant"]
    )
    if worth_simplifying:
        symbolic_match, strict_match = _symbolic_equivalence(
            candidate_equation, ground_truth_formula, variables
        )
    else:
        symbolic_match, strict_match = False, False

    return {
        "symbolic_match": symbolic_match,
        "strict_match": strict_match,
        "numeric_match": numeric_result["numeric_match"],
        "max_relative_error": numeric_result["max_relative_error"],
    }
