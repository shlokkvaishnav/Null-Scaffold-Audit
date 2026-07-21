"""Shared safe-evaluation helpers for candidate equation strings.

Symbolic regression backends emit equations in different notations:
gplearn prints prefix function calls like ``mul(X0, X1)`` (uppercase,
arbitrary variable count), PySR and hand-written templates use ordinary
infix arithmetic like ``x0 * x1``. This module evaluates either form
against a features array via sympy, so callers don't need to know which
backend produced the string.
"""

from __future__ import annotations

import re
from typing import Optional

import numpy as np
import sympy

# See equation_discovery/evaluation/rediscovery.py for the rationale: sympify
# treats an unknown name like "mul" as an opaque Function unless it's given a
# real definition, so gplearn's prefix notation needs these registered.
GPLEARN_FUNCTIONS = {
    "add": lambda a, b: a + b,
    "sub": lambda a, b: a - b,
    "mul": lambda a, b: a * b,
    "div": lambda a, b: a / b,
    "sqrt": lambda a: sympy.sqrt(sympy.Abs(a)),
    "log": lambda a: sympy.log(sympy.Abs(a)),
    "abs": lambda a: sympy.Abs(a),
    "neg": lambda a: -a,
    "inv": lambda a: 1 / a,
    "sin": sympy.sin,
    "cos": sympy.cos,
    "tan": sympy.tan,
    "max": lambda a, b: sympy.Max(a, b),
    "min": lambda a, b: sympy.Min(a, b),
}


def safe_evaluate(equation: str, features: np.ndarray) -> Optional[np.ndarray]:
    """Evaluate an equation string against a (n_samples, n_features) array.

    Supports variable names ``x0``/``X0``, ``x1``/``X1``, ... (case-insensitive,
    positional) and gplearn's prefix function notation. Returns None if the
    equation can't be parsed or evaluated (caller decides the fallback).
    """
    if not equation:
        return None

    features = np.asarray(features)
    if features.ndim == 1:
        features = features.reshape(-1, 1)
    n_samples, n_features = features.shape

    # Canonicalize variable case: map both x{i} and X{i} to the same symbol.
    var_names = [f"x{i}" for i in range(n_features)]
    canonical = equation
    for i in range(n_features - 1, -1, -1):
        canonical = re.sub(rf"\b[Xx]{i}\b", f"__VAR_{i}__", canonical)
    for i in range(n_features):
        canonical = canonical.replace(f"__VAR_{i}__", var_names[i])

    symbols = sympy.symbols(var_names) if n_features > 1 else (sympy.Symbol(var_names[0]),)
    local_dict = {**GPLEARN_FUNCTIONS, **{name: sym for name, sym in zip(var_names, symbols)}}

    try:
        expr = sympy.sympify(canonical, locals=local_dict)
        fn = sympy.lambdify(symbols, expr, modules=["numpy"])
        with np.errstate(all="ignore"):
            result = np.asarray(fn(*[features[:, i] for i in range(n_features)]), dtype=float)
        if result.shape == ():
            result = np.full(n_samples, float(result))
        return result
    except Exception:
        return None
