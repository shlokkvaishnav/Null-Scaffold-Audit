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

import numpy as np
import sympy

# See engine/evaluation/equivalence.py for the rationale: sympify
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


_TOKEN_RE = re.compile(r"[A-Za-z_]\w*|\d+\.?\d*|[+\-*/^]")


def node_count(equation: str) -> float:
    """Size of an expression as its number of expression-tree nodes.

    String length was the unit in use, and it is the wrong one: it makes
    ``x0*x1`` and ``x0 * x1`` different sizes, and it scales with how a backend
    chose to print a coefficient rather than with the complexity of the law.
    Node counts are what the symbolic-regression literature reports, so this is
    also what makes a complexity figure here comparable to a published one.

    Falls back to a token count when the expression cannot be parsed. The units
    then differ -- tokens are not nodes -- which is accepted because it affects
    only candidates no parser could read, and those are already scored as
    maximally unfit by every other metric. Returning NaN would be more honest
    about the unit and would abort the sweep, since the statistics layer rejects
    non-finite observations outright.

    Three cheaper or cleaner-looking units were measured against this one on the
    320 expressions of a real sweep, and none is an improvement:

    * ``sympy.simplify`` first -- unaffordable. It did not finish 40 of the
      bloated expressions in ten minutes, against 84 ms each here, and a search
      routinely emits trees of 100+ nodes.
    * ``sympy.cancel`` first -- actively worse. Putting a bloated tree over a
      common denominator expanded it about tenfold, mean 41 nodes to 483.
    * ``count_ops`` -- 9x faster and roughly half the absolute spread, but the
      spread *relative to the mean* is the same or slightly worse (2.35 -> 2.41,
      1.82 -> 2.06, 1.12 -> 1.13 across three problems).

    That last one is the useful result: the run-to-run variation in this metric
    is proportional to how large an expression the search happened to return, so
    no choice of unit reduces it. A margin fixed in absolute nodes is therefore
    the part that does not fit the measurement -- which is a finding about the
    pre-registration, not something to be quietly repaired here.

    Known perverse incentive in that fallback: a string with no expression
    content at all -- ``"(((("`` -- yields zero tokens and therefore scores as
    maximally *simple*, since lower complexity is better. It is left rather than
    patched with a sentinel, because an invented number would enter a
    pre-registered margin and quietly change what the audit certifies. The case
    is bounded in practice: such a candidate cannot be evaluated, so it takes
    the worst available rmse and scores zero on recovery.
    """
    if not equation:
        return float("nan")
    try:
        expression = sympy.sympify(equation, locals=GPLEARN_FUNCTIONS)
        return float(sum(1 for _ in sympy.preorder_traversal(expression)))
    # Blanket by necessity, for the same reason as everywhere else that parses a
    # search-generated string: sympify raises no single documented type.
    except Exception:  # noqa: BLE001
        return float(len(_TOKEN_RE.findall(equation)))


def safe_evaluate(equation: str, features: np.ndarray) -> np.ndarray | None:
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
    # Blanket by necessity: this evaluates an arbitrary search-generated
    # expression through sympify and lambdify, which between them raise
    # SympifyError, TypeError, AttributeError, NameError, ZeroDivisionError,
    # OverflowError and RecursionError depending on the input. Enumerating them
    # would silently miss cases, and "could not be evaluated" is the same
    # answer for every one of them.
    except Exception:  # noqa: BLE001
        return None
