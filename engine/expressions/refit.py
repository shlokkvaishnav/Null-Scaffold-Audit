"""Refit the numeric constants of a candidate expression, holding its structure fixed.

Symbolic regression is a two-level problem: an outer search over *structures* and an
inner fit of the *constants* inside them. Some searchers do both. Genetic programming
of the gplearn kind does only the first -- constants enter as randomly generated
terminals and are then shuffled by crossover and mutation like any other leaf, never
fitted. A structurally correct candidate can therefore score badly purely because its
constants are wrong, which is a defect of the searcher rather than of the structure it
found.

This module supplies the missing level, so the effect of its absence can be measured
rather than assumed. It is deliberately not wired into any searcher by default: what it
exists for is the comparison between a candidate field with fitted constants and the
same field without, and a default that silently refitted everything would destroy the
baseline that comparison needs.

Three choices here are conservative on purpose, because this is used to test a
hypothesis that predicts refitting *helps*. Each one makes refitting weaker, so any
measured improvement is a lower bound rather than an artifact of a generous
implementation:

* Only floating-point leaves are treated as parameters. Integers are left alone: they
  carry structure (an exponent, a rational coefficient) at least as often as they carry
  a fitted quantity, and refitting an exponent can send a negative base into the
  complex plane and break evaluation outright.
* Repeated occurrences of the same literal collapse to one shared parameter. Two
  independent leaves that happen to hold the same value become tied, which can only
  reduce the fit's freedom.
* Any failure -- an unparseable string, a non-finite residual, an optimiser that will
  not converge -- returns the original expression untouched. A refit can never make a
  candidate worse than it was.

The arithmetic is sympy's, via ``GPLEARN_FUNCTIONS``, not gplearn's internal protected
operators: ``div`` here is plain division rather than gplearn's guarded form. That
difference already exists throughout this project, because every metric the audit
reports is computed through the same sympy path. Refitting through it keeps the fit and
the measurement consistent with each other, which matters more than either matching
gplearn exactly -- a fit optimising one arithmetic while the score measures another
would be optimising the wrong objective.
"""

from __future__ import annotations

import numpy as np
import sympy
from scipy import optimize

from engine.expressions.expression_eval import GPLEARN_FUNCTIONS

__all__ = ["refit_constants"]

_MAX_PARAMETERS = 12
"""Beyond this many free constants the fit is refused rather than attempted.

A bloated candidate can carry dozens of literals, and fitting all of them turns a
structure-preserving refit into an unconstrained curve fit that says nothing about the
structure. The cap keeps the operation the one described above; candidates past it are
returned unchanged and counted as unrefitted.
"""


def _feature_index(symbol: sympy.Symbol) -> int | None:
    """Column index for a variable named ``X<i>``, or None if it is not one.

    Backends name variables positionally, so the digits are the column. Anything else
    is a symbol this module did not introduce and does not understand, which is a
    reason to refuse the refit rather than to guess a column for it.
    """
    name = symbol.name
    if len(name) < 2 or name[0] not in "Xx" or not name[1:].isdigit():
        return None
    return int(name[1:])


def refit_constants(equation: str, features: np.ndarray, targets: np.ndarray) -> str:
    """Return ``equation`` with its float literals refitted by least squares.

    The fit runs on whatever data it is given, and the caller is responsible for that
    being the *training* split. Refitting against the data a candidate is later scored
    on would manufacture exactly the advantage this is used to measure, and no check
    here can catch that -- the arrays arrive without provenance.

    The incumbent constants are the starting point, so the optimiser begins from the
    candidate as found and can only move to a lower training residual. Returns a string
    in sympy's infix form, which ``safe_evaluate`` reads as readily as the prefix form
    a genetic-programming backend emits.
    """
    if not equation:
        return equation

    features = np.asarray(features, dtype=float)
    targets = np.asarray(targets, dtype=float)

    try:
        expression = sympy.sympify(equation, locals=GPLEARN_FUNCTIONS)
    # Blanket by necessity, as everywhere else in this project that parses a
    # search-generated string: sympify raises no single documented exception type for
    # pathological input, and an expression nobody can parse is one nobody can refit.
    except Exception:  # noqa: BLE001
        return equation

    # Floats sitting in an exponent are structure, not parameters. Fitting one turns
    # x**2 into x**1.997 and sends any negative base complex, which fails evaluation
    # for reasons that have nothing to do with the candidate being wrong.
    exponents = {power.exp for power in expression.atoms(sympy.Pow)}
    literals = sorted(
        (atom for atom in expression.atoms(sympy.Float) if atom not in exponents),
        key=float,
    )
    if not literals or len(literals) > _MAX_PARAMETERS:
        return equation

    variables = sorted(
        (symbol for symbol in expression.free_symbols if isinstance(symbol, sympy.Symbol)),
        key=lambda symbol: symbol.name,
    )
    columns = [_feature_index(symbol) for symbol in variables]
    if any(column is None or column >= features.shape[1] for column in columns):
        return equation

    parameters = [sympy.Symbol(f"_c{index}") for index in range(len(literals))]
    parameterised = expression.xreplace(dict(zip(literals, parameters, strict=True)))

    try:
        model = sympy.lambdify((*variables, *parameters), parameterised, modules="numpy")
    except Exception:  # noqa: BLE001
        return equation

    inputs = [features[:, column] for column in columns]
    start = np.array([float(literal) for literal in literals], dtype=float)

    def residual(values: np.ndarray) -> np.ndarray:
        predicted = np.asarray(model(*inputs, *values), dtype=float)
        # A candidate that is non-finite anywhere is one the optimiser cannot descend
        # on. Returning a large finite residual rather than a NaN keeps least_squares
        # from aborting, and steers it away from the region instead.
        predicted = np.where(np.isfinite(predicted), predicted, np.float64(1e30))
        return np.broadcast_to(predicted, targets.shape) - targets

    try:
        if not np.all(np.isfinite(residual(start))):
            return equation
        fitted = optimize.least_squares(residual, start, method="lm", max_nfev=200 * len(start))
    except Exception:  # noqa: BLE001
        return equation

    if not fitted.success or not np.all(np.isfinite(fitted.x)):
        return equation

    # The optimiser starts from the incumbent, so this should always hold. It is checked
    # anyway: `lm` can return success while having wandered, and a refit that raised the
    # training error would corrupt the very comparison this module exists to support.
    if float(np.sum(residual(fitted.x) ** 2)) > float(np.sum(residual(start) ** 2)):
        return equation

    refitted = parameterised.xreplace(
        {
            parameter: sympy.Float(value)
            for parameter, value in zip(parameters, fitted.x, strict=True)
        }
    )
    return str(refitted)
