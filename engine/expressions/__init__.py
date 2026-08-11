"""Symbolic expressions and their safe evaluation.

A candidate is a string until something evaluates it. This package owns that
string: how it is parsed, how it is evaluated over a feature matrix, and what
happens when it cannot be. None of that depends on what the variables mean, so
none of it belongs to a domain.
"""

from __future__ import annotations

from engine.expressions.expression_eval import safe_evaluate
from engine.expressions.hypothesis import Hypothesis

__all__ = ["Hypothesis", "safe_evaluate"]
