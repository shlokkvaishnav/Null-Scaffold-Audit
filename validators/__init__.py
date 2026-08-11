"""Constraint rules a candidate equation can violate.

These implement `engine.plugin.ConstraintValidator`, which is how
`engine.scoring` penalises an invalid candidate without knowing what makes one
invalid. The engine supplies the arithmetic; this package supplies the rules.

Unlike `engine/` and `algorithms/`, this package is deliberately **not** checked
for domain independence, and it would fail if it were -- `equation_validity.py`
names Coulomb's law among its examples. That is the point. "No logarithm of a
negative" is a fact about expressions, but "no negative mass" is a fact about a
field, and a rule set carrying no domain knowledge could not reject anything
worth rejecting. Keeping these outside the core is what lets the core stay
domain independent while the system as a whole still checks real constraints.
"""

from __future__ import annotations

# Relative for the same reason as algorithms/: `validators` is a root the
# engine may not import, and an absolute self-import reads to the checker as
# exactly that violation.
from .dynamical_stability import LyapunovScreener
from .equation_validity import EquationValidator

__all__ = ["EquationValidator", "LyapunovScreener"]
