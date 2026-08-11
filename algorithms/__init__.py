"""Search algorithms, behind the engine's `AlgorithmPlugin` contract.

These sit outside `engine/` on purpose. Article 6 says the engine never depends
on a specific algorithm, library or backend, and every module here imports one:
gplearn, scikit-learn, and optionally xgboost or lightgbm. Putting them in the
core would make the core depend on them by definition, which is why
`tools/check_domain_independence.py` lists this package as forbidden to import
from `engine/`.

Nothing here knows what its inputs mean. A symbolic regressor fits `y` from `X`
whether the columns are voltages or rainfall, which is why these are shared by
every domain plugin rather than owned by one.
"""

from __future__ import annotations

# Relative, not absolute. This package is one of the roots the engine is
# forbidden to import, and the domain-independence checker reads an absolute
# self-import as exactly that violation. A relative import cannot reach a
# sibling top-level package, so it states what is actually meant here.
from .baselines import BaselineModel
from .ensemble import Ensemble
from .symbolic import SymbolicHypothesisGenerator

__all__ = ["BaselineModel", "Ensemble", "SymbolicHypothesisGenerator"]
