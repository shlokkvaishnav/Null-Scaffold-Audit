"""Stable interfaces that algorithm and domain plugins must implement.

These are the seams the orchestrator drives. A plugin implementation lives
outside this package and imports these types -- this module must never import
back from a concrete plugin, and must not name one.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

import numpy as np


@dataclass
class Dataset:
    """A loaded, ready-to-split dataset handed from a DomainPlugin to the orchestrator."""

    X: np.ndarray
    y: np.ndarray
    feature_names: list[str]
    metadata: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class AlgorithmPlugin(Protocol):
    """A discovery/search algorithm: fits candidate model(s) to (X, y).

    Implementations are not required to produce a symbolic equation (e.g. a
    plain regression baseline) -- `equation` returning None is valid and
    downstream stages (constraint checking, symbolic-complexity scoring)
    must treat that as "not applicable", not an error.
    """

    name: str

    def fit(self, X: np.ndarray, y: np.ndarray) -> AlgorithmPlugin: ...

    def predict(self, X: np.ndarray) -> np.ndarray: ...

    @property
    def equation(self) -> str | None: ...


@runtime_checkable
class DomainPlugin(Protocol):
    """A scientific domain: supplies data and knows how to score/validate candidates in it."""

    name: str

    def load_dataset(self, **kwargs: Any) -> Dataset: ...

    def validate(self, equation: str | None) -> dict[str, Any]:
        """Return constraint-violation info for a candidate equation (empty dict if none/valid)."""
        ...

    def score(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        equation: str | None = None,
    ) -> dict[str, float]:
        """Return fit-quality metrics for a candidate's predictions on this domain."""
        ...
