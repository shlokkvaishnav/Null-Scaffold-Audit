from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

import numpy as np


@dataclass
class SymbolicRegressor:
    """Wrapper for symbolic regression backends."""

    config: Dict[str, Any]

    def __post_init__(self) -> None:
        self.config = self.config or {}
        self.backend = str(self.config.get("backend", "gplearn")).lower()
        self.model = self._build_model()

    def _build_model(self):
        if self.backend == "gplearn":
            from gplearn.genetic import SymbolicRegressor as GPRegressor

            return GPRegressor(
                population_size=int(self.config.get("population_size", 500)),
                generations=int(self.config.get("generations", 20)),
                stopping_criteria=float(self.config.get("stopping_criteria", 0.01)),
                p_crossover=float(self.config.get("p_crossover", 0.7)),
                p_subtree_mutation=float(self.config.get("p_subtree_mutation", 0.1)),
                p_hoist_mutation=float(self.config.get("p_hoist_mutation", 0.05)),
                p_point_mutation=float(self.config.get("p_point_mutation", 0.1)),
                max_samples=float(self.config.get("max_samples", 0.9)),
                verbose=int(self.config.get("verbose", 0)),
                random_state=int(self.config.get("random_state", 0)),
            )
        if self.backend == "pysr":
            from pysr import PySRRegressor

            return PySRRegressor(
                niterations=int(self.config.get("niterations", 20)),
                populations=int(self.config.get("populations", 10)),
                population_size=int(self.config.get("population_size", 50)),
                maxsize=int(self.config.get("maxsize", 10)),
                binary_operators=["+", "-", "*"],
                unary_operators=["exp", "log", "sin", "cos"],
                elementwise_loss="loss(x, y) = (x - y)^2",
                verbosity=0,
                progress=False,
            )
        raise ValueError(f"Unsupported symbolic backend: {self.backend}")

    def fit(self, X, y):
        X = np.asarray(X)
        y = np.asarray(y)
        self.model.fit(X, y)
        return self

    def predict(self, X):
        X = np.asarray(X)
        return np.asarray(self.model.predict(X), dtype=float)

    @property
    def equation(self) -> str:
        if self.backend == "gplearn":
            return str(getattr(self.model, "_program", "unknown"))
        eqs = getattr(self.model, "equations_", None)
        if eqs is not None and len(eqs) > 0:
            return str(eqs.iloc[-1]["equation"])
        return "unknown"
