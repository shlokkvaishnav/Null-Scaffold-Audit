from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor


@dataclass
class Ensemble:
    """Simple neural+tree ensemble used as a neural-MoE style baseline."""

    config: dict[str, Any]

    def __post_init__(self) -> None:
        self.config = self.config or {}
        self.random_state = int(self.config.get("random_state", 0))
        self.models: list[Any] = []

    def fit(self, X, y):
        X = np.asarray(X)
        y = np.asarray(y)

        self.models = [
            MLPRegressor(
                hidden_layer_sizes=tuple(self.config.get("hidden_layer_sizes", [64, 32])),
                activation="relu",
                max_iter=int(self.config.get("max_iter", 500)),
                random_state=self.random_state,
            ),
            RandomForestRegressor(
                n_estimators=int(self.config.get("n_estimators", 200)),
                random_state=self.random_state,
            ),
        ]

        for model in self.models:
            model.fit(X, y)
        return self

    def predict(self, X):
        if not self.models:
            raise RuntimeError("Ensemble must be fit before predict.")
        X = np.asarray(X)
        preds = np.stack([np.asarray(model.predict(X), dtype=float) for model in self.models], axis=0)
        return preds.mean(axis=0)
