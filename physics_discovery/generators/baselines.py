from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression


@dataclass
class BaselineModel:
    """Wrapper for baseline regressors used in benchmark scripts."""

    config: dict[str, Any]

    def __post_init__(self) -> None:
        self.config = self.config or {}
        self.model_type = str(self.config.get("model_type", "linear_regression")).lower()
        self.random_state = int(self.config.get("random_state", 0))
        self.model = self._build_model()

    def _build_model(self):
        if self.model_type in {"linear", "linear_regression"}:
            return LinearRegression()
        if self.model_type in {"random_forest", "rf"}:
            return RandomForestRegressor(
                n_estimators=int(self.config.get("n_estimators", 200)),
                random_state=self.random_state,
            )
        if self.model_type in {"xgboost", "xgb"}:
            try:
                from xgboost import XGBRegressor
            except ImportError as exc:
                raise RuntimeError("xgboost is required for model_type='xgboost'.") from exc
            return XGBRegressor(
                n_estimators=int(self.config.get("n_estimators", 300)),
                max_depth=int(self.config.get("max_depth", 6)),
                learning_rate=float(self.config.get("learning_rate", 0.05)),
                subsample=float(self.config.get("subsample", 0.9)),
                colsample_bytree=float(self.config.get("colsample_bytree", 0.9)),
                objective="reg:squarederror",
                random_state=self.random_state,
                verbosity=0,
            )
        if self.model_type in {"lightgbm", "lgbm"}:
            try:
                from lightgbm import LGBMRegressor
            except ImportError:
                # Fallback that keeps pipeline runnable even when optional deps are missing.
                return HistGradientBoostingRegressor(
                    max_depth=int(self.config.get("max_depth", 8)),
                    learning_rate=float(self.config.get("learning_rate", 0.05)),
                    max_iter=int(self.config.get("n_estimators", 300)),
                    random_state=self.random_state,
                )
            return LGBMRegressor(
                n_estimators=int(self.config.get("n_estimators", 300)),
                max_depth=int(self.config.get("max_depth", -1)),
                learning_rate=float(self.config.get("learning_rate", 0.05)),
                random_state=self.random_state,
                verbose=-1,
            )
        raise ValueError(f"Unsupported model_type: {self.model_type}")

    def fit(self, X, y):
        X = np.asarray(X)
        y = np.asarray(y)
        self.model.fit(X, y)
        return self

    def predict(self, X):
        X = np.asarray(X)
        return np.asarray(self.model.predict(X), dtype=float)
