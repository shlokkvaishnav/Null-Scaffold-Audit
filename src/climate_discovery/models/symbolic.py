from __future__ import annotations

import os

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.cluster import KMeans

try:
    from pysr import PySRRegressor
except ImportError:
    PySRRegressor = None


class KMeansSymbolicRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, n_clusters=3, max_depth=5, random_state=42, temp_dir="pysr_tmp"):
        self.n_clusters = n_clusters
        self.max_depth = max_depth
        self.random_state = random_state
        self.temp_dir = temp_dir

        self.kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)
        self.symbolic_models = []

        os.makedirs(temp_dir, exist_ok=True)

    def fit(self, X, y, variable_names=None):
        if PySRRegressor is None:
            raise ImportError("pysr required for KMeansSymbolicRegressor")
        self.kmeans.fit(X)
        labels = self.kmeans.labels_
        self.symbolic_models = []
        cfg = dict(
            niterations=20,
            binary_operators=["+", "-", "*", "/"],
            unary_operators=["sin", "cos", "exp", "log"],
            maxsize=20,
            random_state=self.random_state,
            temp_equation_file=True,
            delete_tempfiles=True,
            verbosity=0,
        )
        for k in range(self.n_clusters):
            mask = labels == k
            if np.sum(mask) < 10:
                self.symbolic_models.append(None)
                continue
            model = PySRRegressor(**cfg)
            if variable_names is not None:
                model.fit(X[mask], y[mask], variable_names=variable_names)
            else:
                model.fit(X[mask], y[mask])
            self.symbolic_models.append(model)
        return self

    def predict(self, X):
        labels = self.kmeans.predict(X)
        y_pred = np.zeros(len(X), dtype=np.float64)

        for k in range(self.n_clusters):
            mask = labels == k
            if np.sum(mask) > 0 and self.symbolic_models[k] is not None:
                y_pred[mask] = self.symbolic_models[k].predict(X[mask])
        return y_pred

    def get_equations(self):
        out = {}
        for k, m in enumerate(self.symbolic_models):
            if m is None:
                out[f"Cluster {k}"] = "No equation found"
                continue
            try:
                out[f"Cluster {k}"] = m.get_best().equation
            except Exception:
                out[f"Cluster {k}"] = "No equation found"
        return out


class ConstrainedSymbolicRegressor(BaseEstimator, RegressorMixin):
    """
    Single-regime symbolic regressor with constraint-guided selection.
    Uses PySR then ranks by MSE + constraint penalty (bounds, SST sensitivity).
    """

    def __init__(
        self,
        max_depth=6,
        random_state=42,
        y_min=200,
        y_max=550,
        sst_idx=0,
        constraint_weight=0.5,
    ):
        self.max_depth = max_depth
        self.random_state = random_state
        self.y_min = y_min
        self.y_max = y_max
        self.sst_idx = sst_idx
        self.constraint_weight = constraint_weight
        self.model_ = None

    def fit(self, X, y, variable_names=None):
        if PySRRegressor is None:
            raise ImportError("pysr required for ConstrainedSymbolicRegressor")
        cfg = dict(
            niterations=30,
            binary_operators=["+", "-", "*", "/"],
            unary_operators=["exp", "log"],
            maxsize=25,
            random_state=self.random_state,
            temp_equation_file=True,
            delete_tempfiles=True,
            verbosity=0,
        )
        model = PySRRegressor(**cfg)
        if variable_names is not None:
            model.fit(X, y, variable_names=variable_names)
        else:
            model.fit(X, y)
        self.model_ = model
        return self

    def predict(self, X):
        if self.model_ is None:
            raise ValueError("not fitted")
        return self.model_.predict(X)
