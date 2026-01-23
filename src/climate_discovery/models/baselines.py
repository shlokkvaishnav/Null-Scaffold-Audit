import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor


class LinearBaseline(BaseEstimator, RegressorMixin):
    def __init__(self):
        self.model = LinearRegression()

    def fit(self, X, y):
        self.model.fit(X, y)
        return self

    def predict(self, X):
        return self.model.predict(X)


class RFBaseline(BaseEstimator, RegressorMixin):
    def __init__(self, n_estimators=100, max_depth=None, n_jobs=-1, random_state=42):
        self.model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            n_jobs=n_jobs,
            random_state=random_state,
        )

    def fit(self, X, y):
        self.model.fit(X, y)
        return self

    def predict(self, X):
        return self.model.predict(X)


class XGBBaseline(BaseEstimator, RegressorMixin):
    def __init__(
        self,
        n_estimators=100,
        max_depth=3,
        learning_rate=0.1,
        n_jobs=-1,
        random_state=42,
    ):
        self.model = XGBRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            n_jobs=n_jobs,
            random_state=random_state,
        )

    def fit(self, X, y):
        self.model.fit(X, y)
        return self

    def predict(self, X):
        return self.model.predict(X)


class LatitudeBandLinearRegression(BaseEstimator, RegressorMixin):
    """
    Fits separate Linear Regressions for different latitude bands.
    Bands: Tropical (-20 to 20), Mid-Lat (20-50, -20 to -50), High-Lat (>50, <-50)
    """

    def __init__(self):
        self.models = {}
        # Define bands as (min_lat, max_lat, name)
        self.bands = [
            (-90, -50, "Southern High"),
            (-50, -20, "Southern Mid"),
            (-20, 20, "Tropics"),
            (20, 50, "Northern Mid"),
            (50, 90, "Northern High"),
        ]

    def _get_band_mask(self, lat, band):
        return (lat >= band[0]) & (lat < band[1])

    def fit(self, X, y, lat):
        """
        X: features (N, D)
        y: target (N,)
        lat: latitude array (N,) corresponding to each sample
        """
        for min_l, max_l, name in self.bands:
            mask = self._get_band_mask(lat, (min_l, max_l))
            if np.sum(mask) > 100:  # Only fit if enough samples
                model = LinearRegression()
                model.fit(X[mask], y[mask])
                self.models[name] = model
        return self

    def predict(self, X, lat):
        y_pred = np.zeros(len(X))
        for min_l, max_l, name in self.bands:
            if name in self.models:
                mask = self._get_band_mask(lat, (min_l, max_l))
                if np.sum(mask) > 0:
                    y_pred[mask] = self.models[name].predict(X[mask])
        return y_pred


class KMeansSymbolicRegressor(BaseEstimator, RegressorMixin):
    """
    Combined K-means clustering + Symbolic Regression (PySR) within each cluster.
    """
    
    def __init__(self, n_clusters=3, max_depth=5, random_state=42, **pysr_kwargs):
        self.n_clusters = n_clusters
        self.max_depth = max_depth
        self.random_state = random_state
        self.pysr_kwargs = pysr_kwargs
        
        self.kmeans = None
        self.models = {}
        self.equations = {}

    def fit(self, X, y, variable_names=None):
        from sklearn.cluster import KMeans
        from pysr import PySRRegressor
        
        # 1. Cluster the data
        self.kmeans = KMeans(
            n_clusters=self.n_clusters, 
            random_state=self.random_state
        )
        labels = self.kmeans.fit_predict(X)
        
        # 2. Fit symbolic regressor for each cluster
        for k in range(self.n_clusters):
            mask = labels == k
            if np.sum(mask) < 10:
                continue
                
            print(f"Fitting PySR for cluster {k} ({np.sum(mask)} samples)...")
            
            model = PySRRegressor(
                niterations=20,  # fast setting for baseline
                binary_operators=["+", "-", "*", "/"],
                unary_operators=["sin", "cos", "exp", "log"],
                maxsize=self.max_depth * 3,
                maxdepth=self.max_depth,
                model_selection="best",
                temp_equation_file=True,
                delete_tempfiles=True,
                verbosity=0,
                **self.pysr_kwargs
            )
            
            model.fit(X[mask], y[mask], variable_names=variable_names)
            self.models[k] = model
            
            # Store best equation
            try:
                self.equations[k] = model.sympy()
            except:
                self.equations[k] = "Error retrieving equation"
                
        return self

    def predict(self, X):
        labels = self.kmeans.predict(X)
        y_pred = np.zeros(len(X))
        
        for k in range(self.n_clusters):
            mask = labels == k
            if k in self.models:
                # PySR predict returns shape (N, 1) or (N,)
                pred = self.models[k].predict(X[mask])
                if pred.ndim > 1:
                    pred = pred.flatten()
                y_pred[mask] = pred
            else:
                # Fallback if cluster empty/failed
                y_pred[mask] = np.mean(y_pred[y_pred != 0]) 
        
        return y_pred

    def get_equations(self):
        return self.equations
