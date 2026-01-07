from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.base import BaseEstimator, RegressorMixin
import numpy as np

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
            random_state=random_state
        )
    
    def fit(self, X, y):
        self.model.fit(X, y)
        return self
    
    def predict(self, X):
        return self.model.predict(X)

class XGBBaseline(BaseEstimator, RegressorMixin):
    def __init__(self, n_estimators=100, max_depth=3, learning_rate=0.1, n_jobs=-1, random_state=42):
        self.model = XGBRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            n_jobs=n_jobs,
            random_state=random_state
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
            (50, 90, "Northern High")
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
            if np.sum(mask) > 100: # Only fit if enough samples
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
