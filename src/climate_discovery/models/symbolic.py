from sklearn.cluster import KMeans
from pysr import PySRRegressor
from sklearn.base import BaseEstimator, RegressorMixin
import numpy as np
import os

class KMeansSymbolicRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, n_clusters=3, max_depth=5, random_state=42, temp_dir="pysr_tmp"):
        self.n_clusters = n_clusters
        self.max_depth = max_depth
        self.random_state = random_state
        self.temp_dir = temp_dir
        
        self.kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)
        self.symbolic_models = []
        
        os.makedirs(temp_dir, exist_ok=True)

    def fit(self, X, y):
        # 1. Fit Clustering
        self.kmeans.fit(X)
        labels = self.kmeans.labels_
        
        # 2. Fit Symbolic Expert per Cluster
        self.symbolic_models = []
        for k in range(self.n_clusters):
            mask = (labels == k)
            if np.sum(mask) == 0:
                self.symbolic_models.append(None)
                continue
                
            print(f"Fitting Symbolic Expert for Cluster {k} ({np.sum(mask)} samples)...")
            
            # Simple PySR configuration for speed
            model = PySRRegressor(
                niterations=20, # Low for prototyping
                binary_operators=["+", "-", "*", "/"],
                unary_operators=["sin", "cos", "exp", "log"],
                maxsize=20,
                random_state=self.random_state,
                temp_equation_file=True,
                delete_tempfiles=True,
                verbosity=0
            )
            
            model.fit(X[mask], y[mask])
            self.symbolic_models.append(model)
            
        return self

    def predict(self, X):
        labels = self.kmeans.predict(X)
        y_pred = np.zeros(len(X))
        
        for k in range(self.n_clusters):
            mask = (labels == k)
            if np.sum(mask) > 0 and self.symbolic_models[k] is not None:
                # PySR predict requires 2D array typically, but check version
                y_pred[mask] = self.symbolic_models[k].predict(X[mask])
                
        return y_pred
    
    def get_equations(self):
        equations = {}
        for k, model in enumerate(self.symbolic_models):
            if model is not None:
                # Get best equation
                try:
                    eqn = model.get_best().equation
                    equations[f"Cluster {k}"] = eqn
                except:
                    equations[f"Cluster {k}"] = "No equation found"
        return equations
