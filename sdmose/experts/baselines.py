class BaselineModel:
    """
    Wrapper for sklearn baselines (Linear Regression, RF, XGBoost).
    """
    def __init__(self, config):
        self.config = config

    def fit(self, X, y):
        pass

    def predict(self, X):
        pass
