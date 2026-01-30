"""Unit tests for uncertainty quantification."""

import numpy as np
import pytest

from climate_discovery.validation.uncertainty import UncertaintyEstimator


@pytest.fixture
def simple_model_factory():
    """Factory for creating simple models."""
    def create_model(X, y, seed=0):
        # Simple linear model
        from sklearn.linear_model import LinearRegression
        np.random.seed(seed)
        model = LinearRegression()
        model.fit(X, y)
        return model
    
    return create_model


def test_bootstrap_ensemble(simple_model_factory):
    """Test bootstrap ensemble creation."""
    np.random.seed(42)
    X = np.random.randn(100, 4)
    y = X @ np.array([1, 2, 3, 4]) + np.random.randn(100) * 0.1
    
    estimator = UncertaintyEstimator(strategy="bootstrap", n_models=10)
    estimator.fit_ensemble(X, y, simple_model_factory, verbose=False)
    
    assert len(estimator.models) == 10


def test_prediction_with_uncertainty(simple_model_factory):
    """Test uncertainty prediction."""
    np.random.seed(42)
    X_train = np.random.randn(100, 4)
    y_train = X_train @ np.array([1, 2, 3, 4])
    
    X_test = np.random.randn(20, 4)
    
    estimator = UncertaintyEstimator(strategy="bootstrap", n_models=10)
    estimator.fit_ensemble(X_train, y_train, simple_model_factory, verbose=False)
    
    mean, std, lower, upper = estimator.predict_with_uncertainty(X_test)
    
    # Check shapes
    assert mean.shape == (20,)
    assert std.shape == (20,)
    assert lower.shape == (20,)
    assert upper.shape == (20,)
    
    # Check that CI contains mean
    assert np.all(lower <= mean)
    assert np.all(mean <= upper)


def test_uncertainty_increases_with_variance():
    """Test that uncertainty is higher when models disagree."""
    # Create two very different model outputs
    class HighVarianceModel:
        def __init__(self, offset):
            self.offset = offset
        
        def predict(self, X):
            return X[:, 0] * 10 + self.offset
    
    estimator = UncertaintyEstimator(n_models=2)
    estimator.models = [
        HighVarianceModel(0),
        HighVarianceModel(100),  # Very different
    ]
    
    X = np.random.randn(10, 4)
    mean, std, lower, upper = estimator.predict_with_uncertainty(X)
    
    # Std should be large
    assert np.mean(std) > 10


def test_confidence_intervals():
    """Test confidence interval coverage."""
    np.random.seed(42)
    
    # True function
    true_func = lambda X: X @ np.array([1, 2, 3, 4])
    
    X_train = np.random.randn(100, 4)
    y_train = true_func(X_train) + np.random.randn(100) * 0.5
    
    X_test = np.random.randn(50, 4)
    y_test = true_func(X_test)
    
    def train_func(X, y, seed):
        from sklearn.linear_model import LinearRegression
        model = LinearRegression()
        model.fit(X, y)
        return model
    
    estimator = UncertaintyEstimator(n_models=20, confidence_level=0.95)
    estimator.fit_ensemble(X_train, y_train, train_func, verbose=False)
    
    mean, std, lower, upper = estimator.predict_with_uncertainty(X_test)
    
    # Check empirical coverage
    in_ci = (y_test >= lower) & (y_test <= upper)
    coverage = np.mean(in_ci)
    
    # Should be close to 0.95 (with some tolerance)
    assert 0.80 < coverage < 1.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
