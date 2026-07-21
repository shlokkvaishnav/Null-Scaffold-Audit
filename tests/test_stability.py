import numpy as np
import unittest
from equation_discovery.validation.dynamical_stability import LyapunovScreener

class MockStableHypothesis:
    # A perfectly stable identity function (Jacobian is flat baseline)
    def evaluate(self, x):
        # Shape (N, D), return (N, 1) summing features
        return np.sum(x, axis=1)
        
class MockExplosiveHypothesis:
    # An unstable polynomial function: evaluate outputs explosive derivatives
    # Specifically, f(x) = sum(x^3) -> df/dx = 3x^2
    def evaluate(self, x):
        return np.sum(x ** 3, axis=1)

class MockOscillatoryHypothesis:
    # Stable cyclic bounds. f(x) = sum(sin(x)) -> df/dx = cos(x) which is bounded [-1, 1]
    def evaluate(self, x):
        return np.sum(np.sin(x), axis=1)

class TestLyapunovStabilityScreener(unittest.TestCase):
    def setUp(self):
        # Matrix of 20 samples, 4 covariates
        self.x = np.random.uniform(0, 5, size=(20, 4))
        # Base screener with delta boundary at 2.0
        self.screener = LyapunovScreener(epsilon=1e-4, delta=2.0)
        
    def test_stable_identity_jacobian(self):
        h = MockStableHypothesis()
        # The true Jacobian of sum(x) w.r.t any x_j is just 1.0 everywhere.
        J = self.screener.compute_jacobian(h, self.x)
        self.assertEqual(J.shape, (20, 4))
        # Ensure finite difference approx captured constant gradient
        np.testing.assert_allclose(J, np.ones((20, 4)), atol=1e-2)
        
        # Stability penalty. J^TJ for 20x4 ones is 20*ones(4,4). Normalized by 20 -> ones(4,4)
        # Max eigenvalue of ones(4,4) is 4.0.
        # Penalty = max(0, 4.0 - 2.0) = 2.0
        penalty = self.screener.compute_stability_penalty(h, self.x)
        self.assertAlmostEqual(penalty, 2.0, places=1)
        
    def test_explosive_divergence_penalty(self):
        h = MockExplosiveHypothesis()
        # High feature values causing massive polynomial gradients
        x_high = np.random.uniform(10, 20, size=(10, 2))
        
        screener = LyapunovScreener(delta=50.0)
        penalty = screener.compute_stability_penalty(h, x_high)
        
        # Gradients are 3x^2, so roughly 3*(15)^2 = ~675. 
        # J^T J eigenvalues will be massive. The penalty MUST trigger heavily.
        self.assertTrue(penalty > 1000.0)
        
    def test_stable_oscillation_penalty(self):
        h = MockOscillatoryHypothesis()
        screener = LyapunovScreener(delta=5.0)
        
        penalty = screener.compute_stability_penalty(h, self.x)
        
        # Sine derivatives are bounded [-1, 1]. J^T J max eig is strictly bounded low.
        # Should be below the delta=5.0 threshold, so penalty returns 0.0 (max threshold limit).
        self.assertEqual(penalty, 0.0)

if __name__ == '__main__':
    unittest.main()
