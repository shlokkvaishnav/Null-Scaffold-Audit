import numpy as np
import unittest
from equation_discovery.core.belief import GeodesicConfidenceUpdater

class TestGeodesicConfidenceUpdater(unittest.TestCase):
    def test_eta_zero_stationary(self):
        # eta = 0 should completely ignore target and stay rigidly on pi(t)
        igbs = GeodesicConfidenceUpdater(num_regimes=3, eta=0.0)
        igbs.pi = np.array([0.7, 0.2, 0.1])

        pi_target = np.array([0.1, 0.8, 0.1])
        new_pi = igbs.update(pi_target)

        np.testing.assert_allclose(new_pi, [0.7, 0.2, 0.1], atol=1e-5)

    def test_eta_one_memoryless(self):
        # eta = 1 should instantly jump entirely to the idealized pi_target
        igbs = GeodesicConfidenceUpdater(num_regimes=3, eta=1.0)
        igbs.pi = np.array([0.7, 0.2, 0.1])

        pi_target = np.array([0.1, 0.8, 0.1])
        new_pi = igbs.update(pi_target)

        np.testing.assert_allclose(new_pi, [0.1, 0.8, 0.1], atol=1e-5)

    def test_eta_half_geometric_mean_interpolation(self):
        # eta = 0.5 computes a mathematically perfect renormalized geometric mean point
        # on the categorical probability simplex space
        # (sqrt(pi) * sqrt(pi_target)) / Z
        igbs = GeodesicConfidenceUpdater(num_regimes=2, eta=0.5)
        igbs.pi = np.array([0.8, 0.2])
        pi_target = np.array([0.2, 0.8])

        new_pi = igbs.update(pi_target)

        # Analytic Unnormalized output:
        # 0: sqrt(0.8)*sqrt(0.2) = sqrt(0.16) = 0.4
        # 1: sqrt(0.2)*sqrt(0.8) = sqrt(0.16) = 0.4
        # Normalized mapping must be: [0.5, 0.5]
        np.testing.assert_allclose(new_pi, [0.5, 0.5], atol=1e-5)

    def test_sum_to_one_simplex_bounds_preservation(self):
        # Formal check that fraction exponents securely remain bounded on the valid
        # strict probability simplex surface summing to exactly 1.0
        igbs = GeodesicConfidenceUpdater(num_regimes=4, eta=0.33)
        igbs.pi = np.array([0.6, 0.2, 0.15, 0.05])
        pi_target = np.array([0.1, 0.1, 0.6, 0.2])

        new_pi = igbs.update(pi_target)

        self.assertAlmostEqual(np.sum(new_pi), 1.0, places=6)

if __name__ == '__main__':
    unittest.main()
