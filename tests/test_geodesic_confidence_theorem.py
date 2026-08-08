import unittest

import numpy as np

from physics_discovery.core.belief import GeodesicConfidenceUpdater


class TestGeodesicConfidenceUpdaterProperties(unittest.TestCase):
    """Formal properties of the geodesic (KL-interpolation) confidence update:

    pi_{t+1}(k) proportional to pi_t(k)^(1-eta) * pi_target(k)^eta

    This is a weighted geometric-mean interpolation between the previous
    belief and a target distribution along the probability simplex, i.e. a
    geodesic under the Fisher information metric. These tests check that the
    interpolation stays on the simplex, reduces to the endpoints at eta=0/1,
    and moves monotonically closer (in KL) to the target as eta increases.
    """

    def test_simplex_invariance(self):
        updater = GeodesicConfidenceUpdater(num_regimes=3, eta=0.35)
        updater.pi = np.array([0.7, 0.2, 0.1])
        out = updater.update(np.array([0.2, 0.3, 0.5]))

        self.assertAlmostEqual(np.sum(out), 1.0, places=8)
        self.assertTrue(np.all(out > 0.0))

    def test_endpoint_conditions(self):
        source = np.array([0.6, 0.3, 0.1])
        target = np.array([0.1, 0.2, 0.7])

        updater0 = GeodesicConfidenceUpdater(num_regimes=3, eta=0.0)
        updater0.pi = source.copy()
        np.testing.assert_allclose(updater0.update(target), source, atol=1e-8)

        updater1 = GeodesicConfidenceUpdater(num_regimes=3, eta=1.0)
        updater1.pi = source.copy()
        np.testing.assert_allclose(updater1.update(target), target, atol=1e-8)

    def test_kl_to_target_decreases_with_eta(self):
        source = np.array([0.8, 0.15, 0.05])
        target = np.array([0.1, 0.2, 0.7])

        etas = [0.1, 0.3, 0.5, 0.8]
        kls = []
        for eta in etas:
            updater = GeodesicConfidenceUpdater(num_regimes=3, eta=eta)
            updater.pi = source.copy()
            updated = updater.update(target)
            kls.append(GeodesicConfidenceUpdater.kl_divergence(updated, target))

        # Monotone interpolation toward target (non-increasing KL to target)
        self.assertTrue(all(kls[i] >= kls[i + 1] - 1e-9 for i in range(len(kls) - 1)))


if __name__ == "__main__":
    unittest.main()
