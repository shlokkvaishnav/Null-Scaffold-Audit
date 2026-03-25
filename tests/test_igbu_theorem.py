import unittest
import numpy as np

from sdmose.agent.belief import InformationGeometricBeliefState


class TestIGBUTheoremProperties(unittest.TestCase):
    def test_simplex_invariance(self):
        igbu = InformationGeometricBeliefState(num_regimes=3, eta=0.35)
        igbu.pi = np.array([0.7, 0.2, 0.1])
        out = igbu.update(np.array([0.2, 0.3, 0.5]))

        self.assertAlmostEqual(np.sum(out), 1.0, places=8)
        self.assertTrue(np.all(out > 0.0))

    def test_endpoint_conditions(self):
        source = np.array([0.6, 0.3, 0.1])
        target = np.array([0.1, 0.2, 0.7])

        igbu0 = InformationGeometricBeliefState(num_regimes=3, eta=0.0)
        igbu0.pi = source.copy()
        np.testing.assert_allclose(igbu0.update(target), source, atol=1e-8)

        igbu1 = InformationGeometricBeliefState(num_regimes=3, eta=1.0)
        igbu1.pi = source.copy()
        np.testing.assert_allclose(igbu1.update(target), target, atol=1e-8)

    def test_kl_to_target_decreases_with_eta(self):
        source = np.array([0.8, 0.15, 0.05])
        target = np.array([0.1, 0.2, 0.7])

        etas = [0.1, 0.3, 0.5, 0.8]
        kls = []
        for eta in etas:
            igbu = InformationGeometricBeliefState(num_regimes=3, eta=eta)
            igbu.pi = source.copy()
            updated = igbu.update(target)
            kls.append(InformationGeometricBeliefState.kl_divergence(updated, target))

        # Monotone interpolation toward target (non-increasing KL to target)
        self.assertTrue(all(kls[i] >= kls[i + 1] - 1e-9 for i in range(len(kls) - 1)))


if __name__ == "__main__":
    unittest.main()
