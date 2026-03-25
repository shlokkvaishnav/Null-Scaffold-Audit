import numpy as np
import unittest
from sdmose.science.constraints import PhysicsConstraints
from sdmose.agent.belief import FactorGraphBeliefState

# Mock hypothesis class targeting string equations and regime ids
class MockHypothesis:
    def __init__(self, eq, regime_id):
        self.equation = eq
        self.regime_id = regime_id

class TestFactorGraphPropagation(unittest.TestCase):
    def setUp(self):
        self.physics = PhysicsConstraints()
        
    def test_clique_potentials(self):
        # 1. Conservation (violates on raw exponential)
        self.assertEqual(self.physics.psi_conservation("exp(x0) + 1"), 0.1)
        self.assertEqual(self.physics.psi_conservation("x0 + x1"), 1.0)
        
        # 2. Thermodynamics (violates on inverse gradient)
        self.assertEqual(self.physics.psi_thermo("-(x1 - x0)"), 0.01)
        self.assertEqual(self.physics.psi_thermo("(x1 - x0)"), 1.0)
        
        # 3. Stability (rewards bounds)
        self.assertEqual(self.physics.psi_stability("sin(x0)"), 1.0)
        self.assertEqual(self.physics.psi_stability("x0 * x1"), 0.5)

    def test_gamma_zero_memoryless(self):
        # Test gamma = 0 updates purely on current factors, ignoring history
        fg = FactorGraphBeliefState(num_regimes=2, gamma=0.0)
        # Set artificial history bias
        fg.pi = np.array([0.9, 0.1]) 
        
        # Both hypotheses equal potential (no violation) -> psi_total = 1.0 * 1.0 * 0.5 = 0.5 for both
        h1 = MockHypothesis("x0", 0)
        h2 = MockHypothesis("x1", 1)
        
        new_pi = fg.update([h1, h2], self.physics)
        
        # Since gamma=0, prior [0.9, 0.1]^0 = [1., 1.], pi should redistribute uniformly to 0.5 and 0.5
        np.testing.assert_allclose(new_pi, [0.5, 0.5])
        
    def test_gamma_one_infinite_memory(self):
        fg = FactorGraphBeliefState(num_regimes=2, gamma=1.0)
        fg.pi = np.array([0.8, 0.2])
        
        h1 = MockHypothesis("x0", 0)
        h2 = MockHypothesis("x1", 1)
        
        new_pi = fg.update([h1, h2], self.physics)
        
        # Since gamma=1, the priors perfectly modulate the equal clique potentials.
        # Should retain original distribution [0.8, 0.2] precisely.
        np.testing.assert_allclose(new_pi, [0.8, 0.2])

    def test_hard_constraint_zero_out(self):
        # Test that a hard physics constraint failure zeroes out the regime belief
        fg = FactorGraphBeliefState(num_regimes=2, gamma=0.5)
        # Regime 0 has very strong prior belief
        fg.pi = np.array([0.9, 0.1])
        
        # Regime 0 proposes a thermodynamically impossible hypothesis
        h0 = MockHypothesis("-(x1 - x0)", 0)
        # Regime 1 proposes a highly stable standard hypothesis
        h1 = MockHypothesis("sin(x0)", 1)
        
        new_pi = fg.update([h0, h1], self.physics)
        
        # new_pi[0] should be extremely small because psi_thermo(h0) = 0.01
        self.assertTrue(new_pi[0] < 0.05)
        # new_pi[1] should capture almost all the mass
        self.assertTrue(new_pi[1] > 0.95)

if __name__ == '__main__':
    unittest.main()
