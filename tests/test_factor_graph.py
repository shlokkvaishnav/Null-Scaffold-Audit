import unittest

import numpy as np

from physics_discovery.core.belief import FactorGraphConfidenceUpdater
from validators.equation_validity import EquationValidator


# Mock hypothesis class targeting string equations and regime ids
class MockHypothesis:
    def __init__(self, eq, regime_id):
        self.equation = eq
        self.regime_id = regime_id


class TestFactorGraphPropagation(unittest.TestCase):
    def setUp(self):
        self.validator = EquationValidator()

    def test_loss_penalty_reflects_validity_violations(self):
        # A clean equation has no validity violations -> zero penalty.
        self.assertEqual(self.validator.loss_penalty("x0 + x1"), 0.0)

        # Explicit division by zero is flagged as a violation (rate 1.0).
        self.assertEqual(self.validator.loss_penalty("x0 / 0"), 1.0)

        # log/sqrt of a negative literal is NOT flagged: candidate equations
        # are evaluated with protected sqrt(|x|)/log(|x|) semantics (see
        # engine.expressions.expression_eval), matching gplearn's own
        # protected-function behavior, so these are not actually ill-defined.
        # (A real regression: this exact pattern once caused a genuinely
        # good Coulomb's-law candidate to be rejected outright.)
        self.assertEqual(self.validator.loss_penalty("log(-1) + x0"), 0.0)
        self.assertEqual(self.validator.loss_penalty("sqrt(-1)"), 0.0)

    def test_gamma_zero_memoryless(self):
        # Test gamma = 0 updates purely on current factors, ignoring history
        fg = FactorGraphConfidenceUpdater(num_regimes=2, gamma=0.0)
        # Set artificial history bias
        fg.pi = np.array([0.9, 0.1])

        # Both hypotheses are valid (no violations) -> identical potential of 1.0 for both
        h1 = MockHypothesis("x0", 0)
        h2 = MockHypothesis("x1", 1)

        new_pi = fg.update([h1, h2], self.validator)

        # Since gamma=0, prior [0.9, 0.1]^0 = [1., 1.], pi should redistribute uniformly to 0.5 and 0.5
        np.testing.assert_allclose(new_pi, [0.5, 0.5])

    def test_gamma_one_infinite_memory(self):
        fg = FactorGraphConfidenceUpdater(num_regimes=2, gamma=1.0)
        fg.pi = np.array([0.8, 0.2])

        h1 = MockHypothesis("x0", 0)
        h2 = MockHypothesis("x1", 1)

        new_pi = fg.update([h1, h2], self.validator)

        # Since gamma=1, the priors perfectly modulate the equal clique potentials.
        # Should retain original distribution [0.8, 0.2] precisely.
        np.testing.assert_allclose(new_pi, [0.8, 0.2])

    def test_validity_violation_suppresses_regime_belief(self):
        # Test that a regime whose hypothesis fails validity checks loses belief mass
        fg = FactorGraphConfidenceUpdater(num_regimes=2, gamma=0.5)
        # Regime 0 has very strong prior belief
        fg.pi = np.array([0.9, 0.1])

        # Regime 0 proposes an equation with a validity violation
        # (division by zero -> potential = 1/(1+1) = 1/2)
        h0 = MockHypothesis("x0 / 0", 0)
        # Regime 1 proposes a fully valid equation (potential = 1/(1+0) = 1.0)
        h1 = MockHypothesis("sin(x0)", 1)

        new_pi = fg.update([h0, h1], self.validator)

        # new_pi[0] should shrink relative to its strong prior because its
        # clique potential (1/3) is much lower than regime 1's (1.0).
        self.assertTrue(new_pi[0] < 0.9)
        self.assertTrue(new_pi[1] > 0.1)
        self.assertAlmostEqual(np.sum(new_pi), 1.0, places=8)


if __name__ == "__main__":
    unittest.main()
