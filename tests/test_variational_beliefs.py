import pytest

torch = pytest.importorskip("torch", reason="torch is an optional extra ([torch]), not installed in the default image")
import unittest
import numpy as np
from physics_discovery.inference.variational import JointELBO
from physics_discovery.core.belief import EquationConfidenceTracker

# Mock Hypothesis class for testing EquationBeliefState
class MockHypothesis:
    def __init__(self, eq, reg, score, comp):
        self.equation = eq
        self.regime_id = reg
        self.score = score
        self.complexity = comp

class TestVariationalBeliefs(unittest.TestCase):
    def test_equation_belief_state(self):
        # 3 hypotheses for regime 0, 1 for regime 1
        h1 = MockHypothesis("x0", 0, 10.0, 1)
        h2 = MockHypothesis("x0+x1", 0, 12.0, 3)
        h3 = MockHypothesis("sin(x0)", 0, 8.0, 2)
        h4 = MockHypothesis("x1", 1, 15.0, 1)
        
        ebs = EquationConfidenceTracker(regime_id=0)
        q_h = ebs.update([h1, h2, h3, h4], temperature=1.0)
        
        # h4 should be ignored (wrong regime)
        self.assertNotIn("x1", q_h)
        self.assertIn("x0", q_h)
        self.assertIn("x0+x1", q_h)
        self.assertIn("sin(x0)", q_h)
        
        # Probabilities should sum to 1 over the tree distribution
        total_prob = sum(q_h.values())
        self.assertAlmostEqual(total_prob, 1.0, places=5)
        
        # h2 has highest score (12.0), should map to highest probability mass
        self.assertTrue(q_h["x0+x1"] > q_h["x0"])
        
    def test_joint_elbo_forward(self):
        num_regimes = 2
        batch_size = 4

        elbo_module = JointELBO(num_regimes=num_regimes, transition_prior=0.9)

        # Mock inputs
        # log_likelihoods: (T, Num_Regimes), already contains the inner
        # hypothesis-expectation term per the JointELBO.forward docstring.
        log_likelihoods = torch.randn(batch_size, num_regimes, requires_grad=True)

        # gate_probs: soft regime assignments from the gate, over the batch
        # (treated as a temporal sequence of length batch_size)
        gate_logits = torch.randn(batch_size, num_regimes, requires_grad=True)
        gate_probs = torch.softmax(gate_logits, dim=-1)

        outputs = elbo_module(log_likelihoods=log_likelihoods, gate_probs=gate_probs)
        loss = outputs["loss"]

        # Ensure it correctly generated the scalar ELBO and graph connects
        self.assertTrue(loss.requires_grad)
        self.assertIsNotNone(loss.item())
        self.assertIn("elbo", outputs)
        self.assertIn("kl_z", outputs)

        # Backprop test
        loss.backward()
        self.assertIsNotNone(log_likelihoods.grad)
        self.assertIsNotNone(gate_logits.grad)
        
if __name__ == '__main__':
    unittest.main()
