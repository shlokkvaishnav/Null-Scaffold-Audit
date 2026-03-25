import torch
import unittest
import numpy as np
from sdmose.learning.variational import JointELBO
from sdmose.agent.belief import EquationBeliefState

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
        
        ebs = EquationBeliefState(regime_id=0)
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
        
        elbo_module = JointELBO(num_regimes=num_regimes, complexity_penalty=0.1)
        
        # Mock inputs
        # log_likelihoods matrix across regimes and max hypotheses
        log_likelihoods = torch.randn(batch_size, num_regimes, 3, requires_grad=True)
        
        # q_z: soft regime assignments from the gate
        q_z_logits = torch.randn(batch_size, num_regimes, requires_grad=True)
        q_z = torch.softmax(q_z_logits, dim=-1)
        
        # q_h_list: distribution over equations per regime
        # Regime 0 has 3 active hypotheses, Regime 1 has 2 active hypotheses
        q_h_0 = torch.softmax(torch.randn(3, requires_grad=True), dim=-1)
        q_h_1 = torch.softmax(torch.randn(2, requires_grad=True), dim=-1)
        q_h_list = [q_h_0, q_h_1]
        
        # complexities of these hypotheses
        comp_0 = torch.tensor([1.0, 3.0, 2.0])
        comp_1 = torch.tensor([1.0, 5.0])
        comp_list = [comp_0, comp_1]
        
        loss = elbo_module(
            log_likelihoods=log_likelihoods,
            q_z=q_z,
            q_h_list=q_h_list,
            h_complexities_list=comp_list
        )
        
        # Ensure it correctly generated the scalar ELBO and graph connects
        self.assertTrue(loss.requires_grad)
        self.assertIsNotNone(loss.item())
        
        # Backprop test
        loss.backward()
        self.assertIsNotNone(log_likelihoods.grad)
        self.assertIsNotNone(q_z_logits.grad)
        
if __name__ == '__main__':
    unittest.main()
