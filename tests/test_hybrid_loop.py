import pytest

torch = pytest.importorskip("torch", reason="torch is an optional extra ([torch]), not installed in the default image")
import torch.nn as nn
import numpy as np
import unittest

from physics_discovery.inference.trainer import DiscoveryTrainer

# Mock SD-MoSE JointELBO Module
class MockJointELBO(nn.Module):
    def forward(self, log_likelihoods, gate_probs, **kwargs):
        # Expected Log-Likelihood E_z [log p(y | x, h)]
        ell = torch.sum(gate_probs * log_likelihoods)
        # Dummy KL penalty
        regime_priors = torch.ones_like(gate_probs) / gate_probs.size(1)
        kl_z = torch.sum(gate_probs * torch.log(gate_probs / regime_priors + 1e-10))
        elbo = ell - kl_z
        # We minimize -ELBO
        return {"loss": -elbo, "elbo": elbo, "kl_z": kl_z}

# Mock Differentiable Symbolic Gate
class MockSymbolicGate(nn.Module):
    def __init__(self, num_features=4, num_regimes=2):
        super().__init__()
        self.num_regimes = num_regimes
        self.linear = nn.Linear(num_features, num_regimes)
        
    def forward(self, x):
        return torch.softmax(self.linear(x), dim=1)
        
    def sparsity_loss(self):
        return torch.tensor(0.0, requires_grad=True)

# Mock Hypothesis
class MockHypothesis:
    def __init__(self, score, regime_id):
        self.score = score
        self.regime_id = regime_id
        if regime_id == 0:
            self.mse = 0.5
            self.stability_penalty = 0.0
            self.violations_penalty = 0.0
        else:
            self.mse = 10.0
            self.stability_penalty = 5.0
            self.violations_penalty = 2.0

# Mock SD-MoSE Agent Orchestrator
class MockAgent:
    def observe(self, data):
        pass
    def retrieve(self):
        pass
    def reason(self):
        pass
    def verify(self):
        # Returns string equations with attached composite physics scores
        # e.g., representing -MSE - lambda_s * Omega_stab
        return [MockHypothesis(score=-0.5, regime_id=0), MockHypothesis(score=-10.0, regime_id=1)]
    def learn(self):
        pass

class TestHybridDiscreteContinuousOptimizer(unittest.TestCase):
    def test_variational_em_train_step(self):
        # 1. Initialize Continuous ML Models
        gate = MockSymbolicGate()
        elbo = MockJointELBO()
        agent = MockAgent()
        
        # Capture pre-step weights to verify gradient descent executes successfully 
        # across the discrete boundary
        initial_weights = gate.linear.weight.clone()
        
        trainer = DiscoveryTrainer(agent=agent, gate_module=gate, elbo_module=elbo, lr=0.1)
        
        # 2. Dummy observational data (16 samples, 4 features)
        x = torch.randn(16, 4)
        y = torch.randn(16, 1)
        
        # 3. Execute Hybrid Optimizer
        metrics = trainer.train_step(x, y, t=0)
        
        # Validate that PyTorch tracking didn't shatter across the discrete verification bounds
        self.assertTrue("loss" in metrics)
        self.assertTrue("elbo" in metrics)
        
        # Validate Continuous Parameter Update (Adam step)
        # The weights MUST have changed due to the discrete MSE/Stability scores 
        # backing up through the JointELBO log_likelihood tensor bridge into the gate routing probabilities
        weights_changed = not torch.allclose(initial_weights, gate.linear.weight)
        self.assertTrue(weights_changed)
        
        # Check entropy limits
        self.assertTrue(metrics["entropy"] > 0.0)

if __name__ == '__main__':
    unittest.main()
