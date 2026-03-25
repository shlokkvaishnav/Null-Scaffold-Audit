import torch
import tests.test_hybrid_loop as t
from sdmose.learning.trainer import SDMoSETrainer

gate = t.MockSymbolicGate()
elbo = t.MockJointELBO()
agent = t.MockAgent()

initial_weights = gate.linear.weight.clone()
trainer = SDMoSETrainer(agent=agent, gate_module=gate, elbo_module=elbo, lr=0.1)

x = torch.randn(16, 4)
y = torch.randn(16, 1)

metrics = trainer.train_step(x, y, t=0)
print("Metrics:", metrics)
print("Gradients:", gate.linear.weight.grad)
print("Weights changed:", not torch.allclose(initial_weights, gate.linear.weight))
