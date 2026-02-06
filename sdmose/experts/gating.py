import torch.nn as nn

class GatingNetwork(nn.Module):
    """
    Soft regime gating network using PyTorch.
    """
    def __init__(self, input_dim, hidden_dim, num_regimes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_regimes),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        return self.net(x)
