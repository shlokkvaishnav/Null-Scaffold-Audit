import torch.nn as nn
import torch.nn.functional as F


class GatingNetwork(nn.Module):
    """
    The 'Gatekeeper' of the Mixture of Experts.
    Input:  Physics (SST, SSS, Chl) + Spatiotemporal (Lat, Lon, SinMonth, CosMonth)
    Output: Soft probability distribution pi_k over K regimes.
    """

    def __init__(self, input_dim, num_regimes, hidden_dim=128):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, num_regimes),
        )

    def forward(self, x):
        logits = self.net(x)
        # Returns (LogProbs, Probs)
        return F.log_softmax(logits, dim=1), F.softmax(logits, dim=1)
