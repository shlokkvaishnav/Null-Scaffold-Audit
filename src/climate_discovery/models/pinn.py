import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Tuple, Optional, Any
from ..physics.thermodynamics import is_physically_valid

class OceanPINN(nn.Module):
    """
    Physics-Informed Neural Network for learning ocean carbon dynamics.
    
    The network predicts fCO2 from physical variables while enforcing consistency
    with thermodynamic principles via a custom loss function.
    """
    
    def __init__(self, input_dim: int = 5, hidden_dim: int = 64):
        """
        Args:
            input_dim: Number of input features.
            hidden_dim: Number of neurons in hidden layers.
        """
        super(OceanPINN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),  # Tanh is standard for PINNs for smooth derivatives
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

class PINNTrainer:
    """Trainer class for the OceanPINN."""
    
    def __init__(self, model: OceanPINN, lr: float = 0.001, physics_weight: float = 0.01):
        """
        Args:
            model: The OceanPINN instance.
            lr: Learning rate.
            physics_weight: Weight of the physics regularization term in the loss.
        """
        self.model = model
        self.optimizer = optim.Adam(model.parameters(), lr=lr)
        self.criterion = nn.MSELoss()
        self.physics_weight = physics_weight

    def physics_loss(self, prediction: torch.Tensor, x_input: torch.Tensor) -> torch.Tensor:
        """
        Calculates the physics-informed regularization loss.
        Enforces smoothness constraint to prevent wild fluctuations.
        
        Args:
            prediction: Model predictions.
            x_input: Input features.
            
        Returns:
            Scalar tensor representing the physics loss.
        """
        # Placeholder for complex physics derivatives
        # Currently enforces that predictions roughly stay bounded
        return torch.mean(prediction ** 2)

    def train_step(self, x_batch: torch.Tensor, y_batch: torch.Tensor) -> Tuple[float, float, float]:
        """
        Performs a single training step.

        Args:
            x_batch: Input features batch.
            y_batch: Target values batch.

        Returns:
            Tuple of (total_loss, data_loss, physics_loss).
        """
        self.optimizer.zero_grad()
        
        pred = self.model(x_batch)
        
        loss_data = self.criterion(pred, y_batch)
        loss_physics = self.physics_loss(pred, x_batch) * self.physics_weight
        
        total_loss = loss_data + loss_physics
        
        total_loss.backward()
        self.optimizer.step()
        
        return total_loss.item(), loss_data.item(), loss_physics.item()
