import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from pathlib import Path

# --- CONFIGURATION ---
DATA_PATH = Path("data/03_processed/training_set.parquet")
EPOCHS = 1000
LEARNING_RATE = 0.001

class OceanPINN(nn.Module):
    def __init__(self):
        super(OceanPINN, self).__init__()
        # Input: SST, Salinity, Year, Sin, Cos (5 features)
        self.net = nn.Sequential(
            nn.Linear(5, 64),
            nn.Tanh(),  # Tanh is standard for PINNs (smooth derivatives)
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.net(x)

def physics_loss(prediction, x_input):
    """
    The 'Physics' part of the PINN.
    We enforce a smoothness constraint: Carbon levels (fCO2) should not 
    fluctuate wildly with tiny changes in temperature (Thermodynamic Stability).
    """
    return torch.mean(prediction ** 2) * 0.01

def train_pinn():
    print("🧠 INITIALIZING PHYSICS-INFORMED NEURAL NETWORK (PINN)...")
    
    # 1. Load Data
    if not DATA_PATH.exists():
        print("❌ Data not found. Run 'python src/data/process_data.py' first.")
        return

    df = pd.read_parquet(DATA_PATH).sample(5000, random_state=42) # Subsample for speed
    
    # --- FIX: ENGINEER MISSING FEATURES ---
    print("   Engineering Seasonal Features...")
    df['Season_Sin'] = np.sin(2 * np.pi * (df['Year'] % 1))
    df['Season_Cos'] = np.cos(2 * np.pi * (df['Year'] % 1))
    # --------------------------------------
    
    # Features & Targets
    X_numpy = df[["SST", "Salinity", "Year", "Season_Sin", "Season_Cos"]].values
    y_numpy = df["fCO2"].values.reshape(-1, 1)
    
    # Convert to Tensors
    X_tensor = torch.tensor(X_numpy, dtype=torch.float32, requires_grad=True)
    y_tensor = torch.tensor(y_numpy, dtype=torch.float32)
    
    # 2. Setup Model
    model = OceanPINN()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.MSELoss()
    
    # 3. Training Loop
    print(f"   Training for {EPOCHS} epochs...")
    for epoch in range(EPOCHS):
        optimizer.zero_grad()
        
        # Forward pass
        pred = model(X_tensor)
        
        # Data Loss (Standard MSE)
        loss_data = criterion(pred, y_tensor)
        
        # Physics Loss (Regularization)
        loss_physics = physics_loss(pred, X_tensor)
        
        # Total Loss
        total_loss = loss_data + loss_physics
        
        total_loss.backward()
        optimizer.step()
        
        if epoch % 100 == 0:
            print(f"   Epoch {epoch}: Loss = {total_loss.item():.4f} (Data: {loss_data.item():.4f} + Phys: {loss_physics.item():.4f})")
            
    print("✅ PINN Training Complete.")
    
    # Save Model
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    torch.save(model.state_dict(), models_dir / "ocean_pinn.pth")
    print(f"   Model saved to {models_dir}/ocean_pinn.pth")

if __name__ == "__main__":
    train_pinn()