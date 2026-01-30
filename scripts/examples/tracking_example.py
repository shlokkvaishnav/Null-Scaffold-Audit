"""Example: SD-MoSE training with experiment tracking.

Demonstrates integration of W&B/MLflow for:
- Hyperparameter logging
- Loss/metric tracking
- Equation versioning
- Model checkpointing

Usage:
    # Weights & Biases
    python -m scripts.examples.tracking_example --backend wandb
    
    # MLflow
    python -m scripts.examples.tracking_example --backend mlflow
    
    # Both (parallel logging)
    python -m scripts.examples.tracking_example --backend both
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.optim as optim

# Add src to path
root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(root / "src"))

from climate_discovery.config import ModelConfig, FEATURES_EXPERT, FEATURES_GATING, TRAIN_NC, VAL_NC
from climate_discovery.data.datasets import ClimateDataset
from climate_discovery.models.gating import GatingNetwork
from climate_discovery.models.losses import SDMoSELoss
from climate_discovery.utils.tracking import init_tracker


def train_with_tracking(config: ModelConfig, backend: str = "wandb"):
    """Train SD-MoSE with experiment tracking."""
    
    print("=" * 70)
    print("SD-MoSE TRAINING WITH EXPERIMENT TRACKING")
    print("=" * 70)
    
    # =========================================================================
    # Initialize Tracker
    # =========================================================================
    tracker = init_tracker(
        config=config,
        backend=backend,
        project=config.tracking_project,
        name="sdmose-tracking-demo",
        tags=["demo", "tracking", f"regimes_{config.num_regimes}"],
        notes="Demonstration of experiment tracking integration",
    )
    
    if tracker is None:
        print("⚠️  Tracking disabled, continuing without logging")
    
    # =========================================================================
    # Load Data
    # =========================================================================
    print("\n📦 Loading data...")
    train_dataset = ClimateDataset(
        TRAIN_NC,
        expert_features=FEATURES_EXPERT,
        gating_features=FEATURES_GATING,
        target="fco2",
    )
    
    val_dataset = ClimateDataset(
        VAL_NC,
        expert_features=FEATURES_EXPERT,
        gating_features=FEATURES_GATING,
        target="fco2",
    )
    
    print(f"✓ Train: {len(train_dataset)} samples")
    print(f"✓ Val: {len(val_dataset)} samples")
    
    # =========================================================================
    # Create Model
    # =========================================================================
    print(f"\n🧠 Creating gating network ({config.num_regimes} regimes)...")
    gating = GatingNetwork(
        input_dim=len(FEATURES_GATING),
        num_regimes=config.num_regimes,
        hidden_dims=[64, 32],
        dropout=config.dropout,
    ).to(config.device)
    
    # =========================================================================
    # Training Setup
    # =========================================================================
    optimizer = optim.Adam(gating.parameters(), lr=1e-3)
    criterion = SDMoSELoss(
        entropy_weight=config.entropy_weight,
        spatial_weight=config.spatial_smoothness_weight,
        temporal_weight=config.temporal_smoothness_weight,
    )
    
    # Dummy expert predictions (for demo)
    expert_preds = torch.randn(len(train_dataset), config.num_regimes)
    
    # =========================================================================
    # Training Loop
    # =========================================================================
    print(f"\n🚀 Training for {10} epochs...")
    
    for epoch in range(10):
        gating.train()
        
        # Simulated mini-batch training
        X_gate = torch.from_numpy(train_dataset.X_gate).float().to(config.device)
        y = torch.from_numpy(train_dataset.y).float().to(config.device)
        
        # Forward
        regime_probs = gating(X_gate)
        y_pred = torch.sum(regime_probs * expert_preds.to(config.device), dim=1)
        
        # Loss
        spatial_coords = X_gate[:, :2]
        year_norm = X_gate[:, 8]
        time_indices = (year_norm * 10).long()
        
        loss_dict = criterion(y_pred, y, regime_probs, spatial_coords, time_indices)
        loss = loss_dict["total"]
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # =====================================================================
        # TRACKING: Log metrics
        # =====================================================================
        if tracker and epoch % 1 == 0:
            metrics = {
                "train/loss": loss.item(),
                "train/prediction_loss": loss_dict["prediction"].item(),
                "train/entropy": loss_dict["entropy"].item(),
                "train/spatial_loss": loss_dict["spatial"].item(),
                "epoch": epoch,
            }
            tracker.log_metrics(metrics, step=epoch)
        
        # =====================================================================
        # TRACKING: Log regime statistics
        # =====================================================================
        if tracker and epoch % 2 == 0:
            tracker.log_regime_statistics(
                regime_probs.detach().cpu().numpy(),
                step=epoch,
            )
        
        # Validation (every 5 epochs)
        if epoch % 5 == 0:
            gating.eval()
            with torch.no_grad():
                X_val = torch.from_numpy(val_dataset.X_gate).float().to(config.device)
                y_val = torch.from_numpy(val_dataset.y).float().to(config.device)
                
                val_probs = gating(X_val)
                val_pred = torch.sum(val_probs * expert_preds[:len(val_dataset)].to(config.device), dim=1)
                
                val_mse = torch.mean((val_pred - y_val) ** 2).item()
                val_r2 = 1 - val_mse / torch.var(y_val).item()
            
            # =================================================================
            # TRACKING: Log validation metrics
            # =================================================================
            if tracker:
                tracker.log_metrics({
                    "val/mse": val_mse,
                    "val/r2": val_r2,
                }, step=epoch)
            
            print(f"Epoch {epoch:2d} | Loss: {loss.item():.4f} | Val R²: {val_r2:.4f}")
    
    # =========================================================================
    # TRACKING: Log equations (simulated)
    # =========================================================================
    if tracker:
        print("\n📝 Logging discovered equations...")
        equations = {
            0: "fCO2 = 349.56 - 2.34 * exp(0.031 * SST)",
            1: "fCO2 = 380.2 + 3.14 * SST - 1.57 * SSS",
            2: "fCO2 = 412.3 * (1 + 0.045 * log(Chl))",
            3: "fCO2 = 295.1 + 1.2 * SST + 0.8 * |∇SST|",
            4: "fCO2 = 445.7 - 3.5 * SSS + 0.02 * SST^2",
            5: "fCO2 = 320.4 + 2.1 * SST * log(Chl + 1)",
        }
        tracker.log_equations(equations, iteration=5)
    
    # =========================================================================
    # TRACKING: Save and log checkpoint
    # =========================================================================
    checkpoint_path = Path("checkpoints/tracking_demo.pth")
    checkpoint_path.parent.mkdir(exist_ok=True)
    
    torch.save({
        "epoch": 10,
        "gating_state_dict": gating.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "config": config,
    }, checkpoint_path)
    
    if tracker:
        print(f"\n💾 Logging checkpoint: {checkpoint_path}")
        tracker.log_model(checkpoint_path, model_name="sd-mose-demo")
    
    # =========================================================================
    # TRACKING: Finish
    # =========================================================================
    if tracker:
        print("\n✓ Finishing experiment tracking...")
        tracker.finish()
    
    print("\n" + "=" * 70)
    print("TRAINING COMPLETE!")
    print("=" * 70)
    
    # Print backend-specific URLs
    if backend == "wandb" and tracker:
        print(f"\n🔗 View run at: {tracker.wandb_run.url}")
    elif backend == "mlflow" and tracker:
        print("\n🔗 View runs: mlflow ui --port 5000")
        print("   Then navigate to: http://localhost:5000")


def main():
    parser = argparse.ArgumentParser(description="SD-MoSE training with tracking")
    parser.add_argument(
        "--backend",
        type=str,
        default="wandb",
        choices=["wandb", "mlflow", "both", "none"],
        help="Tracking backend",
    )
    
    args = parser.parse_args()
    
    # Create config
    config = ModelConfig()
    config.tracking_backend = args.backend
    
    # Train
    train_with_tracking(config, backend=args.backend)


if __name__ == "__main__":
    main()
