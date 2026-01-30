"""Visualize feature importance from Attention-Based Gating Network.

This script demonstrates how to extract and visualize which features
are most important for regime assignment in different regions/seasons.

Usage:
    python -m scripts.viz.plot_feature_importance --checkpoint checkpoints/sdmose_final.pth
"""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import xarray as xr

# Add src to path
root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(root / "src"))

from climate_discovery.config import (
    FEATURES_GATING,
    ModelConfig,
    TEST_NC,
)
from climate_discovery.data.datasets import ClimateDataset
from climate_discovery.models.gating import AttentionGatingNetwork
from climate_discovery.models.mixture import SDMoSE


def visualize_feature_importance(
    model: SDMoSE,
    dataset: ClimateDataset,
    n_samples: int = 1000,
    save_path: str = "figures/feature_importance.png",
):
    """Visualize feature importance across different oceanic regions.
    
    Args:
        model: Trained SD-MoSE model with attention gating
        dataset: Test dataset
        n_samples: Number of samples to analyze
        save_path: Where to save the figure
    """
    # Check if gating is attention-based
    if not isinstance(model.gating_network, AttentionGatingNetwork):
        print("Model does not use AttentionGatingNetwork. Cannot extract feature importance.")
        return
    
    # Sample random points
    indices = np.random.choice(len(dataset), size=min(n_samples, len(dataset)), replace=False)
    X_gate = torch.from_numpy(dataset.X_gate[indices]).float()
    
    # Get feature importance
    print("Computing feature importance...")
    with torch.no_grad():
        importance = model.gating_network.get_feature_importance(X_gate)
        importance_np = importance.cpu().numpy()
        
        # Also get regime assignments
        probs = model.gating_network(X_gate)
        dominant_regime = torch.argmax(probs, dim=1).cpu().numpy()
    
    # Create figure
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle("Feature Importance for Regime Assignment", fontsize=16, fontweight='bold')
    
    # Feature names
    feature_names = [
        "Latitude", "Longitude", "SST", "SSS", "log(Chl)",
        "|∇SST|", "sin(month)", "cos(month)", "Year"
    ]
    
    # Plot 1: Average importance across all samples
    ax = axes[0, 0]
    mean_importance = np.mean(importance_np, axis=0)
    ax.barh(feature_names, mean_importance, color='steelblue')
    ax.set_xlabel("Mean Importance Score")
    ax.set_title("Overall Feature Importance")
    ax.grid(axis='x', alpha=0.3)
    
    # Plot 2: Importance heatmap by regime
    ax = axes[0, 1]
    regime_importance = np.zeros((model.num_regimes, len(feature_names)))
    for k in range(model.num_regimes):
        mask = dominant_regime == k
        if np.sum(mask) > 0:
            regime_importance[k] = np.mean(importance_np[mask], axis=0)
    
    sns.heatmap(
        regime_importance,
        xticklabels=feature_names,
        yticklabels=[f"Regime {k}" for k in range(model.num_regimes)],
        cmap="YlOrRd",
        annot=True,
        fmt=".2f",
        ax=ax,
        cbar_kws={"label": "Importance"}
    )
    ax.set_title("Feature Importance by Regime")
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
    
    # Plot 3: Top 3 features distribution
    ax = axes[0, 2]
    top_3_indices = np.argsort(mean_importance)[-3:][::-1]
    for idx in top_3_indices:
        ax.hist(importance_np[:, idx], bins=30, alpha=0.6, label=feature_names[idx])
    ax.set_xlabel("Importance Score")
    ax.set_ylabel("Frequency")
    ax.set_title("Distribution of Top 3 Features")
    ax.legend()
    ax.grid(alpha=0.3)
    
    # Plot 4-6: Feature importance vs. physical variables
    physical_vars = [
        ("SST", dataset.X_gate[:, 2]),
        ("SSS", dataset.X_gate[:, 3]),
        ("log(Chl)", dataset.X_gate[:, 4]),
    ]
    
    for i, (var_name, var_data) in enumerate(physical_vars):
        ax = axes[1, i]
        var_idx = feature_names.index(var_name)
        
        scatter = ax.scatter(
            var_data[indices],
            importance_np[:, var_idx],
            c=dominant_regime,
            cmap='tab10',
            alpha=0.6,
            s=10
        )
        ax.set_xlabel(f"{var_name} Value")
        ax.set_ylabel(f"{var_name} Importance")
        ax.set_title(f"Importance of {var_name}")
        ax.grid(alpha=0.3)
        
        if i == 2:  # Add colorbar to last plot
            cbar = plt.colorbar(scatter, ax=ax)
            cbar.set_label("Regime")
    
    plt.tight_layout()
    
    # Save figure
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved figure: {save_path}")
    
    # Print summary statistics
    print("\n" + "=" * 60)
    print("FEATURE IMPORTANCE SUMMARY")
    print("=" * 60)
    for i, name in enumerate(feature_names):
        print(f"{name:15s}: {mean_importance[i]:.4f} ± {np.std(importance_np[:, i]):.4f}")
    
    print("\n" + "=" * 60)
    print("TOP FEATURES BY REGIME")
    print("=" * 60)
    for k in range(model.num_regimes):
        mask = dominant_regime == k
        if np.sum(mask) > 0:
            regime_imp = np.mean(importance_np[mask], axis=0)
            top_idx = np.argmax(regime_imp)
            print(f"Regime {k} ({np.sum(mask):4d} samples): {feature_names[top_idx]:15s} ({regime_imp[top_idx]:.4f})")


def main():
    parser = argparse.ArgumentParser(
        description="Visualize feature importance from Attention-Gating"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/sdmose_final.pth",
        help="Path to trained model checkpoint"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="figures/feature_importance.png",
        help="Output figure path"
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        default=5000,
        help="Number of samples to analyze"
    )
    
    args = parser.parse_args()
    
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load config
    config = ModelConfig()
    config.gating_type = "attention"  # Must be attention for feature importance
    
    # Load test data
    print("Loading test dataset...")
    test_dataset = ClimateDataset(
        TEST_NC,
        expert_features=config.FEATURES_EXPERT if hasattr(config, 'FEATURES_EXPERT') else ["sst", "sss", "log_chl", "sst_gradient"],
        gating_features=FEATURES_GATING,
        target="fco2",
        drop_nan=True,
    )
    print(f"Loaded {len(test_dataset)} samples")
    
    # Create model
    print("Creating model...")
    gating = AttentionGatingNetwork(
        input_dim=len(FEATURES_GATING),
        num_regimes=config.n_regimes,
        n_heads=config.attention_n_heads,
        embed_dim=config.attention_embed_dim,
        ff_dim=config.attention_ff_dim,
        dropout=0.0,  # No dropout for inference
    ).to(device)
    
    model = SDMoSE(
        gating_network=gating,
        num_regimes=config.n_regimes,
        expert_features=FEATURES_GATING[:4],  # Just for initialization
        device=device,
    )
    
    # Load checkpoint
    checkpoint_path = Path(args.checkpoint)
    if checkpoint_path.exists():
        print(f"Loading checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        if 'gating_state_dict' in checkpoint:
            model.gating_network.load_state_dict(checkpoint['gating_state_dict'])
        else:
            model.load(checkpoint_path)
        print("✓ Checkpoint loaded")
    else:
        print(f"Warning: Checkpoint not found: {checkpoint_path}")
        print("Using randomly initialized model (for demo purposes)")
    
    # Visualize
    model.eval()
    visualize_feature_importance(
        model,
        test_dataset,
        n_samples=args.n_samples,
        save_path=args.output,
    )


if __name__ == "__main__":
    main()
