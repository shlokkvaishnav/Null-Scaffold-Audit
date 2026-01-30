"""Visualize hierarchical regime structure.

Creates multi-panel visualization showing:
1. Coarse regimes (ocean basins)
2. Fine regimes (nested processes)
3. Hierarchical tree structure
4. Regime equations

Usage:
    python -m scripts.viz.plot_hierarchy --checkpoint checkpoints/hierarchical_final.pth
"""

import argparse
import sys
from pathlib import Path

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import torch

# Add src to path
root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(root / "src"))

from climate_discovery.config import FEATURES_GATING, TEST_NC
from climate_discovery.data.datasets import ClimateDataset
from climate_discovery.models.hierarchical import HierarchicalGatingNetwork, HierarchicalSDMoSE


def plot_hierarchical_regimes(
    model: HierarchicalSDMoSE,
    dataset: ClimateDataset,
    save_path: str = "figures/hierarchical_regimes.png",
):
    """Create comprehensive hierarchical regime visualization."""
    
    # Get regime assignments
    X_gate = torch.from_numpy(dataset.X_gate).float()
    
    with torch.no_grad():
        p_coarse, p_fine, p_joint = model.gating_network(X_gate)
        coarse_labels, fine_labels = model.gating_network.get_regime_labels(X_gate)
    
    # Convert to numpy
    coarse_labels = coarse_labels.cpu().numpy()
    fine_labels = fine_labels.cpu().numpy()
    p_coarse_np = p_coarse.cpu().numpy()
    p_fine_np = p_fine.cpu().numpy()
    
    # Extract coordinates
    lats = dataset.X_gate[:, 0] * 90  # Denormalize
    lons = dataset.X_gate[:, 1] * 180
    
    # Create figure
    fig = plt.figure(figsize=(18, 12))
    
    # =========================================================================
    # Panel 1: Coarse regimes (ocean basins)
    # =========================================================================
    ax1 = plt.subplot(2, 3, 1, projection=ccrs.PlateCarree())
    ax1.coastlines()
    ax1.add_feature(cfeature.LAND, facecolor='lightgray')
    ax1.gridlines(draw_labels=True, alpha=0.3)
    
    scatter1 = ax1.scatter(
        lons, lats,
        c=coarse_labels,
        cmap='Set1',
        s=5,
        alpha=0.6,
        transform=ccrs.PlateCarree()
    )
    ax1.set_title("Level 1: Coarse Regimes (Ocean Basins)", fontsize=14, fontweight='bold')
    cbar1 = plt.colorbar(scatter1, ax=ax1, orientation='horizontal', pad=0.05, shrink=0.8)
    cbar1.set_label("Coarse Regime")
    
    # =========================================================================
    # Panel 2: Fine regimes (full hierarchy)
    # =========================================================================
    ax2 = plt.subplot(2, 3, 2, projection=ccrs.PlateCarree())
    ax2.coastlines()
    ax2.add_feature(cfeature.LAND, facecolor='lightgray')
    ax2.gridlines(draw_labels=True, alpha=0.3)
    
    scatter2 = ax2.scatter(
        lons, lats,
        c=fine_labels,
        cmap='tab10',
        s=5,
        alpha=0.6,
        transform=ccrs.PlateCarree()
    )
    ax2.set_title("Level 2: Fine Regimes (Nested Processes)", fontsize=14, fontweight='bold')
    cbar2 = plt.colorbar(scatter2, ax=ax2, orientation='horizontal', pad=0.05, shrink=0.8)
    cbar2.set_label("Fine Regime (Flat Index)")
    
    # =========================================================================
    # Panel 3: Coarse entropy map
    # =========================================================================
    ax3 = plt.subplot(2, 3, 3, projection=ccrs.PlateCarree())
    ax3.coastlines()
    ax3.add_feature(cfeature.LAND, facecolor='lightgray')
    ax3.gridlines(draw_labels=True, alpha=0.3)
    
    # Compute coarse entropy per sample
    eps = 1e-10
    coarse_entropy = -np.sum(p_coarse_np * np.log(p_coarse_np + eps), axis=1)
    
    scatter3 = ax3.scatter(
        lons, lats,
        c=coarse_entropy,
        cmap='viridis',
        s=5,
        alpha=0.6,
        vmin=0,
        vmax=np.log(model.num_coarse),
        transform=ccrs.PlateCarree()
    )
    ax3.set_title("Coarse Regime Uncertainty", fontsize=14, fontweight='bold')
    cbar3 = plt.colorbar(scatter3, ax=ax3, orientation='horizontal', pad=0.05, shrink=0.8)
    cbar3.set_label("Entropy (High = Transition Zone)")
    
    # =========================================================================
    # Panel 4: Hierarchy tree
    # =========================================================================
    ax4 = plt.subplot(2, 3, 4)
    ax4.axis('off')
    
    # Get regime interpretation
    hierarchy = model.get_regime_interpretation()
    
    # Draw tree
    y_offset = 0.9
    x_coarse = 0.2
    x_fine = 0.6
    
    for k_coarse, data in hierarchy.items():
        # Coarse node
        ax4.add_patch(mpatches.FancyBboxPatch(
            (x_coarse - 0.1, y_offset - 0.05),
            0.2, 0.08,
            boxstyle="round,pad=0.01",
            facecolor='lightblue',
            edgecolor='black',
            linewidth=2
        ))
        ax4.text(x_coarse, y_offset, data['name'], ha='center', va='center', fontsize=10, fontweight='bold')
        
        # Fine nodes
        n_children = len(data['children'])
        y_fine_start = y_offset - 0.15 * (n_children - 1) / 2
        
        for i, child in enumerate(data['children']):
            y_fine = y_fine_start + i * 0.15
            
            # Draw edge
            ax4.plot([x_coarse + 0.1, x_fine - 0.1], [y_offset, y_fine], 'k-', alpha=0.5, linewidth=1)
            
            # Fine node
            ax4.add_patch(mpatches.FancyBboxPatch(
                (x_fine - 0.12, y_fine - 0.04),
                0.24, 0.06,
                boxstyle="round,pad=0.005",
                facecolor='lightgreen',
                edgecolor='gray',
                linewidth=1
            ))
            ax4.text(x_fine, y_fine, child['name'], ha='center', va='center', fontsize=8)
        
        y_offset -= 0.35
    
    ax4.set_xlim(0, 1)
    ax4.set_ylim(0, 1)
    ax4.set_title("Hierarchical Structure", fontsize=14, fontweight='bold')
    
    # =========================================================================
    # Panel 5: Regime statistics
    # =========================================================================
    ax5 = plt.subplot(2, 3, 5)
    ax5.axis('off')
    
    # Compute statistics
    stats_text = "REGIME STATISTICS\n" + "=" * 40 + "\n\n"
    
    for k_coarse in range(model.num_coarse):
        coarse_mask = coarse_labels == k_coarse
        n_coarse = np.sum(coarse_mask)
        pct_coarse = 100 * n_coarse / len(coarse_labels)
        
        stats_text += f"{hierarchy[k_coarse]['name']}:\n"
        stats_text += f"  Samples: {n_coarse} ({pct_coarse:.1f}%)\n"
        
        for k_fine in range(model.num_fine_per_coarse):
            flat_idx = k_coarse * model.num_fine_per_coarse + k_fine
            fine_mask = fine_labels == flat_idx
            n_fine = np.sum(fine_mask)
            pct_fine = 100 * n_fine / n_coarse if n_coarse > 0 else 0
            
            child_name = hierarchy[k_coarse]['children'][k_fine]['name']
            stats_text += f"    └─ {child_name}: {n_fine} ({pct_fine:.1f}%)\n"
        
        stats_text += "\n"
    
    ax5.text(0.05, 0.95, stats_text, transform=ax5.transAxes,
             fontsize=9, verticalalignment='top', fontfamily='monospace')
    ax5.set_title("Distribution", fontsize=14, fontweight='bold')
    
    # =========================================================================
    # Panel 6: Regime confidence
    # =========================================================================
    ax6 = plt.subplot(2, 3, 6)
    
    # Coarse confidence
    coarse_confidence = np.max(p_coarse_np, axis=1)
    
    # Fine confidence (within assigned coarse regime)
    fine_confidence = []
    for i in range(len(coarse_labels)):
        k = coarse_labels[i]
        fine_confidence.append(np.max(p_fine_np[i, k, :]))
    fine_confidence = np.array(fine_confidence)
    
    ax6.hist(coarse_confidence, bins=30, alpha=0.7, label='Coarse', color='blue')
    ax6.hist(fine_confidence, bins=30, alpha=0.7, label='Fine', color='green')
    ax6.set_xlabel("Confidence (max probability)", fontsize=11)
    ax6.set_ylabel("Frequency", fontsize=11)
    ax6.set_title("Regime Confidence", fontsize=14, fontweight='bold')
    ax6.legend()
    ax6.grid(alpha=0.3)
    
    plt.tight_layout()
    
    # Save
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {save_path}")


def main():
    parser = argparse.ArgumentParser(description="Visualize hierarchical regimes")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/hierarchical_final.pth")
    parser.add_argument("--output", type=str, default="figures/hierarchical_regimes.png")
    
    args = parser.parse_args()
    
    # Load data
    dataset = ClimateDataset(TEST_NC, expert_features=[], gating_features=FEATURES_GATING, target="fco2")
    
    # Create model (dummy initialization for visualization)
    gating = HierarchicalGatingNetwork(input_dim=10, n_coarse=3, n_fine_per_coarse=3)
    model = HierarchicalSDMoSE(
        gating_network=gating,
        num_coarse=3,
        num_fine_per_coarse=3,
        expert_features=['sst', 'sss', 'log_chl', 'sst_gradient'],
    )
    
    # Load checkpoint if exists
    checkpoint_path = Path(args.checkpoint)
    if checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        if 'gating_state_dict' in checkpoint:
            model.gating_network.load_state_dict(checkpoint['gating_state_dict'])
        print(f"✓ Loaded checkpoint: {checkpoint_path}")
    else:
        print(f"Warning: Checkpoint not found, using random initialization")
    
    # Visualize
    model.eval()
    plot_hierarchical_regimes(model, dataset, args.output)


if __name__ == "__main__":
    main()
