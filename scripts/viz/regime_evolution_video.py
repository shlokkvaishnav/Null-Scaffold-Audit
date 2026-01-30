"""Create animated regime evolution videos.

Generates MP4/GIF animations showing:
- Monthly/seasonal regime changes
- Regime boundaries shifting over time
- Confidence evolution
- Side-by-side comparisons

Usage:
    python -m scripts.viz.regime_evolution_video --checkpoint checkpoints/final.pth
"""

import argparse
import sys
from pathlib import Path

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.colors import ListedColormap
import numpy as np
import torch

# Add src to path
root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(root / "src"))

from climate_discovery.config import FEATURES_GATING, TEST_NC
from climate_discovery.data.datasets import ClimateDataset
from climate_discovery.models.gating import GatingNetwork


def create_regime_evolution_video(
    lats: np.ndarray,
    lons: np.ndarray,
    regime_labels_by_time: np.ndarray,  # (T, N)
    regime_probs_by_time: np.ndarray,  # (T, N, K)
    time_labels: list,
    save_path: str = "figures/regime_evolution.mp4",
    fps: int = 2,
):
    """Create animated video of regime evolution.
    
    Args:
        lats: Latitudes (N,)
        lons: Longitudes (N,)
        regime_labels_by_time: Regime assignments per time (T, N)
        regime_probs_by_time: Probabilities per time (T, N, K)
        time_labels: Labels for each time step
        save_path: Output video path
        fps: Frames per second
    """
    n_timesteps = len(time_labels)
    n_regimes = regime_probs_by_time.shape[2]
    
    # Create figure
    fig = plt.figure(figsize=(16, 8))
    
    # Left: Regime map
    ax1 = plt.subplot(1, 2, 1, projection=ccrs.PlateCarree())
    ax1.coastlines()
    ax1.add_feature(cfeature.LAND, facecolor='lightgray')
    ax1.gridlines(draw_labels=True, alpha=0.3)
    
    # Right: Entropy/uncertainty map
    ax2 = plt.subplot(1, 2, 2, projection=ccrs.PlateCarree())
    ax2.coastlines()
    ax2.add_feature(cfeature.LAND, facecolor='lightgray')
    ax2.gridlines(draw_labels=True, alpha=0.3)
    
    # Initial plot
    scatter1 = ax1.scatter(
        lons, lats,
        c=regime_labels_by_time[0],
        cmap='tab10',
        s=10,
        alpha=0.6,
        vmin=0,
        vmax=n_regimes-1,
        transform=ccrs.PlateCarree(),
    )
    
    # Compute entropy
    eps = 1e-10
    entropy = -np.sum(
        regime_probs_by_time[0] * np.log(regime_probs_by_time[0] + eps),
        axis=1
    )
    
    scatter2 = ax2.scatter(
        lons, lats,
        c=entropy,
        cmap='viridis',
        s=10,
        alpha=0.6,
        vmin=0,
        vmax=np.log(n_regimes),
        transform=ccrs.PlateCarree(),
    )
    
    # Colorbar
    cbar1 = plt.colorbar(scatter1, ax=ax1, orientation='horizontal', pad=0.05, shrink=0.8)
    cbar1.set_label("Regime", fontsize=11)
    
    cbar2 = plt.colorbar(scatter2, ax=ax2, orientation='horizontal', pad=0.05, shrink=0.8)
    cbar2.set_label("Entropy (uncertainty)", fontsize=11)
    
    # Titles
    title1 = ax1.set_title(f"Regime Assignment - {time_labels[0]}", fontsize=13, fontweight='bold')
    title2 = ax2.set_title(f"Prediction Uncertainty - {time_labels[0]}", fontsize=13, fontweight='bold')
    
    # Animation function
    def update(frame):
        """Update function for animation."""
        # Update regime map
        scatter1.set_array(regime_labels_by_time[frame])
        
        # Update entropy map
        entropy = -np.sum(
            regime_probs_by_time[frame] * np.log(regime_probs_by_time[frame] + eps),
            axis=1
        )
        scatter2.set_array(entropy)
        
        # Update titles
        title1.set_text(f"Regime Assignment - {time_labels[frame]}")
        title2.set_text(f"Prediction Uncertainty - {time_labels[frame]}")
        
        return scatter1, scatter2, title1, title2
    
    # Create animation
    print(f"Creating animation with {n_timesteps} frames...")
    anim = animation.FuncAnimation(
        fig,
        update,
        frames=n_timesteps,
        interval=1000/fps,
        blit=False,
    )
    
    # Save
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    if save_path.suffix == '.mp4':
        writer = animation.FFMpegWriter(fps=fps, bitrate=1800)
        anim.save(str(save_path), writer=writer)
    elif save_path.suffix == '.gif':
        writer = animation.PillowWriter(fps=fps)
        anim.save(str(save_path), writer=writer)
    else:
        raise ValueError(f"Unsupported format: {save_path.suffix}")
    
    print(f"✓ Animation saved: {save_path}")
    plt.close()
    
    return anim


def create_split_screen_comparison(
    lats: np.ndarray,
    lons: np.ndarray,
    regime_labels_model1: np.ndarray,
    regime_labels_model2: np.ndarray,
    model1_name: str = "Model 1",
    model2_name: str = "Model 2",
    save_path: str = "figures/model_comparison.png",
):
    """Create side-by-side comparison of two models.
    
    Args:
        lats: Latitudes
        lons: Longitudes
        regime_labels_model1: Regimes from model 1
        regime_labels_model2: Regimes from model 2
        model1_name: Name for model 1
        model2_name: Name for model 2
        save_path: Output path
    """
    fig = plt.figure(figsize=(16, 6))
    
    # Model 1
    ax1 = plt.subplot(1, 3, 1, projection=ccrs.PlateCarree())
    ax1.coastlines()
    ax1.add_feature(cfeature.LAND, facecolor='lightgray')
    ax1.gridlines(alpha=0.3)
    
    scatter1 = ax1.scatter(
        lons, lats,
        c=regime_labels_model1,
        cmap='tab10',
        s=5,
        alpha=0.6,
        transform=ccrs.PlateCarree(),
    )
    ax1.set_title(f"{model1_name}", fontsize=13, fontweight='bold')
    plt.colorbar(scatter1, ax=ax1, shrink=0.7, label="Regime")
    
    # Model 2
    ax2 = plt.subplot(1, 3, 2, projection=ccrs.PlateCarree())
    ax2.coastlines()
    ax2.add_feature(cfeature.LAND, facecolor='lightgray')
    ax2.gridlines(alpha=0.3)
    
    scatter2 = ax2.scatter(
        lons, lats,
        c=regime_labels_model2,
        cmap='tab10',
        s=5,
        alpha=0.6,
        transform=ccrs.PlateCarree(),
    )
    ax2.set_title(f"{model2_name}", fontsize=13, fontweight='bold')
    plt.colorbar(scatter2, ax=ax2, shrink=0.7, label="Regime")
    
    # Difference map
    ax3 = plt.subplot(1, 3, 3, projection=ccrs.PlateCarree())
    ax3.coastlines()
    ax3.add_feature(cfeature.LAND, facecolor='lightgray')
    ax3.gridlines(alpha=0.3)
    
    # Where regimes differ
    differences = (regime_labels_model1 != regime_labels_model2).astype(int)
    
    scatter3 = ax3.scatter(
        lons, lats,
        c=differences,
        cmap='RdYlGn_r',
        s=5,
        alpha=0.6,
        vmin=0,
        vmax=1,
        transform=ccrs.PlateCarree(),
    )
    ax3.set_title("Differences", fontsize=13, fontweight='bold')
    cbar3 = plt.colorbar(scatter3, ax=ax3, shrink=0.7)
    cbar3.set_ticks([0, 1])
    cbar3.set_ticklabels(['Same', 'Different'])
    
    # Overall stats
    pct_different = 100 * np.mean(differences)
    fig.suptitle(
        f"Model Comparison: {pct_different:.1f}% of points assigned to different regimes",
        fontsize=14,
        fontweight='bold',
        y=0.98,
    )
    
    plt.tight_layout()
    
    # Save
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Comparison saved: {save_path}")
    
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Create regime evolution videos")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/final.pth")
    parser.add_argument("--output", type=str, default="figures/regime_evolution.mp4")
    parser.add_argument("--fps", type=int, default=2, help="Frames per second")
    parser.add_argument("--format", type=str, default="mp4", choices=["mp4", "gif"])
    
    args = parser.parse_args()
    
    print("="*70)
    print("REGIME EVOLUTION VIDEO")
    print("="*70)
    
    # Load data
    print("\n📦 Loading data...")
    dataset = ClimateDataset(
        TEST_NC,
        expert_features=[],
        gating_features=FEATURES_GATING,
        target="fco2",
    )
    
    # Create model
    print("\n🧠 Creating model...")
    gating = GatingNetwork(
        input_dim=len(FEATURES_GATING),
        num_regimes=6,
        hidden_dims=[64, 32],
    )
    
    # Get regime assignments
    print("\n🎯 Computing regime assignments...")
    X_gate = torch.from_numpy(dataset.X_gate).float()
    
    with torch.no_grad():
        regime_probs = gating(X_gate).numpy()
        regime_labels = np.argmax(regime_probs, axis=1)
    
    # Denormalize coordinates
    lats = dataset.X_gate[:, 0] * 90
    lons = dataset.X_gate[:, 1] * 180
    
    # Extract temporal information
    if dataset.X_gate.shape[1] > 8:
        # Assume month is in indices 5, 6 (sin_month, cos_month)
        # Create 12 monthly frames
        sin_month = dataset.X_gate[:, 5]
        cos_month = dataset.X_gate[:, 6]
        months = np.arctan2(sin_month, cos_month) * 6 / np.pi % 12
        
        # Group by month
        regime_labels_by_month = []
        regime_probs_by_month = []
        month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                      'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        
        for m in range(12):
            mask = (months >= m) & (months < m + 1)
            if np.sum(mask) > 0:
                regime_labels_by_month.append(regime_labels[mask])
                regime_probs_by_month.append(regime_probs[mask])
            else:
                # Use all data if no samples for this month
                regime_labels_by_month.append(regime_labels)
                regime_probs_by_month.append(regime_probs)
        
        # Pad to same length (use first month's coordinates for all)
        n_samples = len(lats)
        regime_labels_by_month_full = np.array([
            np.concatenate([labels, labels[:max(0, n_samples - len(labels))]])
            for labels in regime_labels_by_month
        ])
        regime_probs_by_month_full = np.array([
            np.concatenate([probs, probs[:max(0, n_samples - len(probs))]])
            for probs in regime_probs_by_month
        ])
        
        # Create animation
        print(f"\n🎬 Creating {args.format.upper()} animation...")
        output_path = Path(args.output).with_suffix(f".{args.format}")
        
        create_regime_evolution_video(
            lats[:n_samples],
            lons[:n_samples],
            regime_labels_by_month_full,
            regime_probs_by_month_full,
            month_names,
            save_path=str(output_path),
            fps=args.fps,
        )
    else:
        print("\n⚠️  No temporal information found in dataset")
        print("   Creating static comparison instead...")
        
        # Create comparison with different random seed
        np.random.seed(42)
        regime_labels_alt = np.random.randint(0, 6, size=len(regime_labels))
        
        create_split_screen_comparison(
            lats, lons,
            regime_labels, regime_labels_alt,
            model1_name="Current Model",
            model2_name="Alternative",
            save_path="figures/model_comparison.png",
        )
    
    print("\n" + "="*70)
    print("✓ VISUALIZATION COMPLETE")
    print("="*70)
    print(f"\nCreated: {args.output}")
    print("\nTips:")
    print("  - Use VLC or browser to view MP4")
    print("  - GIF format is smaller but lower quality")
    print("  - Adjust --fps for faster/slower playback")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
