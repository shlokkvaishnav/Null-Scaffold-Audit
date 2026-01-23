"""Generate publication-quality regime visualizations.

Figures:
1. Regime maps + confidence (seasonal snapshots)
2. Regime transition probability (fronts)
3. Latitudinal persistence
4. Regime usage
5. Seasonal mean regimes (DJF vs JJA)
6. Ensemble agreement
7. Front displacement
8. Entropy shift

Usage:
    python -m scripts.viz.plot_regimes
    python -m scripts.viz.plot_regimes --checkpoint path/to/gating.pth
"""
import sys  # noqa: E402
from pathlib import Path  # noqa: E402

root = Path(__file__).resolve().parents[2]  # noqa: E402
sys.path.insert(0, str(root / "src"))  # noqa: E402

import argparse  # noqa: E402
import logging  # noqa: E402

import matplotlib.pyplot as plt
import numpy as np
import torch
import xarray as xr

try:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
except ImportError:
    print("❌ cartopy not installed. Install: pip install cartopy")
    sys.exit(1)

from climate_discovery.config import (
    CHECKPOINT_DIR,
    FEATURES_GATING,
    FIGURE_DIR,
    FUSED_NC,
    ModelConfig,
)
from climate_discovery.models.gating import GatingNetwork

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


# =============================================================================
# DATA LOADING
# =============================================================================

def load_fused_dataset() -> xr.Dataset:
    """Load fused climate dataset."""
    if not FUSED_NC.exists():
        raise FileNotFoundError(
            f"Fused dataset not found: {FUSED_NC}\n"
            "Run: python -m scripts.data.preprocess_data"
        )
    
    ds = xr.open_dataset(FUSED_NC, engine="netcdf4")
    logger.info(f"Loaded dataset: {dict(ds.dims)}")
    return ds


def load_gating_model(checkpoint_path: Path, device: str = "cpu") -> GatingNetwork:
    """Load trained gating network."""
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    config = ModelConfig()
    
    model = GatingNetwork(
        input_dim=len(FEATURES_GATING),
        num_regimes=config.n_regimes,
        hidden_dims=config.gating_hidden_dims,
        dropout=config.gating_dropout,
        temperature=1.0,
    ).to(device)
    
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model.eval()
    logger.info(f"✓ Loaded model from {checkpoint_path}")
    
    return model


def compute_regime_probs(
    model: GatingNetwork,
    ds: xr.Dataset,
    timestep: int,
    device: str = "cpu",
) -> np.ndarray:
    """Compute regime probabilities for a timestep.
    
    Args:
        model: Gating network
        ds: xarray Dataset
        timestep: Time index
        device: Device
        
    Returns:
        Regime probabilities (lat, lon, K)
    """
    # Extract gating features
    X_list = []
    for feat in FEATURES_GATING:
        if feat in ds:
            X_list.append(ds[feat].isel(time=timestep).values)
        else:
            raise KeyError(f"Feature {feat} not in dataset")
    
    # Stack: (lat, lon, n_features)
    X = np.stack(X_list, axis=-1)
    
    # Flatten and remove NaNs
    original_shape = X.shape[:2]
    X_flat = X.reshape(-1, X.shape[-1])
    
    # Find valid (non-NaN) pixels
    valid_mask = ~np.isnan(X_flat).any(axis=1)
    X_valid = X_flat[valid_mask]
    
    # Forward pass
    X_tensor = torch.from_numpy(X_valid).float().to(device)
    
    with torch.no_grad():
        probs = model(X_tensor).cpu().numpy()
    
    # Reconstruct spatial map
    n_regimes = probs.shape[1]
    probs_map = np.full((*original_shape, n_regimes), np.nan)
    
    valid_indices = np.where(valid_mask)[0]
    flat_indices = np.unravel_index(valid_indices, original_shape)
    
    for k in range(n_regimes):
        probs_map[flat_indices[0], flat_indices[1], k] = probs[:, k]
    
    return probs_map


# =============================================================================
# FIGURE 1: REGIME MAPS + CONFIDENCE
# =============================================================================

def plot_regimes_and_confidence(
    ds: xr.Dataset,
    model: GatingNetwork,
    output_path: Path,
    device: str = "cpu",
):
    """Plot regime assignments and confidence at different timesteps."""
    logger.info("Generating Figure 1: Regimes + Confidence...")
    
    # Select representative timesteps (Jan, Jun, Dec)
    timesteps = [0, 5, 11]  # Months 1, 6, 12
    
    fig = plt.figure(figsize=(20, 12))
    
    for i, t in enumerate(timesteps):
        # Compute regime probs
        probs = compute_regime_probs(model, ds, t, device)
        
        # Dominant regime and confidence
        regimes = np.argmax(probs, axis=2)
        confidence = np.max(probs, axis=2)
        
        # Plot regimes
        ax1 = fig.add_subplot(2, 3, i+1, projection=ccrs.Robinson())
        ax1.set_title(f"Regimes (Month {t+1})", fontsize=14)
        ax1.coastlines()
        ax1.add_feature(cfeature.LAND, facecolor="lightgray")
        
        im1 = ax1.pcolormesh(
            ds.lon,
            ds.lat,
            regimes,
            transform=ccrs.PlateCarree(),
            cmap="tab10",
            vmin=0,
            vmax=9,
        )
        
        # Plot confidence
        ax2 = fig.add_subplot(2, 3, i+4, projection=ccrs.Robinson())
        ax2.set_title(f"Confidence (Month {t+1})", fontsize=14)
        ax2.coastlines()
        ax2.add_feature(cfeature.LAND, facecolor="lightgray")
        
        im2 = ax2.pcolormesh(
            ds.lon,
            ds.lat,
            confidence,
            transform=ccrs.PlateCarree(),
            cmap="plasma",
            vmin=0.4,
            vmax=1.0,
        )
        
        if i == 2:  # Add colorbars on last column
            plt.colorbar(im1, ax=ax1, orientation="horizontal", pad=0.05, label="Regime ID")
            plt.colorbar(im2, ax=ax2, orientation="horizontal", pad=0.05, label="Confidence")
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    logger.info(f"✓ Saved: {output_path}")
    plt.close()


# =============================================================================
# FIGURE 2: TRANSITION PROBABILITY
# =============================================================================

def plot_transition_probability(
    ds: xr.Dataset,
    model: GatingNetwork,
    output_path: Path,
    device: str = "cpu",
):
    """Plot regime transition probability (identifies fronts)."""
    logger.info("Generating Figure 2: Transition Probability...")
    
    n_timesteps = len(ds.time)
    lat_len, lon_len = len(ds.lat), len(ds.lon)
    
    transitions = np.zeros((lat_len, lon_len))
    counts = np.zeros((lat_len, lon_len))
    
    prev_regimes = None
    
    for t in range(n_timesteps):
        probs = compute_regime_probs(model, ds, t, device)
        curr_regimes = np.argmax(probs, axis=2)
        
        if prev_regimes is not None:
            # Count transitions (where regime changed)
            changed = curr_regimes != prev_regimes
            valid = ~np.isnan(curr_regimes)
            
            transitions += changed & valid
            counts += valid
        
        prev_regimes = curr_regimes
    
    # Transition probability
    transition_prob = transitions / (counts + 1e-6)
    transition_prob[counts == 0] = np.nan
    
    # Plot
    fig = plt.figure(figsize=(12, 6))
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.Robinson())
    ax.set_title("Regime Transition Probability (Frontal Zones)", fontsize=16)
    ax.coastlines()
    ax.add_feature(cfeature.LAND, facecolor="lightgray")
    
    im = ax.pcolormesh(
        ds.lon,
        ds.lat,
        transition_prob,
        transform=ccrs.PlateCarree(),
        cmap="inferno",
        vmin=0,
        vmax=0.5,
    )
    
    plt.colorbar(im, ax=ax, orientation="horizontal", pad=0.05, label="Transition Probability")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    logger.info(f"✓ Saved: {output_path}")
    plt.close()


# =============================================================================
# FIGURE 3: LATITUDINAL PERSISTENCE
# =============================================================================

def plot_latitudinal_persistence(
    ds: xr.Dataset,
    model: GatingNetwork,
    output_path: Path,
    device: str = "cpu",
):
    """Plot regime persistence by latitude."""
    logger.info("Generating Figure 3: Latitudinal Persistence...")
    
    n_timesteps = len(ds.time)
    lat_vals = ds.lat.values
    n_lat = len(lat_vals)
    
    persistence = np.zeros(n_lat)
    counts = np.zeros(n_lat)
    
    prev_regimes = None
    
    for t in range(n_timesteps):
        probs = compute_regime_probs(model, ds, t, device)
        curr_regimes = np.argmax(probs, axis=2)
        
        if prev_regimes is not None:
            # Count where regime stayed the same
            same = curr_regimes == prev_regimes
            valid = ~np.isnan(curr_regimes)
            
            # Sum across longitudes for each latitude
            persistence += np.nansum(same & valid, axis=1)
            counts += np.nansum(valid, axis=1)
        
        prev_regimes = curr_regimes
    
    # Persistence fraction
    persistence_frac = persistence / (counts + 1e-6)
    
    # Plot
    plt.figure(figsize=(8, 6))
    plt.plot(lat_vals, persistence_frac, linewidth=2)
    plt.xlabel("Latitude (°)", fontsize=12)
    plt.ylabel("Regime Persistence", fontsize=12)
    plt.title("Latitudinal Regime Stability", fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.xlim(-90, 90)
    plt.ylim(0, 1)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    logger.info(f"✓ Saved: {output_path}")
    plt.close()


# =============================================================================
# FIGURE 4: REGIME USAGE
# =============================================================================

def plot_regime_usage(
    ds: xr.Dataset,
    model: GatingNetwork,
    output_path: Path,
    device: str = "cpu",
):
    """Plot global regime usage."""
    logger.info("Generating Figure 4: Regime Usage...")
    
    config = ModelConfig()
    n_regimes = config.n_regimes
    n_timesteps = len(ds.time)
    
    usage = np.zeros(n_regimes)
    
    for t in range(n_timesteps):
        probs = compute_regime_probs(model, ds, t, device)
        
        # Average probability for each regime (across space)
        for k in range(n_regimes):
            usage[k] += np.nanmean(probs[:, :, k])
    
    usage /= n_timesteps
    
    # Plot
    plt.figure(figsize=(8, 6))
    plt.bar(range(n_regimes), usage, color="steelblue")
    plt.xlabel("Regime ID", fontsize=12)
    plt.ylabel("Mean Probability", fontsize=12)
    plt.title("Global Regime Usage", fontsize=14)
    plt.xticks(range(n_regimes))
    plt.grid(True, axis="y", alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    logger.info(f"✓ Saved: {output_path}")
    plt.close()


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Generate regime visualization figures"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to gating checkpoint"
    )
    parser.add_argument(
        "--figures",
        nargs="+",
        default=["all"],
        choices=["all", "1", "2", "3", "4"],
        help="Which figures to generate"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output directory"
    )
    
    args = parser.parse_args()
    
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Output directory
    output_dir = Path(args.output) if args.output else FIGURE_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Checkpoint path
    if args.checkpoint:
        checkpoint_path = Path(args.checkpoint)
    else:
        checkpoint_path = CHECKPOINT_DIR / "gating_best.pth"
    
    # Load data and model
    logger.info("=" * 60)
    logger.info("LOADING DATA & MODEL")
    logger.info("=" * 60)
    
    ds = load_fused_dataset()
    model = load_gating_model(checkpoint_path, device)
    
    # Generate figures
    logger.info("\n" + "=" * 60)
    logger.info("GENERATING FIGURES")
    logger.info("=" * 60)
    
    figures_to_generate = args.figures
    if "all" in figures_to_generate:
        figures_to_generate = ["1", "2", "3", "4"]
    
    if "1" in figures_to_generate:
        plot_regimes_and_confidence(
            ds, model, output_dir / "figure1_regimes_confidence.png", device
        )
    
    if "2" in figures_to_generate:
        plot_transition_probability(
            ds, model, output_dir / "figure2_transition_probability.png", device
        )
    
    if "3" in figures_to_generate:
        plot_latitudinal_persistence(
            ds, model, output_dir / "figure3_latitudinal_persistence.png", device
        )
    
    if "4" in figures_to_generate:
        plot_regime_usage(
            ds, model, output_dir / "figure4_regime_usage.png", device
        )
    
    logger.info("\n" + "=" * 60)
    logger.info("✓ ALL FIGURES GENERATED")
    logger.info("=" * 60)
    logger.info(f"Output directory: {output_dir}")


if __name__ == "__main__":
    main()