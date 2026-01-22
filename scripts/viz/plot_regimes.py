"""Plot regime maps and confidence (requires cartopy)."""
import sys
from pathlib import Path

root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(root / "src"))

import numpy as np
import torch
import matplotlib.pyplot as plt

try:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
except ImportError:
    print("Install cartopy: pip install cartopy")
    sys.exit(1)

from climate_discovery.config import FUSED_NC, CHECKPOINT_DIR, FIGURE_DIR, N_REGIMES
from climate_discovery.data.datasets import ClimateSpatialDataset
from climate_discovery.models.gating import GatingNetwork

FEATURES = ["sst", "sss", "sin_month", "cos_month", "log_chl"]
CHECKPOINT_PATH = CHECKPOINT_DIR / "gating_warmstart.pth"
FIGURE_DIR.mkdir(parents=True, exist_ok=True)


def main():
    model = GatingNetwork(input_dim=len(FEATURES), num_regimes=N_REGIMES)
    try:
        model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location="cpu"))
    except Exception:
        model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location="cpu"), strict=False)
    model.eval()

    dataset = ClimateSpatialDataset(str(FUSED_NC), FEATURES, mode="train")
    indices = [0, 6, 11]
    fig = plt.figure(figsize=(20, 12))

    for i, idx in enumerate(indices):
        if idx >= len(dataset):
            break
        sample = dataset[idx]
        img = sample["image"].unsqueeze(0)
        mask = sample["mask"].numpy()
        with torch.no_grad():
            B, C, H, W = img.shape
            _, probs = model(img.permute(0, 2, 3, 1).reshape(-1, C))
            probs_map = probs.reshape(H, W, N_REGIMES).numpy()
        regimes = np.argmax(probs_map, axis=2)
        confidence = np.max(probs_map, axis=2)

        ax = fig.add_subplot(2, 3, i + 1, projection=ccrs.Robinson())
        ax.set_title(f"Regimes (Step {idx})")
        ax.coastlines()
        ax.add_feature(cfeature.LAND, facecolor="gray")
        plot_data = np.ma.masked_where(~mask, regimes)
        mesh = ax.pcolormesh(dataset.ds.lon, dataset.ds.lat, plot_data, transform=ccrs.PlateCarree(), cmap="tab10")
        plt.colorbar(mesh, ax=ax, orientation="horizontal", pad=0.05, label="Regime ID")

        ax2 = fig.add_subplot(2, 3, i + 4, projection=ccrs.Robinson())
        ax2.set_title(f"Confidence (Step {idx})")
        ax2.coastlines()
        ax2.add_feature(cfeature.LAND, facecolor="gray")
        conf_data = np.ma.masked_where(~mask, confidence)
        mesh2 = ax2.pcolormesh(dataset.ds.lon, dataset.ds.lat, conf_data, transform=ccrs.PlateCarree(), cmap="plasma", vmin=0.4, vmax=1.0)
        plt.colorbar(mesh2, ax=ax2, orientation="horizontal", pad=0.05, label="Probability")

    plt.tight_layout()
    path = FIGURE_DIR / "regime_evolution.png"
    plt.savefig(path, dpi=300)
    print("Saved", path)


if __name__ == "__main__":
    main()
