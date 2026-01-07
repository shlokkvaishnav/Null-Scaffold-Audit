import sys
from pathlib import Path
import torch
import matplotlib.pyplot as plt
import numpy as np

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

try:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
except ImportError:
    print("❌ Cartopy missing. Install with 'pip install cartopy'")
    sys.exit(1)

from climate_discovery.models.gating import GatingNetwork
from climate_discovery.data.dataset import ClimateSpatialDataset

# --- CONFIG ---
CHECKPOINT_PATH = Path("checkpoints/gating_warmstart.pth")
DATA_PATH = Path("data/03_processed/climate_fused_dataset.nc")
FIGURE_DIR = Path("figures")
FIGURE_DIR.mkdir(exist_ok=True)
FEATURES = ['sst', 'sss', 'sin_month', 'cos_month', 'log_chl']
N_REGIMES = 6

def visualize():
    print("1. Loading Model & Data...")
    
    # Load Model
    model = GatingNetwork(input_dim=len(FEATURES), num_regimes=N_REGIMES)
    try:
        model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location='cpu'))
    except Exception:
        # Fallback for minor architecture mismatches
        model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location='cpu'), strict=False)
    model.eval()
    
    # Load Dataset
    dataset = ClimateSpatialDataset(DATA_PATH, FEATURES, mode='train')
    
    # Visualize 3 different time steps (Jan, June, Dec)
    indices = [0, 6, 11] 
    
    fig = plt.figure(figsize=(20, 12))
    
    for i, idx in enumerate(indices):
        if idx >= len(dataset): 
            break
        
        sample = dataset[idx]
        img = sample['image'].unsqueeze(0) # (1, C, H, W)
        mask = sample['mask'].numpy()
        
        with torch.no_grad():
            B, C, H, W = img.shape
            img_flat = img.permute(0, 2, 3, 1).reshape(-1, C)
            _, probs = model(img_flat)
            probs_map = probs.reshape(H, W, N_REGIMES).numpy()
            
            # Hard Assignment
            regimes = np.argmax(probs_map, axis=2)
            # Confidence
            confidence = np.max(probs_map, axis=2)

        # Plot Regimes
        ax = fig.add_subplot(2, 3, i+1, projection=ccrs.Robinson())
        ax.set_title(f"Regimes (Step {idx})")
        ax.coastlines()
        ax.add_feature(cfeature.LAND, facecolor='gray')
        
        plot_data = np.ma.masked_where(~mask, regimes)
        mesh = ax.pcolormesh(dataset.ds.lon, dataset.ds.lat, plot_data, 
                             transform=ccrs.PlateCarree(), cmap='tab10')
        plt.colorbar(mesh, ax=ax, orientation='horizontal', pad=0.05, label='Regime ID')
        
        # Plot Confidence
        ax2 = fig.add_subplot(2, 3, i+4, projection=ccrs.Robinson())
        ax2.set_title(f"Confidence (Step {idx})")
        ax2.coastlines()
        ax2.add_feature(cfeature.LAND, facecolor='gray')
        
        conf_data = np.ma.masked_where(~mask, confidence)
        mesh2 = ax2.pcolormesh(dataset.ds.lon, dataset.ds.lat, conf_data,
                               transform=ccrs.PlateCarree(), cmap='plasma', vmin=0.4, vmax=1.0)
        plt.colorbar(mesh2, ax=ax2, orientation='horizontal', pad=0.05, label='Probability')

    plt.tight_layout()
    save_path = FIGURE_DIR / "regime_evolution.png"
    plt.savefig(save_path, dpi=300)
    print(f"✅ Visualization saved to {save_path}")

if __name__ == "__main__":
    visualize()