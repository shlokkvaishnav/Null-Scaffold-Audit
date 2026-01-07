import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

import torch
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from climate_discovery.models.gating import GatingNetwork
from climate_discovery.data.dataset import ClimateDataset

# --- CONFIG ---
DATA_PATH = Path("data/03_processed/climate_fused_dataset.nc")
MODEL_PATH = Path("checkpoints/gating_warmstart.pth")
OUTPUT_IMG = Path("figures/soft_regimes_map.png")
FEATURES = ['sst', 'sss', 'sin_month', 'cos_month', 'log_chl']
N_REGIMES = 6

def main():
    # 1. Load Data
    print("Loading data...")
    dataset = ClimateDataset(DATA_PATH, FEATURES, mode='train')
    
    # 2. Load Model
    print("Loading trained model...")
    model = GatingNetwork(input_dim=len(FEATURES), num_regimes=N_REGIMES)
    model.load_state_dict(torch.load(MODEL_PATH, map_location='cpu'))
    model.eval()
    
    # 3. Predict Soft Probabilities
    print("Predicting soft regimes...")
    with torch.no_grad():
        x = torch.tensor(dataset.X)
        _, probs = model(x)
        
    # Convert to DataFrame for plotting
    df = pd.DataFrame(dataset.coords, columns=['lat', 'lon'])
    
    # We want to see the "Confidence" of the model.
    # High confidence = Stable Regime. Low confidence = Dynamic Front.
    # We take the Max Probability as a proxy for confidence.
    max_probs, labels = torch.max(probs, dim=1)
    
    df['regime'] = labels.numpy()
    df['confidence'] = max_probs.numpy()
    
    # 4. Plot 1: The Regimes (should look like K-Means)
    print("Generating maps...")
    fig, axes = plt.subplots(1, 2, figsize=(20, 8))
    
    # Subplot 1: Hard Regimes
    sns.scatterplot(
        data=df.sample(50000), x='lon', y='lat', hue='regime', 
        palette='viridis', s=2, ax=axes[0], edgecolor='none'
    )
    axes[0].set_title("Neural Network Regimes (Hard Argmax)")
    
    # Subplot 2: Confidence (Where are the fronts?)
    # Low confidence (lighter color) means the model is "unsure" -> effectively a Front!
    sc = axes[1].scatter(
        df['lon'], df['lat'], c=df['confidence'], 
        cmap='plasma', s=1, alpha=0.5
    )
    plt.colorbar(sc, ax=axes[1], label="Regime Certainty (Probability)")
    axes[1].set_title("Regime Confidence (Low = Dynamic Fronts)")
    
    OUTPUT_IMG.parent.mkdir(exist_ok=True)
    plt.savefig(OUTPUT_IMG, dpi=150)
    print(f"✅ Saved visualization to {OUTPUT_IMG}")

if __name__ == "__main__":
    main()