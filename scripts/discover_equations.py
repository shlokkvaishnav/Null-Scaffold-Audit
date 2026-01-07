import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

import torch
import numpy as np
from pysr import PySRRegressor
import logging
from climate_discovery.models.gating import GatingNetwork
from climate_discovery.data.dataset import ClimateDataset

# Setup Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger(__name__)

# --- CONFIG ---
DATA_PATH = Path("data/03_processed/climate_fused_dataset.nc")
MODEL_PATH = Path("checkpoints/gating_warmstart.pth")
FEATURES = ['sst', 'sss', 'sin_month', 'cos_month', 'log_chl']
TARGET = 'fco2'
N_REGIMES = 6

# PySR Settings (Fast Discovery Mode)
PYSR_CONFIG = {
    "niterations": 40,  # Low for prototyping (increase to 100+ for final paper)
    "binary_operators": ["+", "-", "*", "/"],
    "unary_operators": ["exp", "square"], # 'log' is dangerous if inputs are negative, 'square' is safe
    "model_selection": "best", # Pick the best equation
    "loss": "loss(prediction, target) = (prediction - target)^2",
    "verbosity": 0,
    "temp_equation_file": True
}

def main():
    # 1. Load Data & Model
    logger.info("1. Loading Data & Gating Model...")
    dataset = ClimateDataset(DATA_PATH, FEATURES, mode='train')
    model = GatingNetwork(input_dim=len(FEATURES), num_regimes=N_REGIMES)
    model.load_state_dict(torch.load(MODEL_PATH, map_location='cpu'))
    model.eval()
    
    # 2. Assign Points to Regimes (Hard Assignment for now)
    logger.info("2. Partitioning Data by Regime...")
    with torch.no_grad():
        x_tensor = torch.tensor(dataset.X)
        _, probs = model(x_tensor)
        regime_labels = probs.argmax(dim=1).numpy()
    
    # Un-normalize data for PySR (We want equations in REAL units, like degrees C)
    # This makes the equations readable (e.g., "0.04 * SST" instead of "0.04 * (SST - 20)/5")
    X_raw = (dataset.X * (dataset.X_std + 1e-6)) + dataset.X_mean
    y_raw = dataset.y.ravel()
    
    # 3. Discover Equations per Regime
    equations = {}
    
    for k in range(N_REGIMES):
        # Filter data for this regime
        mask = regime_labels == k
        X_regime = X_raw[mask]
        y_regime = y_raw[mask]
        
        n_samples = len(y_regime)
        logger.info(f"\n🔍 Analyzing Regime {k} ({n_samples} samples)...")
        
        if n_samples < 1000:
            logger.warning(f"   ⚠️ Skipping Regime {k} (Too few samples)")
            continue
            
        # Downsample for speed (Symbolic Regression is slow on 100k+ points)
        if n_samples > 5000:
            indices = np.random.choice(n_samples, 5000, replace=False)
            X_regime = X_regime[indices]
            y_regime = y_regime[indices]
            
        # Run PySR
        regressor = PySRRegressor(**PYSR_CONFIG)
        regressor.fit(X_regime, y_regime)
        
        # Save Best Equation
        best_eq = regressor.sympy()
        score = regressor.get_best().score
        
        equations[k] = str(best_eq)
        logger.info(f"   🧪 Discovered Law: fCO2 = {best_eq}")
        logger.info(f"   📈 Score: {score:.4f}")

    # 4. Summary
    logger.info("\n📜 --- DISCOVERY REPORT ---")
    for k, eq in equations.items():
        print(f"Regime {k}: {eq}")

if __name__ == "__main__":
    main()