import sys
from pathlib import Path
import numpy as np
import torch
import logging

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

# --- FIX: Import PySR before Torch to silence Segfault warning ---
from pysr import PySRRegressor
from climate_discovery.models.gating import GatingNetwork
from climate_discovery.data.dataset import ClimateSpatialDataset

# Setup Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger(__name__)

# --- CONFIG ---
DATA_PATH = Path("data/03_processed/climate_fused_dataset.nc")
CHECKPOINT_PATH = Path("checkpoints/gating_warmstart.pth")

# Features used by the Gating Network
GATING_FEATURES = ['sst', 'sss', 'sin_month', 'cos_month', 'log_chl']
TARGET_VAR = 'fco2' # Ensure this matches your NetCDF variable name
N_REGIMES = 6

# --- PHYSICS-INFORMED LOSS (Julia) ---
physics_loss_julia = """
function physics_loss(tree, dataset::Dataset, options)
    y_pred, flag = eval_tree_array(tree, dataset.X, options)
    if !flag
        return 1f9
    end
    mse = sum((y_pred .- dataset.y) .^ 2) / length(dataset.y)

    # Constraint: d/dSST should generally be positive (Thermodynamics)
    # We punish negative slopes heavily.
    epsilon = 1e-4
    X_plus = copy(dataset.X)
    X_plus[1, :] .+= epsilon # Nudge SST (Feature index 1 in Julia)
    
    y_plus, flag_plus = eval_tree_array(tree, X_plus, options)
    if !flag_plus
        return 1f9
    end

    d_sst = (y_plus .- y_pred) ./ epsilon
    violation_ratio = count(d_sst .< 0) / length(dataset.y)
    
    return mse + (100.0 * violation_ratio)
end
"""

PYSR_CONFIG = {
    "niterations": 40,  
    "binary_operators": ["+", "-", "*", "/"],
    "unary_operators": ["exp", "square", "log"], 
    "model_selection": "best",
    "loss_function": physics_loss_julia,
    "maxsize": 25, 
    "verbosity": 0,
    "temp_equation_file": True
}

def get_regime_data(dataset, model, device='cpu'):
    """
    Passes spatial maps through the Gating Network to split points into regimes.
    """
    logger.info("1. Partitioning Data by Regime...")
    model.eval()
    
    regime_buckets = {k: {'X': [], 'y': []} for k in range(N_REGIMES)}
    n_feats = len(GATING_FEATURES)
    
    for i in range(len(dataset)):
        sample = dataset[i]
        
        # 1. Prepare Input for Gating Network
        img_all = sample['image'] # (C_total, H, W)
        mask = sample['mask'].numpy() # (H, W)
        
        # Split Inputs (X) vs Target (y)
        img_gating = img_all[:n_feats] 
        img_target = img_all[n_feats]  
        
        # 2. Get Regime Labels
        with torch.no_grad():
            img_tensor = img_gating.unsqueeze(0).to(device)
            B, C, H, W = img_tensor.shape
            img_flat = img_tensor.permute(0, 2, 3, 1).reshape(-1, C)
            _, probs = model(img_flat)
            regimes_flat = probs.argmax(dim=1).cpu().numpy() 
            
        # 3. Flatten Data
        mask_flat = mask.flatten()
        target_flat = img_target.numpy().flatten()
        features_np = img_gating.numpy().reshape(n_feats, -1).T 
        
        # 4. Sort into Buckets (Valid pixels only)
        valid_indices = np.where(mask_flat)[0]
        
        if len(valid_indices) > 0:
            valid_regimes = regimes_flat[valid_indices]
            valid_feats = features_np[valid_indices]
            valid_targets = target_flat[valid_indices]
            
            for r in range(N_REGIMES):
                r_idx = np.where(valid_regimes == r)[0]
                if len(r_idx) > 0:
                    regime_buckets[r]['X'].append(valid_feats[r_idx])
                    regime_buckets[r]['y'].append(valid_targets[r_idx])

    # Merge buckets
    final_data = {}
    for r in range(N_REGIMES):
        if regime_buckets[r]['X']:
            final_data[r] = {
                'X': np.vstack(regime_buckets[r]['X']),
                'y': np.concatenate(regime_buckets[r]['y'])
            }
    return final_data

def main():
    logger.info("Loading Gating Network...")
    model = GatingNetwork(input_dim=len(GATING_FEATURES), num_regimes=N_REGIMES)
    
    # --- FIX: Caught generic Exception ---
    try:
        model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location='cpu'))
    except Exception:
        logger.warning("Strict loading failed, trying non-strict...")
        model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location='cpu'), strict=False)

    # Load Data
    load_features = GATING_FEATURES + [TARGET_VAR]
    logger.info(f"Loading Dataset with target: {TARGET_VAR}")
    dataset = ClimateSpatialDataset(DATA_PATH, load_features, mode='train')
    
    # Split by Regime
    regime_data = get_regime_data(dataset, model)
    
    equations = {}
    
    # Run PySR per Regime
    for r, data in regime_data.items():
        X_regime = data['X']
        y_regime = data['y']
        
        n_samples = len(y_regime)
        logger.info(f"\n🔍 Analyzing Regime {r} ({n_samples} samples)...")
        
        if n_samples < 1000:
            logger.warning("   Skipping (Too few samples)")
            continue
            
        # Subsample for speed
        if n_samples > 10000:
            idx = np.random.choice(n_samples, 10000, replace=False)
            X_regime = X_regime[idx]
            y_regime = y_regime[idx]
            
        regressor = PySRRegressor(**PYSR_CONFIG)
        regressor.fit(X_regime, y_regime, variable_names=GATING_FEATURES)
        
        best_eq = regressor.sympy()
        score = regressor.get_best().score
        
        equations[r] = str(best_eq)
        logger.info(f"   🧪 Law: {best_eq}")
        logger.info(f"   📈 Score: {score:.4f}")

    logger.info("\n📜 --- FINAL DISCOVERY REPORT ---")
    for k, eq in equations.items():
        print(f"Regime {k}: {eq}")

if __name__ == "__main__":
    main()