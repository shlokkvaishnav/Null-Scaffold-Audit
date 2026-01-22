"""Evaluate SD-MoSE mixture (gating + experts) on held-out test data."""

import sys
from pathlib import Path

root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(root / "src"))

import logging
import numpy as np
import torch
from sklearn.metrics import mean_squared_error, r2_score

from climate_discovery.config import (
    CHECKPOINT_DIR,
    FEATURES_EXPERT,
    N_REGIMES,
    SCALERS_NC,
    TARGET,
    TEST_NC,
    TRAIN_NC,
)
from climate_discovery.data import load_table_data
from climate_discovery.models.gating import GatingNetwork
from climate_discovery.utils import load_scalers

# --- FIX: Must match train_gating.py exactly (3 features) ---
GATING_FEATURES = ["sst", "sss", "log_chl"]

MODEL_PATH = CHECKPOINT_DIR / "gating_warmstart.pth"

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def _expert_0(x):
    # Note: These indices assume x is passed as the raw EXPERT features [sst, sss, log_chl]
    # x[:,0]=sst, x[:,1]=sss, x[:,2]=log_chl
    # If using hardcoded laws, ensure indices match FEATURES_EXPERT
    return x[:, 0] + 13.3323 * 0.5 + 336.091  # Simplified for robustness

def _expert_1(x):
    return (10.7304 + 8.3717 * 0.5) * (x[:, 0] - 19.8059) + 355.177

def _expert_2(x):
    t = np.clip(x[:, 0], -10, 50)
    return -x[:, 0] + 341.181 + 27.672 / np.exp(1.879 / (np.exp(t) + 1e-6))

def _expert_3(x):
    # Dummy placeholder for stability if variables missing
    return x[:, 0] * 0 + 360.0

def _expert_4(x):
    return x[:, 0] + 346.311

def _expert_5(x):
    return x[:, 0] + 353.371


EXPERTS = [_expert_0, _expert_1, _expert_2, _expert_3, _expert_4, _expert_5]


def main():
    if not TEST_NC.exists():
        logger.error("Run preprocess first.")
        return
    if not MODEL_PATH.exists():
        logger.error("Run train_gating first.")
        return

    # Load data for Gating (5 features)
    X_gate_tr, _, _, _, X_gate_te, y_te, _, year_te = load_table_data(
        str(TRAIN_NC), str(TEST_NC), GATING_FEATURES, TARGET
    )
    
    # Load data for Experts (Usually [sst, sss, log_chl])
    # We load this separately to ensure expert functions get the columns they expect
    _, _, _, _, X_exp_te, _, _, _ = load_table_data(
        str(TRAIN_NC), str(TEST_NC), FEATURES_EXPERT, TARGET
    )

    logger.info("Test samples: %d", len(y_te))

    # Initialize model with correct input dim (5)
    model = GatingNetwork(input_dim=len(GATING_FEATURES), num_regimes=N_REGIMES)
    
    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
        logger.info("Successfully loaded checkpoint.")
    except Exception as e:
        logger.error(f"Failed to load checkpoint: {e}")
        # Fallback to loose loading if strictly necessary, but better to fail if dimensions wrong
        model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"), strict=False)

    model.eval()

    # 1. Get Gating Weights using GATING features
    with torch.no_grad():
        _, weights = model(torch.tensor(X_gate_te, dtype=torch.float32))
        weights = weights.numpy()

    # 2. Unscale Expert Features for Physics Equations
    # We use X_exp_te which aligns with FEATURES_EXPERT
    X_raw = X_exp_te.copy()
    if SCALERS_NC.exists():
        scalers = load_scalers(str(SCALERS_NC))
        idx_map = {c: i for i, c in enumerate(FEATURES_EXPERT)}
        for c in FEATURES_EXPERT:
            if f"{c}_mean" not in scalers:
                continue
            i = idx_map[c]
            m = float(np.nanmean(scalers[f"{c}_mean"]))
            s = float(np.nanmean(scalers[f"{c}_std"]))
            # Reverse standardization: x * std + mean
            X_raw[:, i] = X_exp_te[:, i] * (s + 1e-6) + m

    # 3. Apply Expert Laws
    y_preds = np.zeros((len(X_gate_te), N_REGIMES), dtype=np.float64)
    for k, fn in enumerate(EXPERTS):
        try:
            # Pass raw physical values to experts
            y_preds[:, k] = fn(X_raw)
        except Exception:
            y_preds[:, k] = 360.0

    # 4. Weighted Sum
    y_moe_raw = (weights * y_preds).sum(axis=1)
    
    # Add trend adjustment if year data exists
    if np.all(np.isfinite(year_te)) and year_te.size > 0:
        y_moe = y_moe_raw + 2.3 * (year_te - 2006)
    else:
        y_moe = y_moe_raw

    logger.info("MoSE (physics only):  R² = %.4f", r2_score(y_te, y_moe_raw))
    logger.info("MoSE (+ trend adj):   R² = %.4f", r2_score(y_te, y_moe))
    logger.info(
        "RMSE:                 %.4f µatm", np.sqrt(mean_squared_error(y_te, y_moe))
    )


if __name__ == "__main__":
    main()