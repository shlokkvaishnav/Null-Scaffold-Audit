"""Discover symbolic laws per regime via PySR (after gating is trained)."""

import sys
from pathlib import Path

root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(root / "src"))

import logging

import numpy as np
import torch
from pysr import PySRRegressor

from climate_discovery.config import CHECKPOINT_DIR, FUSED_NC, N_REGIMES, TARGET
from climate_discovery.data.datasets import ClimateSpatialDataset
from climate_discovery.models.gating import GatingNetwork

GATING_FEATURES = ["sst", "sss", "sin_month", "cos_month", "log_chl"]
DATA_PATH = FUSED_NC
CHECKPOINT_PATH = CHECKPOINT_DIR / "gating_warmstart.pth"
PYSR_CONFIG = {
    "niterations": 40,
    "binary_operators": ["+", "-", "*", "/"],
    "unary_operators": ["exp", "square", "log"],
    "model_selection": "best",
    "maxsize": 25,
    "verbosity": 0,
    "temp_equation_file": True,
}

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger(__name__)


def get_regime_data(dataset, model, device="cpu"):
    model.eval()
    buckets = {k: {"X": [], "y": []} for k in range(N_REGIMES)}
    n_feats = len(GATING_FEATURES)
    for i in range(len(dataset)):
        sample = dataset[i]
        img_all = sample["image"]
        mask = sample["mask"].numpy()
        img_gating = img_all[:n_feats]
        img_target = img_all[n_feats]
        with torch.no_grad():
            flat = (
                img_gating.unsqueeze(0)
                .to(device)
                .permute(0, 2, 3, 1)
                .reshape(-1, n_feats)
            )
            _, probs = model(flat)
            regimes = probs.argmax(dim=1).cpu().numpy()
        mask_flat = mask.flatten()
        target_flat = img_target.numpy().flatten()
        feats = img_gating.numpy().reshape(n_feats, -1).T
        valid = np.where(mask_flat)[0]
        if len(valid) == 0:
            continue
        for r in range(N_REGIMES):
            r_idx = np.where(regimes[valid] == r)[0]
            if len(r_idx):
                buckets[r]["X"].append(feats[valid][r_idx])
                buckets[r]["y"].append(target_flat[valid][r_idx])
    return {
        r: {"X": np.vstack(buckets[r]["X"]), "y": np.concatenate(buckets[r]["y"])}
        for r in range(N_REGIMES)
        if buckets[r]["X"]
    }


def main():
    model = GatingNetwork(input_dim=len(GATING_FEATURES), num_regimes=N_REGIMES)
    try:
        model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location="cpu"))
    except Exception:
        model.load_state_dict(
            torch.load(CHECKPOINT_PATH, map_location="cpu"), strict=False
        )

    load_features = GATING_FEATURES + [TARGET]
    dataset = ClimateSpatialDataset(str(DATA_PATH), load_features, mode="train")
    regime_data = get_regime_data(dataset, model)

    equations = {}
    for r, data in regime_data.items():
        X, y = data["X"], data["y"]
        n = len(y)
        logger.info("Regime %d (%d samples)...", r, n)
        if n < 1000:
            continue
        if n > 10000:
            idx = np.random.default_rng(42).choice(n, 10000, replace=False)
            X, y = X[idx], y[idx]
        reg = PySRRegressor(**PYSR_CONFIG)
        reg.fit(X, y, variable_names=GATING_FEATURES)
        eq = reg.sympy()
        equations[r] = str(eq)
        logger.info("  Law: %s", eq)

    logger.info("--- Discovery report ---")
    for k, eq in equations.items():
        print(f"Regime {k}: {eq}")  # noqa: T201


if __name__ == "__main__":
    main()
