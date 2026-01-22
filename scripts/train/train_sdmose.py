"""SD-MoSE training loop: soft gating + symbolic experts, iterate 3–5x."""

import sys
from pathlib import Path

root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(root / "src"))

import logging

import numpy as np
import torch
import torch.optim as optim

from climate_discovery.config import (
    CHECKPOINT_DIR,
    FEATURES_ALL,
    N_REGIMES,
    SDMOSE_ITERATIONS,
    TARGET,
    TEST_NC,
    TRAIN_NC,
)
from climate_discovery.data import load_table_data
from climate_discovery.evaluation import compute_r2_rmse
from climate_discovery.models.gating import GatingNetwork
from climate_discovery.models.symbolic import KMeansSymbolicRegressor

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 4096
EPOCHS = 10
LR = 1e-3
N_SAMPLES = 50_000

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger(__name__)


def main():
    if not TRAIN_NC.exists() or not TEST_NC.exists():
        logger.error("Run preprocess first.")
        return

    X_tr, y_tr, _, _, X_te, y_te, _, _ = load_table_data(
        str(TRAIN_NC), str(TEST_NC), FEATURES_ALL, TARGET
    )
    n = min(N_SAMPLES, len(y_tr))
    rng = np.random.default_rng(42)
    idx = rng.choice(len(y_tr), n, replace=False)
    X_tr, y_tr = X_tr[idx].astype(np.float32), y_tr[idx].astype(np.float32)

    km_sr = KMeansSymbolicRegressor(n_clusters=N_REGIMES, max_depth=6)
    km_sr.fit(X_tr, y_tr, variable_names=FEATURES_ALL)
    labels = km_sr.kmeans.predict(X_tr)

    expert_preds = np.zeros((len(X_tr), N_REGIMES), dtype=np.float32)
    for k in range(N_REGIMES):
        mask = labels == k
        if np.sum(mask) > 0 and km_sr.symbolic_models[k] is not None:
            expert_preds[mask, k] = km_sr.symbolic_models[k].predict(X_tr[mask])
        else:
            expert_preds[:, k] = np.nanmean(y_tr)
    for k in range(N_REGIMES):
        bad = ~np.isfinite(expert_preds[:, k])
        if np.any(bad):
            expert_preds[bad, k] = np.nanmean(y_tr)

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    gate = GatingNetwork(input_dim=X_tr.shape[1], num_regimes=N_REGIMES).to(DEVICE)
    opt = optim.Adam(gate.parameters(), lr=LR)
    y_tr_t = torch.from_numpy(y_tr).float().to(DEVICE)

    for it in range(SDMOSE_ITERATIONS):
        logger.info("--- SD-MoSE iteration %d ---", it + 1)
        with torch.no_grad():
            _, probs = gate(torch.from_numpy(X_tr).float().to(DEVICE))
            probs = probs.cpu().numpy()
        hard = probs.argmax(axis=1)
        for k in range(N_REGIMES):
            mask = hard == k
            if np.sum(mask) < 20:
                continue
            if km_sr.symbolic_models[k] is not None:
                km_sr.symbolic_models[k].fit(
                    X_tr[mask], y_tr[mask], variable_names=FEATURES_ALL
                )
            else:
                from pysr import PySRRegressor

                m = PySRRegressor(
                    niterations=15,
                    binary_operators=["+", "-", "*"],
                    unary_operators=[],
                    maxsize=12,
                    random_state=42,
                    verbosity=0,
                )
                m.fit(X_tr[mask], y_tr[mask], variable_names=FEATURES_ALL)
                km_sr.symbolic_models[k] = m

        expert_preds = np.zeros((len(X_tr), N_REGIMES), dtype=np.float32)
        for k in range(N_REGIMES):
            if km_sr.symbolic_models[k] is not None:
                expert_preds[:, k] = km_sr.symbolic_models[k].predict(X_tr)
            else:
                expert_preds[:, k] = np.nanmean(y_tr)

        gate.train()
        for _ in range(EPOCHS):
            perm = rng.permutation(len(X_tr))
            for start in range(0, len(X_tr), BATCH_SIZE):
                end = min(start + BATCH_SIZE, len(X_tr))
                bi = perm[start:end]
                x_b = torch.from_numpy(X_tr[bi]).float().to(DEVICE)
                e_b = torch.from_numpy(expert_preds[bi]).float().to(DEVICE)
                opt.zero_grad()
                _, pi = gate(x_b)
                loss = ((pi * e_b).sum(dim=1) - y_tr_t[bi]).pow(2).mean()
                loss.backward()
                opt.step()

        gate.eval()
        with torch.no_grad():
            _, pi_te = gate(torch.from_numpy(X_te.astype(np.float32)).to(DEVICE))
        e_te = np.column_stack(
            [
                (
                    km_sr.symbolic_models[k].predict(X_te)
                    if km_sr.symbolic_models[k] is not None
                    else np.full(len(X_te), np.nanmean(y_tr))
                )
                for k in range(N_REGIMES)
            ]
        )
        y_mix = (pi_te.cpu().numpy() * e_te).sum(axis=1)
        r2, rmse = compute_r2_rmse(y_te, y_mix)
        logger.info("   Test R² = %.4f  RMSE = %.4f", r2, rmse)

    path = CHECKPOINT_DIR / "sdmose_gating.pth"
    torch.save(gate.state_dict(), path)
    logger.info("Saved %s", path)


if __name__ == "__main__":
    main()
