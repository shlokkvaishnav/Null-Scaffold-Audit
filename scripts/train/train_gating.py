"""Train soft gating network (K-means teacher + spatial + temporal smoothness)."""

import argparse
import sys
from pathlib import Path

root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(root / "src"))

import logging
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.cluster import KMeans
from torch.utils.data import DataLoader
from tqdm import tqdm

from climate_discovery.config import CHECKPOINT_DIR, FUSED_NC, N_REGIMES
from climate_discovery.data.datasets import ClimateSpatialDataset
from climate_discovery.models.gating import GatingNetwork
from climate_discovery.models.losses import RegimeConsistencyLoss

# ======================================================
# Gating features (NO raw time, safe dynamics only)
# ======================================================
FEATURES = [
    "lat_norm",
    "sin_lon",
    "cos_lon",
    "sst",
    "sss",
    "log_chl",
    "season_strength",
]

# ======================================================
# Hyperparameters
# ======================================================
BATCH_SIZE = 64
EPOCHS = 10              # ↑ slightly longer to allow smooth fronts
LR = 1e-3
SPATIAL_WEIGHT = 1.5     # reduced from 2.0 to allow front movement
TEMPORAL_WEIGHT = 0.1    # weak but essential front persistence
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out_dir", type=str, default=str(CHECKPOINT_DIR))
    args = parser.parse_args()

    # 🔒 Fix randomness (VERY IMPORTANT for ensemble)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Training gating model with seed={args.seed}")
    logger.info("Loading spatial dataset...")

    dataset = ClimateSpatialDataset(str(FUSED_NC), FEATURES, mode="train")

    # ==================================================
    # K-means teacher (warm start)
    # ==================================================
    full_data = dataset.data.permute(0, 2, 3, 1)  # (T, H, W, C)
    mask = dataset.mask
    mask_expanded = mask.unsqueeze(0).expand(full_data.shape[0], -1, -1)
    X_all = full_data[mask_expanded].numpy()

    n_samples = min(100_000, len(X_all))
    rng = np.random.default_rng(42)
    idx = rng.choice(len(X_all), n_samples, replace=False)

    kmeans = KMeans(n_clusters=N_REGIMES, n_init=5, random_state=42)
    kmeans.fit(X_all[idx])

    labels_flat = []
    chunk_size = 500_000
    for i in range(0, len(X_all), chunk_size):
        labels_flat.append(kmeans.predict(X_all[i : i + chunk_size]))
    labels_flat = np.concatenate(labels_flat)

    teacher_targets = torch.full(
        (full_data.shape[0], full_data.shape[1], full_data.shape[2]),
        -1,
        dtype=torch.long,
    )
    teacher_targets[mask_expanded] = torch.from_numpy(labels_flat).long()
    dataset.set_teacher_targets(teacher_targets)

    logger.info("Teacher targets ready.")

    # ==================================================
    # Model & losses
    # ==================================================
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

    model = GatingNetwork(
        input_dim=len(FEATURES),
        num_regimes=N_REGIMES,
        hidden_dim=64,
    ).to(DEVICE)

    optimizer = optim.Adam(model.parameters(), lr=LR)
    task_loss_fn = nn.NLLLoss(ignore_index=-1)
    spatial_loss_fn = RegimeConsistencyLoss(weight=SPATIAL_WEIGHT)

    # ==================================================
    # Training loop
    # ==================================================
    model.train()
    for epoch in range(EPOCHS):
        epoch_loss = 0.0

        for batch in tqdm(loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
            img = batch["image"].to(DEVICE)    # (B, C, H, W)
            target = batch["target"].to(DEVICE)

            optimizer.zero_grad()

            B, C, H, W = img.shape
            img_flat = img.permute(0, 2, 3, 1).reshape(-1, C)

            log_probs, probs = model(img_flat)

            # --------------------------
            # Supervised (teacher) loss (with annealing)
            # --------------------------
            teacher_weight = max(0.2, 1.0 - epoch / (0.7 * EPOCHS))
            loss_task = teacher_weight * task_loss_fn(log_probs, target.view(-1))

            probs_map = probs.reshape(B, H, W, N_REGIMES)

            # --------------------------
            # Regime usage entropy (avoid collapse to single regime)
            # --------------------------
            mean_probs = probs_map.mean(dim=(0, 1, 2))  # (K,)
            entropy_loss = -torch.sum(mean_probs * torch.log(mean_probs + 1e-8))

            # --------------------------
            # Spatial smoothness
            # --------------------------
            loss_spatial = spatial_loss_fn(probs_map)

            # --------------------------
            # Temporal persistence (front continuity)
            # --------------------------
            if B > 1:
                loss_temporal = torch.mean(
                    (probs_map[1:] - probs_map[:-1]) ** 2
                )
            else:
                loss_temporal = torch.tensor(0.0, device=DEVICE)

            # --------------------------
            # Total loss
            # --------------------------
            loss = (
                loss_task
                + loss_spatial
                + TEMPORAL_WEIGHT * loss_temporal
                - 0.05 * entropy_loss
            )

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        logger.info(
            "Epoch %d | Loss = %.4f",
            epoch + 1,
            epoch_loss / len(loader),
        )

    # ==================================================
    # Save checkpoint
    # ==================================================
    path = out_dir / "gating.pth"
    torch.save(model.state_dict(), path)
    logger.info("Saved %s", path)


if __name__ == "__main__":
    main()