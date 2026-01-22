"""Train soft gating network (K-means teacher + spatial smoothness)."""

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

FEATURES = ["sst", "sss", "log_chl"]
SPATIAL_WEIGHT = 1.0
BATCH_SIZE = 64
EPOCHS = 5
LR = 1e-3
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger(__name__)


def main():
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    logger.info("Loading spatial dataset...")
    dataset = ClimateSpatialDataset(str(FUSED_NC), FEATURES, mode="train")

    full_data = dataset.data.permute(0, 2, 3, 1)
    mask = dataset.mask
    mask_expanded = mask.unsqueeze(0).expand(full_data.shape[0], -1, -1)
    X_all = full_data[mask_expanded].numpy()

    n_samples = min(100_000, len(X_all))
    rng = np.random.default_rng(42)
    idx = rng.choice(len(X_all), n_samples, replace=False)
    kmeans = KMeans(n_clusters=N_REGIMES, n_init=5, random_state=42)
    kmeans.fit(X_all[idx])

    chunk_size = 500_000
    labels_flat = []
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

    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    model = GatingNetwork(input_dim=len(FEATURES), num_regimes=N_REGIMES).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    spatial_loss_fn = RegimeConsistencyLoss(weight=SPATIAL_WEIGHT)
    task_loss_fn = nn.NLLLoss(ignore_index=-1)

    model.train()
    for epoch in range(EPOCHS):
        for batch in tqdm(loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
            img = batch["image"].to(DEVICE)
            target = batch["target"].to(DEVICE)
            optimizer.zero_grad()
            B, C, H, W = img.shape
            img_flat = img.permute(0, 2, 3, 1).reshape(-1, C)
            log_probs, probs = model(img_flat)
            loss_task = task_loss_fn(log_probs, target.view(-1))
            probs_map = probs.reshape(B, H, W, N_REGIMES)
            loss_spatial = spatial_loss_fn(probs_map)
            (loss_task + loss_spatial).backward()
            optimizer.step()

    path = CHECKPOINT_DIR / "gating_warmstart.pth"
    torch.save(model.state_dict(), path)
    logger.info("Saved %s", path)


if __name__ == "__main__":
    main()
