import sys
from pathlib import Path
import logging
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.cluster import KMeans
from tqdm import tqdm
import numpy as np

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from climate_discovery.data.dataset import ClimateSpatialDataset
from climate_discovery.models.gating import GatingNetwork
from climate_discovery.models.loss import RegimeConsistencyLoss

# --- CONFIG ---
DATA_PATH = Path("data/03_processed/climate_fused_dataset.nc")
CHECKPOINT_DIR = Path("checkpoints")
FEATURES = ['sst', 'sss', 'sin_month', 'cos_month', 'log_chl']
N_REGIMES = 6
BATCH_SIZE = 64       # INCREASED from 16 to 64 for speed
EPOCHS = 5            # REDUCED from 10 to 5 (It converges fast anyway)
LR = 1e-3
SPATIAL_WEIGHT = 0.01 
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Setup Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger(__name__)

def main():
    CHECKPOINT_DIR.mkdir(exist_ok=True)
    
    logger.info("1. Loading Climate Spatial Dataset...")
    dataset = ClimateSpatialDataset(DATA_PATH, FEATURES, mode='train')
    
    # --- OPTIMIZATION 1: Subsample K-Means ---
    logger.info("2. Training Teacher (K-Means) on subset...")
    
    full_data = dataset.data.permute(0, 2, 3, 1) # (T, H, W, C)
    mask = dataset.mask # (H, W)
    mask_expanded = mask.unsqueeze(0).expand(full_data.shape[0], -1, -1)
    
    # Extract all valid pixels
    X_all = full_data[mask_expanded].numpy()
    
    # Subsample to max 100k points for K-Means fitting (Huge speedup)
    n_samples = min(100000, len(X_all))
    indices = np.random.choice(len(X_all), n_samples, replace=False)
    X_train_subset = X_all[indices]
    
    kmeans = KMeans(n_clusters=N_REGIMES, n_init=5, random_state=42) # n_init reduced to 5
    kmeans.fit(X_train_subset)
    
    # Predict labels for everyone (Fast) using the trained centers
    # We do this in chunks to avoid memory crash
    logger.info("   Predicting full dataset labels...")
    labels_flat = []
    chunk_size = 500000
    for i in range(0, len(X_all), chunk_size):
        chunk = X_all[i:i+chunk_size]
        labels_flat.append(kmeans.predict(chunk))
    labels_flat = np.concatenate(labels_flat)
    
    # Map back to grid
    teacher_targets = torch.full((full_data.shape[0], full_data.shape[1], full_data.shape[2]), -1, dtype=torch.long)
    teacher_targets[mask_expanded] = torch.from_numpy(labels_flat).to(torch.long)
    dataset.set_teacher_targets(teacher_targets)
    logger.info("✅ Teacher targets generated.")

    # --- OPTIMIZATION 2: Parallel Data Loading ---
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0) # Windows likes num_workers=0
    
    model = GatingNetwork(input_dim=len(FEATURES), num_regimes=N_REGIMES).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    spatial_loss_fn = RegimeConsistencyLoss(weight=SPATIAL_WEIGHT)
    task_loss_fn = nn.NLLLoss(ignore_index=-1)
    
    logger.info(f"3. Training Student (Gating Net) on {DEVICE}...")
    model.train()
    
    for epoch in range(EPOCHS):
        total_loss = 0
        
        progress = tqdm(loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        
        for batch in progress:
            img = batch['image'].to(DEVICE)
            target = batch['target'].to(DEVICE)
            
            optimizer.zero_grad()
            
            B, C, H, W = img.shape
            img_flat = img.permute(0, 2, 3, 1).reshape(-1, C)
            log_probs_flat, probs_flat = model(img_flat)
            
            target_flat = target.view(-1)
            loss_task = task_loss_fn(log_probs_flat, target_flat)
            
            probs_map = probs_flat.reshape(B, H, W, N_REGIMES)
            loss_spatial = spatial_loss_fn(probs_map)
            
            loss = loss_task + loss_spatial
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            progress.set_postfix({"Task": loss_task.item()})
            
    save_path = CHECKPOINT_DIR / "gating_warmstart.pth"
    torch.save(model.state_dict(), save_path)
    logger.info(f"✅ Model saved to {save_path}")

if __name__ == "__main__":
    main()