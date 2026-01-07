import sys
from pathlib import Path

# Add 'src' to Python path so we can import our modules
sys.path.append(str(Path(__file__).parent.parent / "src"))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.cluster import KMeans
from tqdm import tqdm
import logging

# Import your new modules
from climate_discovery.data.dataset import ClimateDataset
from climate_discovery.models.gating import GatingNetwork

# Setup Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger(__name__)

# --- CONFIG ---
DATA_PATH = Path("data/03_processed/climate_fused_dataset.nc")
CHECKPOINT_DIR = Path("checkpoints")
FEATURES = ['sst', 'sss', 'sin_month', 'cos_month', 'log_chl']
N_REGIMES = 6
BATCH_SIZE = 1024
EPOCHS = 20  # Fast warm-up
LR = 0.001
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def main():
    CHECKPOINT_DIR.mkdir(exist_ok=True)
    
    # 1. Load Data
    logger.info("1. Loading Dataset...")
    dataset = ClimateDataset(DATA_PATH, FEATURES, mode='train')
    
    # 2. Generate K-Means Labels (The "Teacher")
    logger.info(f"2. Running K-Means (K={N_REGIMES}) to create target labels...")
    # We use the normalized features stored in the dataset
    X_numpy = dataset.X 
    kmeans = KMeans(n_clusters=N_REGIMES, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X_numpy)
    
    # Attach labels to dataset
    dataset.set_kmeans_labels(labels)
    logger.info("   ✅ K-Means complete. Labels attached.")

    # 3. Setup DataLoader & Model
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    model = GatingNetwork(input_dim=len(FEATURES), num_regimes=N_REGIMES).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    criterion = nn.NLLLoss() # Negative Log Likelihood (since model outputs log_softmax)

    # 4. Training Loop
    logger.info(f"3. Training Gating Network on {DEVICE}...")
    model.train()
    
    for epoch in range(EPOCHS):
        total_loss = 0
        correct = 0
        total = 0
        
        progress = tqdm(loader, desc=f"Epoch {epoch+1}/{EPOCHS}", leave=False)
        
        for batch in progress:
            # Move to GPU/CPU
            x = batch['features'].to(DEVICE)
            y_target = batch['regime'].to(DEVICE) # The K-Means label
            
            # Forward
            optimizer.zero_grad()
            log_probs, probs = model(x)
            
            # Loss: Distance between Network prediction and K-Means label
            loss = criterion(log_probs, y_target)
            
            # Backward
            loss.backward()
            optimizer.step()
            
            # Stats
            total_loss += loss.item()
            pred = probs.argmax(dim=1)
            correct += (pred == y_target).sum().item()
            total += y_target.size(0)
            
            progress.set_postfix(loss=loss.item())
            
        avg_loss = total_loss / len(loader)
        acc = 100 * correct / total
        logger.info(f"   Epoch {epoch+1}: Loss = {avg_loss:.4f} | Accuracy = {acc:.2f}%")

    # 5. Save Checkpoint
    save_path = CHECKPOINT_DIR / "gating_warmstart.pth"
    torch.save(model.state_dict(), save_path)
    logger.info(f"✅ Model saved to {save_path}")
    logger.info("   The Gating Network has now learned the 'Shape' of the ocean.")

if __name__ == "__main__":
    main()