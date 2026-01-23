"""Train gating network for SD-MoSE.

Training strategy:
1. Optional K-means warm-start (supervised initialization)
2. Train gating network with:
   - Prediction loss (with fixed random expert assignments initially)
   - Entropy regularization (confident regime assignments)
   - Load balancing (prevent regime collapse)
3. Validation on held-out data
4. Save checkpoint

Usage:
    python -m scripts.train.train_gating
    python -m scripts.train.train_gating --seed 0 --epochs 100
"""
import sys  # noqa: E402
from pathlib import Path  # noqa: E402

root = Path(__file__).resolve().parents[2]  # noqa: E402
sys.path.insert(0, str(root / "src"))  # noqa: E402

import argparse  # noqa: E402
import logging  # noqa: E402

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.cluster import KMeans
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from climate_discovery.config import (
    CHECKPOINT_DIR,
    FEATURES_EXPERT,
    FEATURES_GATING,
    ModelConfig,
    TARGET,
    TRAIN_NC,
)
from climate_discovery.data.datasets import ClimateDataset
from climate_discovery.models.gating import GatingNetwork
from climate_discovery.models.losses import SDMoSELoss, EarlyStopping

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


def kmeans_warm_start(
    dataset: ClimateDataset,
    n_regimes: int,
    n_samples: int = 50000,
    random_state: int = 42,
) -> torch.Tensor:
    """Initialize gating network with K-means clustering.
    
    Args:
        dataset: ClimateDataset instance
        n_regimes: Number of regimes
        n_samples: Subsample size for clustering
        random_state: Random seed
        
    Returns:
        Soft labels (N, K) from K-means cluster distances
    """
    logger.info("Computing K-means warm-start labels...")
    
    # Get gating features
    X_gate = dataset.X_gate
    n_total = len(X_gate)
    
    # Subsample for efficiency
    if n_total > n_samples:
        rng = np.random.default_rng(random_state)
        indices = rng.choice(n_total, n_samples, replace=False)
        X_subsample = X_gate[indices]
    else:
        X_subsample = X_gate
    
    # Fit K-means
    kmeans = KMeans(
        n_clusters=n_regimes,
        n_init=10,
        random_state=random_state,
        verbose=0
    )
    kmeans.fit(X_subsample)
    
    # Get cluster assignments for full dataset
    logger.info("Assigning clusters to full dataset...")
    batch_size = 10000
    labels = []
    for i in range(0, n_total, batch_size):
        batch = X_gate[i:i+batch_size]
        labels.append(kmeans.predict(batch))
    labels = np.concatenate(labels)
    
    # Convert hard labels to soft probabilities via distance
    distances = kmeans.transform(X_gate)  # (N, K)
    # Softmax over negative distances (closer = higher prob)
    soft_labels = torch.softmax(torch.from_numpy(-distances).float(), dim=1)
    
    logger.info("K-means initialization complete. Regime distribution:")
    for k in range(n_regimes):
        count = np.sum(labels == k)
        logger.info(f"  Regime {k}: {count} samples ({count/n_total*100:.1f}%)")
    
    return soft_labels


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: optim.Optimizer,
    criterion: SDMoSELoss,
    device: str,
    use_teacher: bool = False,
    teacher_labels: torch.Tensor = None,
    teacher_weight: float = 0.5,
) -> dict:
    """Train for one epoch.
    
    Args:
        model: Gating network
        loader: Training data loader
        optimizer: Optimizer
        criterion: Loss function
        device: Device
        use_teacher: Whether to use K-means teacher labels
        teacher_labels: Optional teacher labels
        teacher_weight: Weight for teacher loss
        
    Returns:
        Dictionary with average losses
    """
    model.train()
    
    total_loss = 0.0
    total_pred_loss = 0.0
    total_entropy_loss = 0.0
    total_balance_loss = 0.0
    n_batches = 0
    
    for batch_idx, (X_expert, X_gate, y) in enumerate(tqdm(loader, desc="Training")):
        X_gate = X_gate.to(device)
        y = y.to(device)
        
        optimizer.zero_grad()
        
        # Forward pass
        probs = model(X_gate)  # (N, K)
        
        # For initial training, use random expert predictions
        # (In real SD-MoSE, these come from symbolic experts)
        # Here we just need to train the gating network structure
        expert_preds = torch.randn_like(y).unsqueeze(1).expand(-1, probs.shape[1])
        y_pred = torch.sum(probs * expert_preds, dim=1)
        
        # Compute loss
        loss_dict = criterion(y_pred, y, probs)
        loss = loss_dict["total"]
        
        # Optional: Add K-means teacher loss
        if use_teacher and teacher_labels is not None:
            batch_start = batch_idx * loader.batch_size
            batch_end = batch_start + len(y)
            teacher_batch = teacher_labels[batch_start:batch_end].to(device)
            
            # KL divergence between predicted probs and teacher
            teacher_loss = nn.KLDivLoss(reduction="batchmean")(
                torch.log(probs + 1e-10),
                teacher_batch
            )
            loss = loss + teacher_weight * teacher_loss
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        # Accumulate losses
        total_loss += loss.item()
        total_pred_loss += loss_dict["prediction"].item()
        total_entropy_loss += loss_dict["entropy"]
        total_balance_loss += loss_dict["balance"]
        n_batches += 1
    
    return {
        "loss": total_loss / n_batches,
        "pred_loss": total_pred_loss / n_batches,
        "entropy": total_entropy_loss / n_batches,
        "balance": total_balance_loss / n_batches,
    }


@torch.no_grad()
def validate(
    model: nn.Module,
    loader: DataLoader,
    criterion: SDMoSELoss,
    device: str,
) -> dict:
    """Validate model.
    
    Returns:
        Dictionary with validation metrics
    """
    model.eval()
    
    all_probs = []
    total_entropy = 0.0
    n_samples = 0
    
    for X_expert, X_gate, y in loader:
        X_gate = X_gate.to(device)
        
        probs = model(X_gate)
        all_probs.append(probs.cpu())
        
        # Compute entropy
        entropy = -torch.sum(probs * torch.log(probs + 1e-10), dim=1)
        total_entropy += torch.sum(entropy).item()
        n_samples += len(probs)
    
    all_probs = torch.cat(all_probs, dim=0)
    
    # Regime statistics
    dominant = torch.argmax(all_probs, dim=1)
    regime_counts = torch.bincount(dominant, minlength=all_probs.shape[1])
    
    # Balance metric
    balance_cv = torch.std(regime_counts.float()) / (torch.mean(regime_counts.float()) + 1e-10)
    
    return {
        "mean_entropy": total_entropy / n_samples,
        "regime_usage": regime_counts.numpy(),
        "balance_cv": balance_cv.item(),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Train gating network for SD-MoSE"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed"
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to config file (optional)"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=2048,
        help="Batch size"
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help="Learning rate"
    )
    parser.add_argument(
        "--use_kmeans_init",
        action="store_true",
        help="Use K-means warm start"
    )
    parser.add_argument(
        "--teacher_weight",
        type=float,
        default=0.5,
        help="Weight for K-means teacher loss"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output checkpoint path"
    )
    
    args = parser.parse_args()
    
    # Set random seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Load config
    config = ModelConfig()
    
    # Load dataset
    logger.info("Loading training data...")
    full_dataset = ClimateDataset(
        TRAIN_NC,
        expert_features=FEATURES_EXPERT,
        gating_features=FEATURES_GATING,
        target=TARGET,
        drop_nan=True,
    )
    
    logger.info(f"Dataset size: {len(full_dataset)} valid samples")
    
    # Train/val split (85/15)
    train_size = int(0.85 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(
        full_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(args.seed)
    )
    
    logger.info(f"Train: {train_size}, Val: {val_size}")
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        drop_last=False,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )
    
    # K-means warm start (optional)
    teacher_labels = None
    if args.use_kmeans_init:
        teacher_labels = kmeans_warm_start(
            full_dataset,
            n_regimes=config.n_regimes,
            random_state=args.seed
        )
    
    # Create model
    logger.info("Initializing gating network...")
    model = GatingNetwork(
        input_dim=len(FEATURES_GATING),
        num_regimes=config.n_regimes,
        hidden_dims=config.gating_hidden_dims,
        dropout=config.gating_dropout,
        temperature=1.0,
    ).to(device)
    
    logger.info(f"Model: {sum(p.numel() for p in model.parameters())} parameters")
    
    # Optimizer
    optimizer = optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=config.gating_weight_decay
    )
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=10
    )
    
    # Loss function
    criterion = SDMoSELoss(
        prediction_loss="mse",
        entropy_weight=config.entropy_weight,
        balance_weight=0.1,
    )
    
    # Early stopping
    early_stop = EarlyStopping(patience=20, min_delta=1e-4)
    
    # Training loop
    logger.info("=" * 60)
    logger.info("Starting training...")
    logger.info("=" * 60)
    
    best_val_entropy = float('inf')
    
    for epoch in range(args.epochs):
        logger.info(f"\nEpoch {epoch + 1}/{args.epochs}")
        
        # Teacher annealing (reduce K-means influence over time)
        teacher_weight = args.teacher_weight * max(0.1, 1.0 - epoch / (args.epochs * 0.7))
        
        # Train
        train_metrics = train_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            device,
            use_teacher=args.use_kmeans_init,
            teacher_labels=teacher_labels,
            teacher_weight=teacher_weight,
        )
        
        # Validate
        val_metrics = validate(model, val_loader, criterion, device)
        
        # Log metrics
        logger.info(
            f"Train Loss: {train_metrics['loss']:.4f} | "
            f"Entropy: {train_metrics['entropy']:.4f} | "
            f"Balance: {train_metrics['balance']:.4f}"
        )
        logger.info(
            f"Val Entropy: {val_metrics['mean_entropy']:.4f} | "
            f"Balance CV: {val_metrics['balance_cv']:.4f}"
        )
        logger.info(f"Regime usage: {val_metrics['regime_usage']}")
        
        # Learning rate scheduling
        scheduler.step(val_metrics['mean_entropy'])
        
        # Save best model
        if val_metrics['mean_entropy'] < best_val_entropy:
            best_val_entropy = val_metrics['mean_entropy']
            
            output_path = Path(args.output) if args.output else CHECKPOINT_DIR / "gating_best.pth"
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_entropy': val_metrics['mean_entropy'],
                'regime_usage': val_metrics['regime_usage'],
                'config': config.__dict__,
            }, output_path)
            
            logger.info(f"✓ Saved best model: {output_path}")
        
        # Early stopping
        if early_stop(val_metrics['mean_entropy']):
            logger.info(f"Early stopping at epoch {epoch + 1}")
            break
    
    # Final save
    final_path = CHECKPOINT_DIR / "gating_final.pth"
    torch.save(model.state_dict(), final_path)
    logger.info(f"✓ Saved final model: {final_path}")
    
    logger.info("=" * 60)
    logger.info("Training complete!")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()