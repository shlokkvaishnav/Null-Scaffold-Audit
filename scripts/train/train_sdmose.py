"""SD-MoSE alternating optimization: Train gating ↔ Refit symbolic experts.

Training loop:
1. Initialize: Train gating with K-means + random experts
2. For T iterations:
   a. Fix gating → Discover symbolic experts with PySR
   b. Fix experts → Train gating to minimize mixture loss
3. Save final model

This is the complete SD-MoSE training pipeline.

Usage:
    python -m scripts.train.train_sdmose
    python -m scripts.train.train_sdmose --iterations 5 --gating_epochs 50
"""
import sys  # noqa: E402
from pathlib import Path  # noqa: E402

root = Path(__file__).resolve().parents[2]  # noqa: E402
sys.path.insert(0, str(root / "src"))  # noqa: E402

import argparse  # noqa: E402
import logging  # noqa: E402

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from climate_discovery.config import (
    CHECKPOINT_DIR,
    FEATURES_EXPERT,
    FEATURES_GATING,
    ModelConfig,
    RESULTS_DIR,
    TARGET,
    TEST_NC,
    TRAIN_NC,
)
from climate_discovery.data.datasets import ClimateDataset
from climate_discovery.models.gating import GatingNetwork
from climate_discovery.models.losses import SDMoSELoss
from climate_discovery.models.mixture import SDMoSE, evaluate_mixture
from climate_discovery.models.symbolic import MixtureOfSymbolicExperts

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


def train_gating_step(
    model: SDMoSE,
    train_loader: DataLoader,
    optimizer: optim.Optimizer,
    criterion: SDMoSELoss,
    expert_predictions: np.ndarray,
    device: str,
    epochs: int = 10,
) -> float:
    """Train gating network with fixed expert predictions.
    
    Args:
        model: SDMoSE model
        train_loader: Training data loader
        optimizer: Optimizer
        criterion: Loss function
        expert_predictions: Pre-computed expert outputs (N, K)
        device: Device
        epochs: Number of epochs to train
        
    Returns:
        Final training loss
    """
    model.train()
    expert_preds_tensor = torch.from_numpy(expert_predictions).float()
    
    for epoch in range(epochs):
        total_loss = 0.0
        n_batches = 0
        
        pbar = tqdm(train_loader, desc=f"Gating Epoch {epoch+1}/{epochs}")
        
        for batch_idx, (X_expert, X_gate, y) in enumerate(pbar):
            X_gate = X_gate.to(device)
            y = y.to(device)
            
            # Get expert predictions for this batch
            batch_start = batch_idx * train_loader.batch_size
            batch_end = batch_start + len(y)
            expert_batch = expert_preds_tensor[batch_start:batch_end].to(device)
            
            optimizer.zero_grad()
            
            # Forward: mixture prediction
            y_pred, probs = model.forward_mixture(X_gate, expert_batch)
            
            # Compute loss
            loss_dict = criterion(y_pred, y, probs)
            loss = loss_dict["total"]
            
            # Backward
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            total_loss += loss.item()
            n_batches += 1
            
            pbar.set_postfix({"loss": loss.item()})
        
        avg_loss = total_loss / n_batches
        logger.info(f"  Gating Epoch {epoch+1}: Loss = {avg_loss:.4f}")
    
    return avg_loss


def fit_symbolic_experts(
    X_expert: np.ndarray,
    y: np.ndarray,
    regime_probs: np.ndarray,
    config: ModelConfig,
    variable_names: list,
) -> MixtureOfSymbolicExperts:
    """Fit symbolic experts for each regime.
    
    Args:
        X_expert: Expert features (N, D)
        y: Targets (N,)
        regime_probs: Regime probabilities (N, K)
        config: Model configuration
        variable_names: Feature names
        
    Returns:
        Fitted MixtureOfSymbolicExperts
    """
    logger.info("Fitting symbolic experts with PySR...")
    
    expert_config = {
        "niterations": config.pysr_niterations,
        "populations": config.pysr_populations,
        "binary_operators": config.pysr_binary_operators,
        "unary_operators": config.pysr_unary_operators,
        "complexity_penalty": config.pysr_complexity_penalty,
        "maxsize": 25,
        "verbosity": 0,  # Quiet mode (set to 1 for debugging)
    }
    
    experts = MixtureOfSymbolicExperts(
        num_regimes=config.n_regimes,
        expert_config=expert_config,
    )
    
    experts.fit(
        X_expert,
        y,
        regime_probs,
        variable_names=variable_names,
        min_samples=100,
    )
    
    # Log discovered equations
    equations = experts.get_all_equations()
    logger.info("Discovered equations:")
    for k, eq in equations.items():
        logger.info(f"  Regime {k}: {eq}")
    
    return experts


def compute_expert_predictions(
    experts: MixtureOfSymbolicExperts,
    X_expert: np.ndarray,
) -> np.ndarray:
    """Compute predictions from all experts.
    
    Args:
        experts: MixtureOfSymbolicExperts
        X_expert: Expert features (N, D)
        
    Returns:
        Expert predictions (N, K)
    """
    n_samples = len(X_expert)
    k = experts.num_regimes
    expert_preds = np.zeros((n_samples, k), dtype=np.float32)
    
    for k_idx, expert in enumerate(experts.experts):
        try:
            expert_preds[:, k_idx] = expert.predict(X_expert)
        except Exception as e:
            logger.warning(f"Expert {k_idx} prediction failed: {e}")
            expert_preds[:, k_idx] = 0.0
    
    # Replace NaN/Inf with zero
    expert_preds = np.nan_to_num(expert_preds, nan=0.0, posinf=0.0, neginf=0.0)
    
    return expert_preds


def main():
    parser = argparse.ArgumentParser(
        description="SD-MoSE alternating optimization"
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=5,
        help="Number of alternating optimization iterations"
    )
    parser.add_argument(
        "--gating_epochs",
        type=int,
        default=20,
        help="Gating epochs per iteration"
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
        "--init_checkpoint",
        type=str,
        default=None,
        help="Initial gating checkpoint (optional)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output directory for checkpoints"
    )
    
    args = parser.parse_args()
    
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Load config
    config = ModelConfig()
    
    # Output directory
    output_dir = Path(args.output) if args.output else CHECKPOINT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # =========================================================================
    # LOAD DATA
    # =========================================================================
    logger.info("=" * 60)
    logger.info("LOADING DATA")
    logger.info("=" * 60)
    
    train_dataset = ClimateDataset(
        TRAIN_NC,
        expert_features=FEATURES_EXPERT,
        gating_features=FEATURES_GATING,
        target=TARGET,
        drop_nan=True,
    )
    
    test_dataset = ClimateDataset(
        TEST_NC,
        expert_features=FEATURES_EXPERT,
        gating_features=FEATURES_GATING,
        target=TARGET,
        drop_nan=True,
    )
    
    logger.info(f"Train: {len(train_dataset)} samples")
    logger.info(f"Test: {len(test_dataset)} samples")
    
    # Create data loader
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
    )
    
    # =========================================================================
    # INITIALIZE MODEL
    # =========================================================================
    logger.info("=" * 60)
    logger.info("INITIALIZING MODEL")
    logger.info("=" * 60)
    
    # Create gating network
    gating = GatingNetwork(
        input_dim=len(FEATURES_GATING),
        num_regimes=config.n_regimes,
        hidden_dims=config.gating_hidden_dims,
        dropout=config.gating_dropout,
        temperature=1.0,
    ).to(device)
    
    # Load initial checkpoint if provided
    if args.init_checkpoint:
        checkpoint = torch.load(args.init_checkpoint, map_location=device)
        if 'model_state_dict' in checkpoint:
            gating.load_state_dict(checkpoint['model_state_dict'])
        else:
            gating.load_state_dict(checkpoint)
        logger.info(f"✓ Loaded initial checkpoint: {args.init_checkpoint}")
    
    # Create SD-MoSE model
    model = SDMoSE(
        gating_network=gating,
        num_regimes=config.n_regimes,
        expert_features=FEATURES_EXPERT,
        device=device,
    )
    
    # Optimizer and loss
    optimizer = optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=config.gating_weight_decay
    )
    
    criterion = SDMoSELoss(
        prediction_loss="mse",
        entropy_weight=config.entropy_weight,
        balance_weight=0.1,
    )
    
    # =========================================================================
    # ALTERNATING OPTIMIZATION
    # =========================================================================
    logger.info("=" * 60)
    logger.info(f"STARTING ALTERNATING OPTIMIZATION ({args.iterations} iterations)")
    logger.info("=" * 60)
    
    for iteration in range(args.iterations):
        logger.info("\n" + "=" * 60)
        logger.info(f"ITERATION {iteration + 1}/{args.iterations}")
        logger.info("=" * 60)
        
        # ---------------------------------------------------------------------
        # STEP 1: Compute regime probabilities with current gating
        # ---------------------------------------------------------------------
        logger.info("\n[1/3] Computing regime probabilities...")
        
        model.eval()
        with torch.no_grad():
            X_gate_tensor = torch.from_numpy(train_dataset.X_gate).float().to(device)
            regime_probs = model.get_regime_probs(X_gate_tensor).cpu().numpy()
        
        # Log regime statistics
        dominant = np.argmax(regime_probs, axis=1)
        logger.info("Regime distribution:")
        for k in range(config.n_regimes):
            count = np.sum(dominant == k)
            logger.info(f"  Regime {k}: {count} ({count/len(regime_probs)*100:.1f}%)")
        
        # ---------------------------------------------------------------------
        # STEP 2: Fit symbolic experts with current regime assignments
        # ---------------------------------------------------------------------
        logger.info("\n[2/3] Fitting symbolic experts...")
        
        symbolic_experts = fit_symbolic_experts(
            train_dataset.X_expert,
            train_dataset.y,
            regime_probs,
            config,
            FEATURES_EXPERT,
        )
        
        # Compute expert predictions
        expert_preds = compute_expert_predictions(
            symbolic_experts,
            train_dataset.X_expert,
        )
        
        # ---------------------------------------------------------------------
        # STEP 3: Train gating with fixed experts
        # ---------------------------------------------------------------------
        logger.info("\n[3/3] Training gating network...")
        
        train_gating_step(
            model,
            train_loader,
            optimizer,
            criterion,
            expert_preds,
            device,
            epochs=args.gating_epochs,
        )
        
        # ---------------------------------------------------------------------
        # EVALUATE ON TEST SET
        # ---------------------------------------------------------------------
        logger.info("\nEvaluating on test set...")
        
        model.attach_symbolic_experts(symbolic_experts)
        
        test_metrics = evaluate_mixture(
            model,
            test_dataset.X_gate,
            test_dataset.X_expert,
            test_dataset.y,
            symbolic_experts,
        )
        
        logger.info(
            f"Test Performance: "
            f"R²={test_metrics['r2']:.4f}, "
            f"RMSE={test_metrics['rmse']:.4f}, "
            f"MAE={test_metrics['mae']:.4f}"
        )
        
        # Save checkpoint after each iteration
        checkpoint_path = output_dir / f"sdmose_iter{iteration+1}.pth"
        model.save(checkpoint_path)
        
        # Save equations
        eq_path = RESULTS_DIR / f"equations_iter{iteration+1}.txt"
        eq_path.parent.mkdir(parents=True, exist_ok=True)
        symbolic_experts.save_equations(eq_path)
        
        logger.info(f"✓ Checkpoint saved: {checkpoint_path}")
        logger.info(f"✓ Equations saved: {eq_path}")
    
    # =========================================================================
    # FINAL SAVE
    # =========================================================================
    logger.info("=" * 60)
    logger.info("TRAINING COMPLETE")
    logger.info("=" * 60)
    
    final_path = output_dir / "sdmose_final.pth"
    model.save(final_path)
    logger.info(f"✓ Final model saved: {final_path}")
    
    # Final test evaluation
    logger.info("\nFinal Test Performance:")
    test_metrics = evaluate_mixture(
        model,
        test_dataset.X_gate,
        test_dataset.X_expert,
        test_dataset.y,
        symbolic_experts,
    )
    
    logger.info(f"  R² = {test_metrics['r2']:.4f}")
    logger.info(f"  RMSE = {test_metrics['rmse']:.4f}")
    logger.info(f"  MAE = {test_metrics['mae']:.4f}")
    logger.info(f"  Mean Entropy = {test_metrics['mean_entropy']:.4f}")


if __name__ == "__main__":
    main()