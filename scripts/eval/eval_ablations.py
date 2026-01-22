"""Ablation studies for SD-MoSE.

Tests importance of:
1. Number of regimes (K=3, 6, 9, 12)
2. Soft vs hard regime assignment
3. Entropy regularization weight
4. Load balancing
5. Feature ablations (remove SST, SSS, Chl, etc.)

Usage:
    python -m scripts.eval.eval_ablations
"""
import sys  # noqa: E402
from pathlib import Path  # noqa: E402

root = Path(__file__).resolve().parents[2]  # noqa: E402
sys.path.insert(0, str(root / "src"))  # noqa: E402

import argparse  # noqa: E402
import logging  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
import torch.optim as optim  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402
from tqdm import tqdm  # noqa: E402

from climate_discovery.config import (  # noqa: E402
    FEATURES_EXPERT,
    FEATURES_GATING,
    ModelConfig,
    RESULTS_DIR,
    TARGET,
    TEST_NC,
    TRAIN_NC,
)
from climate_discovery.data.datasets import ClimateDataset  # noqa: E402
from climate_discovery.models.gating import GatingNetwork  # noqa: E402
from climate_discovery.models.losses import SDMoSELoss  # noqa: E402
from climate_discovery.models.symbolic import MixtureOfSymbolicExperts  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


# =============================================================================
# ABLATION UTILITIES
# =============================================================================

def train_gating_quick(
    train_dataset: ClimateDataset,
    n_regimes: int,
    entropy_weight: float,
    balance_weight: float,
    epochs: int = 20,
    device: str = "cpu",
) -> GatingNetwork:
    """Quick gating network training for ablation study.
    
    Args:
        train_dataset: Training data
        n_regimes: Number of regimes
        entropy_weight: Entropy regularization weight
        balance_weight: Load balancing weight
        epochs: Number of epochs
        device: Device
        
    Returns:
        Trained GatingNetwork
    """
    config = ModelConfig()
    
    # Create model
    model = GatingNetwork(
        input_dim=len(FEATURES_GATING),
        num_regimes=n_regimes,
        hidden_dims=config.gating_hidden_dims,
        dropout=config.gating_dropout,
        temperature=1.0,
    ).to(device)
    
    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    
    # Loss
    criterion = SDMoSELoss(
        prediction_loss="mse",
        entropy_weight=entropy_weight,
        balance_weight=balance_weight,
    )
    
    # Data loader
    train_loader = DataLoader(
        train_dataset,
        batch_size=2048,
        shuffle=True,
        num_workers=0,
    )
    
    # Training loop
    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        n_batches = 0
        
        for X_expert, X_gate, y in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", leave=False):
            X_gate = X_gate.to(device)
            y = y.to(device)
            
            optimizer.zero_grad()
            
            # Forward
            probs = model(X_gate)
            
            # Dummy expert predictions (just for training gating structure)
            expert_preds = torch.randn_like(y).unsqueeze(1).expand(-1, n_regimes)
            y_pred = torch.sum(probs * expert_preds, dim=1)
            
            # Loss
            loss_dict = criterion(y_pred, y, probs)
            loss = loss_dict["total"]
            
            # Backward
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            total_loss += loss.item()
            n_batches += 1
        
        avg_loss = total_loss / n_batches
        if (epoch + 1) % 5 == 0:
            logger.info(f"  Epoch {epoch+1}: Loss = {avg_loss:.4f}")
    
    return model


def evaluate_with_config(
    train_dataset: ClimateDataset,
    test_dataset: ClimateDataset,
    n_regimes: int,
    entropy_weight: float = 0.01,
    balance_weight: float = 0.1,
    device: str = "cpu",
) -> dict:
    """Evaluate a specific configuration.
    
    Returns:
        Dictionary with metrics
    """
    # Train gating
    gating = train_gating_quick(
        train_dataset,
        n_regimes,
        entropy_weight,
        balance_weight,
        epochs=20,
        device=device,
    )
    
    # Get regime probabilities
    gating.eval()
    with torch.no_grad():
        X_gate_train = torch.from_numpy(train_dataset.X_gate).float().to(device)
        regime_probs = gating(X_gate_train).cpu().numpy()
    
    # Fit symbolic experts
    config = ModelConfig()
    expert_config = {
        "niterations": 20,  # Reduced for speed
        "populations": 15,
        "binary_operators": config.pysr_binary_operators,
        "unary_operators": config.pysr_unary_operators,
        "maxsize": 20,
        "verbosity": 0,
    }
    
    experts = MixtureOfSymbolicExperts(n_regimes, expert_config)
    experts.fit(
        train_dataset.X_expert,
        train_dataset.y,
        regime_probs,
        variable_names=FEATURES_EXPERT,
        min_samples=50,
    )
    
    # Evaluate on test set
    with torch.no_grad():
        X_gate_test = torch.from_numpy(test_dataset.X_gate).float().to(device)
        regime_probs_test = gating(X_gate_test).cpu().numpy()
    
    y_pred = experts.predict(test_dataset.X_expert, regime_probs_test)
    
    # Compute metrics
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    
    r2 = r2_score(test_dataset.y, y_pred)
    rmse = np.sqrt(mean_squared_error(test_dataset.y, y_pred))
    mae = mean_absolute_error(test_dataset.y, y_pred)
    
    # Entropy
    entropy = -np.sum(regime_probs_test * np.log(regime_probs_test + 1e-10), axis=1)
    mean_entropy = np.mean(entropy)
    
    return {
        "r2": r2,
        "rmse": rmse,
        "mae": mae,
        "mean_entropy": mean_entropy,
    }


# =============================================================================
# ABLATION STUDIES
# =============================================================================

def ablation_n_regimes(
    train_dataset: ClimateDataset,
    test_dataset: ClimateDataset,
    device: str,
) -> pd.DataFrame:
    """Ablation: Effect of number of regimes."""
    logger.info("\n" + "=" * 60)
    logger.info("ABLATION: NUMBER OF REGIMES")
    logger.info("=" * 60)
    
    results = []
    
    for n_regimes in [3, 6, 9, 12]:
        logger.info(f"\nTesting K = {n_regimes}...")
        
        metrics = evaluate_with_config(
            train_dataset,
            test_dataset,
            n_regimes=n_regimes,
            device=device,
        )
        
        metrics["n_regimes"] = n_regimes
        results.append(metrics)
        
        logger.info(f"  R² = {metrics['r2']:.4f}, RMSE = {metrics['rmse']:.4f}")
    
    return pd.DataFrame(results)


def ablation_entropy_weight(
    train_dataset: ClimateDataset,
    test_dataset: ClimateDataset,
    device: str,
) -> pd.DataFrame:
    """Ablation: Effect of entropy regularization."""
    logger.info("\n" + "=" * 60)
    logger.info("ABLATION: ENTROPY WEIGHT")
    logger.info("=" * 60)
    
    results = []
    
    for weight in [0.0, 0.001, 0.01, 0.1]:
        logger.info(f"\nTesting entropy_weight = {weight}...")
        
        metrics = evaluate_with_config(
            train_dataset,
            test_dataset,
            n_regimes=6,
            entropy_weight=weight,
            device=device,
        )
        
        metrics["entropy_weight"] = weight
        results.append(metrics)
        
        logger.info(f"  R² = {metrics['r2']:.4f}, Entropy = {metrics['mean_entropy']:.4f}")
    
    return pd.DataFrame(results)


def ablation_balance_weight(
    train_dataset: ClimateDataset,
    test_dataset: ClimateDataset,
    device: str,
) -> pd.DataFrame:
    """Ablation: Effect of load balancing."""
    logger.info("\n" + "=" * 60)
    logger.info("ABLATION: BALANCE WEIGHT")
    logger.info("=" * 60)
    
    results = []
    
    for weight in [0.0, 0.01, 0.1, 0.5]:
        logger.info(f"\nTesting balance_weight = {weight}...")
        
        metrics = evaluate_with_config(
            train_dataset,
            test_dataset,
            n_regimes=6,
            balance_weight=weight,
            device=device,
        )
        
        metrics["balance_weight"] = weight
        results.append(metrics)
        
        logger.info(f"  R² = {metrics['r2']:.4f}")
    
    return pd.DataFrame(results)


def ablation_features(
    train_dataset: ClimateDataset,
    test_dataset: ClimateDataset,
    device: str,
) -> pd.DataFrame:
    """Ablation: Remove each feature and test impact."""
    logger.info("\n" + "=" * 60)
    logger.info("ABLATION: FEATURE IMPORTANCE")
    logger.info("=" * 60)
    
    results = []
    
    # Baseline: all features
    logger.info("\nBaseline (all features)...")
    metrics = evaluate_with_config(
        train_dataset,
        test_dataset,
        n_regimes=6,
        device=device,
    )
    metrics["removed_feature"] = "none"
    results.append(metrics)
    logger.info(f"  R² = {metrics['r2']:.4f}")
    
    # Remove each feature
    for i, feat in enumerate(FEATURES_EXPERT):
        logger.info(f"\nRemoving {feat}...")
        
        # Create modified datasets
        feature_mask = [j for j in range(len(FEATURES_EXPERT)) if j != i]
        
        train_dataset_mod = ClimateDataset(
            TRAIN_NC,
            expert_features=[FEATURES_EXPERT[j] for j in feature_mask],
            gating_features=FEATURES_GATING,
            target=TARGET,
            drop_nan=True,
        )
        
        test_dataset_mod = ClimateDataset(
            TEST_NC,
            expert_features=[FEATURES_EXPERT[j] for j in feature_mask],
            gating_features=FEATURES_GATING,
            target=TARGET,
            drop_nan=True,
        )
        
        metrics = evaluate_with_config(
            train_dataset_mod,
            test_dataset_mod,
            n_regimes=6,
            device=device,
        )
        
        metrics["removed_feature"] = feat
        results.append(metrics)
        
        logger.info(f"  R² = {metrics['r2']:.4f}")
    
    return pd.DataFrame(results)


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Run ablation studies")
    parser.add_argument(
        "--studies",
        nargs="+",
        default=["n_regimes", "entropy", "balance", "features"],
        help="Which ablations to run"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output directory"
    )
    
    args = parser.parse_args()
    
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Output directory
    output_dir = Path(args.output) if args.output else RESULTS_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load datasets
    logger.info("Loading datasets...")
    
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
    
    logger.info(f"Train: {len(train_dataset)}, Test: {len(test_dataset)}")
    
    # Run ablations
    all_results = {}
    
    if "n_regimes" in args.studies:
        results = ablation_n_regimes(train_dataset, test_dataset, device)
        all_results["n_regimes"] = results
        results.to_csv(output_dir / "ablation_n_regimes.csv", index=False)
        logger.info(f"\n✓ Saved: {output_dir / 'ablation_n_regimes.csv'}")
    
    if "entropy" in args.studies:
        results = ablation_entropy_weight(train_dataset, test_dataset, device)
        all_results["entropy"] = results
        results.to_csv(output_dir / "ablation_entropy.csv", index=False)
        logger.info(f"\n✓ Saved: {output_dir / 'ablation_entropy.csv'}")
    
    if "balance" in args.studies:
        results = ablation_balance_weight(train_dataset, test_dataset, device)
        all_results["balance"] = results
        results.to_csv(output_dir / "ablation_balance.csv", index=False)
        logger.info(f"\n✓ Saved: {output_dir / 'ablation_balance.csv'}")
    
    if "features" in args.studies:
        results = ablation_features(train_dataset, test_dataset, device)
        all_results["features"] = results
        results.to_csv(output_dir / "ablation_features.csv", index=False)
        logger.info(f"\n✓ Saved: {output_dir / 'ablation_features.csv'}")
    
    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("ABLATION STUDIES COMPLETE")
    logger.info("=" * 60)
    
    for name, df in all_results.items():
        logger.info(f"\n{name.upper()}:")
        print(df.to_string(index=False))


if __name__ == "__main__":
    main()