"""Discover symbolic laws for each regime using PySR.

After gating network is trained, this script:
1. Loads trained gating weights
2. Assigns regime probabilities to training data
3. Fits symbolic regressor for each regime (weighted by probabilities)
4. Validates discovered equations
5. Saves equations to file

Usage:
    python -m scripts.train.discover_laws
    python -m scripts.train.discover_laws --gating_checkpoint path/to/gating.pth
"""
import sys  # noqa: E402
from pathlib import Path  # noqa: E402

root = Path(__file__).resolve().parents[2]  # noqa: E402
sys.path.insert(0, str(root / "src"))  # noqa: E402

import argparse  # noqa: E402
import logging  # noqa: E402

import numpy as np
import torch

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
from climate_discovery.models.symbolic import MixtureOfSymbolicExperts

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


def load_trained_gating(
    checkpoint_path: Path,
    input_dim: int,
    num_regimes: int,
    device: str = "cpu",
) -> GatingNetwork:
    """Load trained gating network from checkpoint.
    
    Args:
        checkpoint_path: Path to checkpoint file
        input_dim: Input dimension
        num_regimes: Number of regimes
        device: Device to load model on
        
    Returns:
        Loaded GatingNetwork
    """
    logger.info(f"Loading gating network from {checkpoint_path}")
    
    config = ModelConfig()
    
    # Create model
    model = GatingNetwork(
        input_dim=input_dim,
        num_regimes=num_regimes,
        hidden_dims=config.gating_hidden_dims,
        dropout=config.gating_dropout,
        temperature=1.0,
    ).to(device)
    
    # Load weights
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # Handle different checkpoint formats
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model.eval()
    logger.info("✓ Gating network loaded")
    
    return model


@torch.no_grad()
def compute_regime_probabilities(
    model: GatingNetwork,
    dataset: ClimateDataset,
    device: str = "cpu",
    batch_size: int = 4096,
) -> np.ndarray:
    """Compute regime probabilities for entire dataset.
    
    Args:
        model: Trained gating network
        dataset: ClimateDataset
        device: Device
        batch_size: Batch size for inference
        
    Returns:
        Regime probabilities (N, K)
    """
    logger.info("Computing regime probabilities...")
    
    model.eval()
    all_probs = []
    
    # Process in batches
    for i in range(0, len(dataset), batch_size):
        batch_end = min(i + batch_size, len(dataset))
        
        # Get batch
        X_gate_batch = torch.from_numpy(
            dataset.X_gate[i:batch_end]
        ).float().to(device)
        
        # Forward pass
        probs = model(X_gate_batch)
        all_probs.append(probs.cpu().numpy())
    
    regime_probs = np.concatenate(all_probs, axis=0)
    
    # Log regime statistics
    dominant = np.argmax(regime_probs, axis=1)
    logger.info("Regime distribution:")
    for k in range(regime_probs.shape[1]):
        count = np.sum(dominant == k)
        frac = count / len(regime_probs)
        logger.info(f"  Regime {k}: {count} samples ({frac*100:.1f}%)")
    
    return regime_probs


def main():
    parser = argparse.ArgumentParser(
        description="Discover symbolic laws for each regime"
    )
    parser.add_argument(
        "--gating_checkpoint",
        type=str,
        default=None,
        help="Path to trained gating checkpoint"
    )
    parser.add_argument(
        "--niterations",
        type=int,
        default=40,
        help="PySR iterations per regime"
    )
    parser.add_argument(
        "--populations",
        type=int,
        default=31,
        help="PySR populations"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output path for equations"
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Validate on test set"
    )
    
    args = parser.parse_args()
    
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Load config
    config = ModelConfig()
    
    # Checkpoint path
    if args.gating_checkpoint:
        checkpoint_path = Path(args.gating_checkpoint)
    else:
        checkpoint_path = CHECKPOINT_DIR / "gating_best.pth"
    
    if not checkpoint_path.exists():
        logger.error(f"Checkpoint not found: {checkpoint_path}")
        logger.info("Run: python -m scripts.train.train_gating")
        sys.exit(1)
    
    # Load training data
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
    
    logger.info(f"Training samples: {len(train_dataset)}")
    
    # Load trained gating network
    logger.info("=" * 60)
    logger.info("LOADING GATING NETWORK")
    logger.info("=" * 60)
    
    gating_model = load_trained_gating(
        checkpoint_path,
        input_dim=len(FEATURES_GATING),
        num_regimes=config.n_regimes,
        device=device,
    )
    
    # Compute regime probabilities
    logger.info("=" * 60)
    logger.info("COMPUTING REGIME PROBABILITIES")
    logger.info("=" * 60)
    
    regime_probs = compute_regime_probabilities(
        gating_model,
        train_dataset,
        device=device,
    )
    
    # Get features and targets
    X_expert = train_dataset.X_expert
    y = train_dataset.y
    
    # Discover symbolic laws
    logger.info("=" * 60)
    logger.info("DISCOVERING SYMBOLIC LAWS")
    logger.info("=" * 60)
    
    expert_config = {
        "niterations": args.niterations,
        "populations": args.populations,
        "binary_operators": config.pysr_binary_operators,
        "unary_operators": config.pysr_unary_operators,
        "complexity_penalty": config.pysr_complexity_penalty,
        "maxsize": 25,
        "verbosity": 1,
    }
    
    experts = MixtureOfSymbolicExperts(
        num_regimes=config.n_regimes,
        expert_config=expert_config,
    )
    
    experts.fit(
        X_expert,
        y,
        regime_probs,
        variable_names=FEATURES_EXPERT,
        min_samples=100,
    )
    
    # Get discovered equations
    equations = experts.get_all_equations()
    
    logger.info("=" * 60)
    logger.info("DISCOVERED EQUATIONS")
    logger.info("=" * 60)
    
    for k, eq in equations.items():
        logger.info(f"Regime {k}: {eq}")
    
    # Save equations
    output_path = Path(args.output) if args.output else RESULTS_DIR / "equations.txt"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    experts.save_equations(output_path)
    logger.info(f"✓ Equations saved to {output_path}")
    
    # Optional: Validate on test set
    if args.validate:
        logger.info("=" * 60)
        logger.info("VALIDATION ON TEST SET")
        logger.info("=" * 60)
        
        test_dataset = ClimateDataset(
            TEST_NC,
            expert_features=FEATURES_EXPERT,
            gating_features=FEATURES_GATING,
            target=TARGET,
            drop_nan=True,
        )
        
        logger.info(f"Test samples: {len(test_dataset)}")
        
        # Compute regime probs for test set
        test_regime_probs = compute_regime_probabilities(
            gating_model,
            test_dataset,
            device=device,
        )
        
        # Validate
        validations = experts.validate_all(
            test_dataset.X_expert,
            test_dataset.y,
            test_regime_probs,
        )
        
        logger.info("\nValidation Results:")
        for val in validations:
            logger.info(
                f"Regime {val['regime_id']}: "
                f"R²={val['r2']:.4f}, MSE={val['mse']:.4f}, "
                f"Invalid={val['frac_invalid']*100:.1f}%"
            )
    
    logger.info("=" * 60)
    logger.info("SYMBOLIC DISCOVERY COMPLETE")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()