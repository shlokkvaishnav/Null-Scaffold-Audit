"""Train ensemble of gating networks with different random seeds.

This script trains multiple gating networks independently to assess
regime robustness. High ensemble agreement → robust regimes.

Usage:
    python -m scripts.train.run_gating_ensemble
    python -m scripts.train.run_gating_ensemble --seeds 0 1 2 3 4 --parallel
"""
import sys  # noqa: E402
from pathlib import Path  # noqa: E402

root = Path(__file__).resolve().parents[2]  # noqa: E402
sys.path.insert(0, str(root / "src"))  # noqa: E402

import argparse  # noqa: E402
import logging  # noqa: E402
import multiprocessing as mp
import subprocess

from climate_discovery.config import CHECKPOINT_DIR, ModelConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


def train_single_member(
    seed: int,
    output_dir: Path,
    epochs: int,
    batch_size: int,
    use_kmeans: bool,
) -> bool:
    """Train a single ensemble member.
    
    Args:
        seed: Random seed
        output_dir: Output directory for this member
        epochs: Number of epochs
        batch_size: Batch size
        use_kmeans: Whether to use K-means initialization
        
    Returns:
        True if successful, False otherwise
    """
    logger.info(f"Training ensemble member with seed={seed}")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Build command
    cmd = [
        sys.executable,
        "-m",
        "scripts.train.train_gating",
        "--seed", str(seed),
        "--epochs", str(epochs),
        "--batch_size", str(batch_size),
        "--output", str(output_dir / "gating.pth"),
    ]
    
    if use_kmeans:
        cmd.append("--use_kmeans_init")
    
    # Run training
    try:
        subprocess.run(
            cmd,
            check=True,
            capture_output=True,
            text=True,
            env={**subprocess.os.environ, "PYTHONHASHSEED": str(seed)}
        )
        
        logger.info(f"✓ Seed {seed} complete")
        return True
        
    except subprocess.CalledProcessError as e:
        logger.error(f"❌ Seed {seed} failed: {e}")
        logger.error(f"stdout: {e.stdout}")
        logger.error(f"stderr: {e.stderr}")
        return False


def train_parallel(
    seeds: list,
    output_root: Path,
    epochs: int,
    batch_size: int,
    use_kmeans: bool,
    max_workers: int = None,
) -> dict:
    """Train ensemble members in parallel.
    
    Args:
        seeds: List of random seeds
        output_root: Root directory for ensemble checkpoints
        epochs: Number of epochs per member
        batch_size: Batch size
        use_kmeans: Whether to use K-means initialization
        max_workers: Maximum parallel workers (default: num CPUs)
        
    Returns:
        Dictionary mapping seed to success status
    """
    if max_workers is None:
        max_workers = min(len(seeds), mp.cpu_count())
    
    logger.info(f"Training {len(seeds)} ensemble members with {max_workers} workers")
    
    # Create args for each member
    args_list = [
        (seed, output_root / f"seed_{seed}", epochs, batch_size, use_kmeans)
        for seed in seeds
    ]
    
    # Train in parallel
    with mp.Pool(processes=max_workers) as pool:
        results = pool.starmap(train_single_member, args_list)
    
    # Map results
    status = dict(zip(seeds, results))
    
    # Summary
    n_success = sum(results)
    n_failed = len(results) - n_success
    
    logger.info("Ensemble training complete:")
    logger.info(f"  Success: {n_success}/{len(seeds)}")
    logger.info(f"  Failed: {n_failed}/{len(seeds)}")
    
    return status


def analyze_ensemble_agreement(
    ensemble_dir: Path,
    seeds: list,
) -> None:
    """Analyze regime agreement across ensemble members.
    
    Args:
        ensemble_dir: Directory containing ensemble checkpoints
        seeds: List of seeds used
    """
    logger.info("Analyzing ensemble agreement...")
    
    import numpy as np
    import torch
    from climate_discovery.config import FEATURES_GATING, TRAIN_NC
    from climate_discovery.data.datasets import ClimateDataset
    from climate_discovery.models.gating import GatingNetwork
    
    config = ModelConfig()
    
    # Load dataset
    dataset = ClimateDataset(
        TRAIN_NC,
        expert_features=[],  # Not needed for gating only
        gating_features=FEATURES_GATING,
        target=None,
        drop_nan=True,
    )
    
    X_gate = torch.from_numpy(dataset.X_gate).float()
    
    # Load all ensemble members
    all_regime_assignments = []
    
    for seed in seeds:
        checkpoint_path = ensemble_dir / f"seed_{seed}" / "gating.pth"
        
        if not checkpoint_path.exists():
            logger.warning(f"Checkpoint not found for seed {seed}")
            continue
        
        # Load model
        model = GatingNetwork(
            input_dim=len(FEATURES_GATING),
            num_regimes=config.n_regimes,
            hidden_dims=config.gating_hidden_dims,
            dropout=0.0,
        )
        
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        
        model.eval()
        
        # Get regime assignments
        with torch.no_grad():
            probs = model(X_gate)
            regime_ids = torch.argmax(probs, dim=1).numpy()
        
        all_regime_assignments.append(regime_ids)
    
    if len(all_regime_assignments) < 2:
        logger.error("Need at least 2 ensemble members for agreement analysis")
        return
    
    # Stack: (n_ensemble, n_samples)
    regime_matrix = np.stack(all_regime_assignments, axis=0)
    
    # Compute agreement
    from scipy.stats import mode
    mode_regime, mode_count = mode(regime_matrix, axis=0, keepdims=False)
    agreement = mode_count / len(all_regime_assignments)
    
    # Statistics
    logger.info("\nEnsemble Agreement Statistics:")
    logger.info(f"  Mean agreement: {np.mean(agreement):.3f}")
    logger.info(f"  Median agreement: {np.median(agreement):.3f}")
    logger.info(f"  Min agreement: {np.min(agreement):.3f}")
    logger.info(f"  Max agreement: {np.max(agreement):.3f}")
    
    # Histogram
    logger.info("\nAgreement Distribution:")
    for threshold in [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
        frac = np.mean(agreement >= threshold)
        logger.info(f"  ≥{threshold:.1f}: {frac*100:.1f}%")
    
    # Save agreement map
    output_path = ensemble_dir / "ensemble_agreement.npy"
    np.save(output_path, agreement)
    logger.info(f"\n✓ Agreement map saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Train ensemble of gating networks"
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=[0, 1, 2, 3, 4],
        help="Random seeds for ensemble members"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output directory for ensemble"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Number of epochs per member"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=2048,
        help="Batch size"
    )
    parser.add_argument(
        "--use_kmeans",
        action="store_true",
        help="Use K-means warm start"
    )
    parser.add_argument(
        "--parallel",
        action="store_true",
        help="Train members in parallel"
    )
    parser.add_argument(
        "--max_workers",
        type=int,
        default=None,
        help="Maximum parallel workers"
    )
    parser.add_argument(
        "--analyze",
        action="store_true",
        help="Analyze ensemble agreement after training"
    )
    
    args = parser.parse_args()
    
    # Output directory
    output_dir = Path(args.output) if args.output else CHECKPOINT_DIR / "ensemble"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("=" * 60)
    logger.info("ENSEMBLE TRAINING")
    logger.info("=" * 60)
    logger.info(f"Seeds: {args.seeds}")
    logger.info(f"Output: {output_dir}")
    logger.info(f"Epochs: {args.epochs}")
    logger.info(f"Batch size: {args.batch_size}")
    logger.info(f"K-means init: {args.use_kmeans}")
    logger.info(f"Parallel: {args.parallel}")
    
    # Train ensemble
    if args.parallel:
        status = train_parallel(
            args.seeds,
            output_dir,
            args.epochs,
            args.batch_size,
            args.use_kmeans,
            args.max_workers,
        )
    else:
        # Sequential training
        status = {}
        for seed in args.seeds:
            member_dir = output_dir / f"seed_{seed}"
            success = train_single_member(
                seed,
                member_dir,
                args.epochs,
                args.batch_size,
                args.use_kmeans,
            )
            status[seed] = success
    
    # Check if all succeeded
    if not all(status.values()):
        logger.error("Some ensemble members failed to train")
        failed_seeds = [seed for seed, success in status.items() if not success]
        logger.error(f"Failed seeds: {failed_seeds}")
        sys.exit(1)
    
    logger.info("=" * 60)
    logger.info("✓ ALL ENSEMBLE MEMBERS TRAINED SUCCESSFULLY")
    logger.info("=" * 60)
    
    # Optional: Analyze ensemble agreement
    if args.analyze:
        logger.info("\n" + "=" * 60)
        logger.info("ENSEMBLE AGREEMENT ANALYSIS")
        logger.info("=" * 60)
        analyze_ensemble_agreement(output_dir, args.seeds)


if __name__ == "__main__":
    main()