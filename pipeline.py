#!/usr/bin/env python3
"""SD-MoSE Pipeline - Soft-Dynamic Mixture of Symbolic Experts.

Main entry point for ocean CO₂ equation discovery.

Usage:
    python pipeline.py --n-regimes 6 --pysr-iterations 40
    
    # Quick test (5 iterations)
    python pipeline.py --n-regimes 6 --pysr-iterations 5 --test
"""

import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.cluster import KMeans

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="SD-MoSE: Discover interpretable ocean CO2 equations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python pipeline.py                          # Full run with defaults
  python pipeline.py --n-regimes 6            # 6 ocean regimes
  python pipeline.py --pysr-iterations 5      # Quick test
  python pipeline.py --test                   # Test mode (small subset)
        """
    )
    
    # Core parameters
    parser.add_argument('--n-regimes', type=int, default=6,
                        help='Number of ocean regimes (default: 6)')
    parser.add_argument('--pysr-iterations', type=int, default=40,
                        help='PySR iterations per regime (default: 40)')
    
    # Data parameters
    parser.add_argument('--train-years', type=str, default='2000-2020',
                        help='Training years (default: 2000-2020)')
    parser.add_argument('--test-years', type=str, default='2021-2023',
                        help='Test years (default: 2021-2023)')
    
    # Mode flags
    parser.add_argument('--test', action='store_true',
                        help='Test mode: use small data subset')
    parser.add_argument('--skip-symbolic', action='store_true',
                        help='Skip symbolic regression (gating only)')
    
    # Output
    parser.add_argument('--output-dir', type=str, default='results',
                        help='Output directory (default: results)')
    
    return parser.parse_args()


def main():
    """Main pipeline execution."""
    args = parse_args()
    
    # Parse year ranges
    train_start, train_end = map(int, args.train_years.split('-'))
    test_start, test_end = map(int, args.test_years.split('-'))
    
    # Banner
    print("=" * 70)
    print("SD-MoSE: SOFT-DYNAMIC MIXTURE OF SYMBOLIC EXPERTS")
    print("Discovering Interpretable Ocean CO2 Equations")
    print("=" * 70)
    print(f"  Regimes: {args.n_regimes}")
    print(f"  PySR iterations: {args.pysr_iterations}")
    print(f"  Train years: {train_start}-{train_end}")
    print(f"  Test years: {test_start}-{test_end}")
    print(f"  Mode: {'TEST' if args.test else 'FULL'}")
    print("=" * 70)
    print()
    
    start_time = time.time()
    
    # =========================================================================
    # STAGE 1: DATA LOADING
    # =========================================================================
    logger.info("STAGE 1: Loading data...")
    
    try:
        from data.loader import SDMoSEDataLoader
    except ImportError as e:
        logger.error(f"Failed to import data loader: {e}")
        logger.error("Make sure you're running from the project root directory")
        sys.exit(1)
    
    loader = SDMoSEDataLoader()
    train_df, test_df = loader.load(
        train_years=(train_start, train_end),
        test_years=(test_start, test_end),
    )
    
    # Extract arrays
    EXPERT_FEATURES = ['sst', 'sss', 'log_chl']
    GATING_FEATURES = ['lat_norm', 'lon_norm', 'sst', 'sin_month', 'cos_month']
    TARGET = 'fco2'
    
    X_expert_train = train_df[EXPERT_FEATURES].values
    X_gate_train = train_df[GATING_FEATURES].values
    y_train = train_df[TARGET].values
    
    X_expert_test = test_df[EXPERT_FEATURES].values
    X_gate_test = test_df[GATING_FEATURES].values
    y_test = test_df[TARGET].values
    
    # Test mode: subsample
    if args.test:
        n_test_samples = min(5000, len(y_train))
        idx = np.random.choice(len(y_train), n_test_samples, replace=False)
        X_expert_train = X_expert_train[idx]
        X_gate_train = X_gate_train[idx]
        y_train = y_train[idx]
        logger.info(f"TEST MODE: Subsampled to {n_test_samples} samples")
    
    logger.info(f"Train samples: {len(y_train)}")
    logger.info(f"Test samples: {len(y_test)}")
    
    # =========================================================================
    # STAGE 2: REGIME ASSIGNMENT (K-means Gating)
    # =========================================================================
    logger.info(f"\nSTAGE 2: Assigning {args.n_regimes} ocean regimes...")
    
    kmeans = KMeans(n_clusters=args.n_regimes, random_state=42, n_init=10)
    regime_labels_train = kmeans.fit_predict(X_gate_train)
    regime_labels_test = kmeans.predict(X_gate_test)
    
    # Create soft assignments (one-hot for K-means)
    regime_probs_train = np.zeros((len(y_train), args.n_regimes))
    regime_probs_train[np.arange(len(y_train)), regime_labels_train] = 1.0
    
    regime_probs_test = np.zeros((len(y_test), args.n_regimes))
    regime_probs_test[np.arange(len(y_test)), regime_labels_test] = 1.0
    
    # Print regime distribution
    print("\nRegime Distribution:")
    print("-" * 40)
    for k in range(args.n_regimes):
        n_k = np.sum(regime_labels_train == k)
        pct = 100 * n_k / len(y_train)
        print(f"  Regime {k}: {n_k:>6} samples ({pct:>5.1f}%)")
    print()
    
    if args.skip_symbolic:
        logger.info("Skipping symbolic regression (--skip-symbolic flag)")
        return
    
    # =========================================================================
    # STAGE 3: SYMBOLIC REGRESSION (PySR per regime)
    # =========================================================================
    logger.info(f"\nSTAGE 3: Discovering symbolic equations...")
    logger.info(f"  PySR iterations: {args.pysr_iterations}")
    
    try:
        from models.symbolic import MixtureOfSymbolicExperts
    except ImportError as e:
        logger.error(f"Failed to import symbolic module: {e}")
        logger.error("Ensure PySR is installed: pip install pysr")
        sys.exit(1)
    
    expert_config = {
        'niterations': args.pysr_iterations,
        'populations': 31,
        'maxsize': 20,
        'binary_operators': ['+', '-', '*', '/'],
        'unary_operators': ['exp', 'log', 'sqrt', 'square'],
    }
    
    experts = MixtureOfSymbolicExperts(
        num_regimes=args.n_regimes,
        expert_config=expert_config,
    )
    
    experts.fit(
        X=X_expert_train,
        y=y_train,
        regime_probs=regime_probs_train,
        variable_names=EXPERT_FEATURES,
        max_samples=10000,  # Limit samples per regime for speed
    )
    
    # =========================================================================
    # STAGE 4: EVALUATION
    # =========================================================================
    logger.info("\nSTAGE 4: Evaluating model...")
    
    from utils.metrics import calculate_metrics, print_metrics
    
    # Predictions
    y_pred_train = experts.predict(X_expert_train, regime_probs_train)
    y_pred_test = experts.predict(X_expert_test, regime_probs_test)
    
    # Calculate metrics
    train_metrics = calculate_metrics(
        y_train, y_pred_train, 
        regime_labels_train, args.n_regimes
    )
    test_metrics = calculate_metrics(
        y_test, y_pred_test,
        regime_labels_test, args.n_regimes
    )
    
    print("\n" + "=" * 60)
    print("TRAIN SET PERFORMANCE")
    print_metrics(train_metrics)
    
    print("\n" + "=" * 60)
    print("TEST SET PERFORMANCE")
    print_metrics(test_metrics)
    
    # =========================================================================
    # STAGE 5: SAVE RESULTS
    # =========================================================================
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Save equations
    equations_file = output_dir / "equations.txt"
    experts.save_equations(equations_file)
    logger.info(f"Saved equations to {equations_file}")
    
    # Print discovered equations
    print("\n" + "=" * 60)
    print("DISCOVERED EQUATIONS")
    print("=" * 60)
    equations = experts.get_all_equations()
    for k, eq in enumerate(equations):
        n_k = np.sum(regime_labels_train == k)
        pct = 100 * n_k / len(y_train)
        print(f"\nRegime {k} ({pct:.1f}% of ocean):")
        print(f"  pCO2 = {eq}")
    print()
    
    # Summary
    elapsed = time.time() - start_time
    print("=" * 60)
    print("PIPELINE COMPLETE")
    print("=" * 60)
    print(f"  Total time: {elapsed/60:.1f} minutes")
    print(f"  Results saved to: {output_dir}")
    print(f"  Equations: {equations_file}")
    print()
    print("Next steps:")
    print("  1. Review equations in results/equations.txt")
    print("  2. Generate figures with: python utils/visualization.py")
    print()


if __name__ == "__main__":
    main()
