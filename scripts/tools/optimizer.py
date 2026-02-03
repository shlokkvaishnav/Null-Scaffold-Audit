"""Generalized Regime Optimizer for SD-MoSE

This script optimizes any regime(s) with custom PySR settings.
Replaces the hardcoded scripts/rerun_regime3.py with a flexible tool.

Usage:
    # Optimize single regime
    python scripts/tools/optimize_regime.py --regime 3 --iterations 1000 --maxdepth 6
    
    # Optimize multiple regimes
    python scripts/tools/optimize_regime.py --regimes 0 3 5 --iterations 1000
    
    # Custom operators
    python scripts/tools/optimize_regime.py --regime 3 --operators "abs,tanh,cos"
"""

import sys
from pathlib import Path
import argparse
import logging
import time
import numpy as np
import pandas as pd
import xarray as xr

# Setup paths
root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(root / "src"))

from sdmose.config import (
    FUSED_NC,
    FEATURES_EXPERT,
    TARGET,
    ModelConfig,
)
from sdmose.models.symbolic import SymbolicExpert

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_regime_assignments(results_dir="results"):
    """Load existing regime assignments from previous run."""
    predictions_file = Path(results_dir) / "uncertainty_predictions.csv"
    
    if not predictions_file.exists():
        raise FileNotFoundError(
            f"No previous results found at {predictions_file}. "
            "Run the full pipeline first: python scripts/run_complete_pipeline.py"
        )
    
    df = pd.read_csv(predictions_file)
    return df['regime'].values


def optimize_regime(
    regime_id, 
    X_expert, 
    y, 
    regime_assignments,
    variable_names,
    iterations=1000,
    maxdepth=6,
    operators=None,
    max_samples=11000,
    results_dir="results"
):
    """Optimize a single regime with custom settings.
    
    Args:
        regime_id: ID of regime to optimize
        X_expert: Expert features (N, D)
        y: Target values (N,)
        regime_assignments: Regime assignment for each sample (N,)
        variable_names: List of feature names
        iterations: PySR iterations
        maxdepth: Max tree depth
        operators: Custom unary operators (comma-separated string)
        max_samples: Max samples for PySR
        results_dir: Directory to save results
        
    Returns:
        dict with equation, complexity, score, r2, rmse
    """
    logger.info(f"\n{'='*70}")
    logger.info(f"Optimizing Regime {regime_id}")
    logger.info(f"{'='*70}")
    logger.info(f"Settings:")
    logger.info(f"  - Iterations: {iterations}")
    logger.info(f"  - Max Depth: {maxdepth}")
    logger.info(f"  - Custom operators: {operators or 'Default'}")
    
    # Filter to regime samples
    regime_mask = regime_assignments == regime_id
    n_samples = np.sum(regime_mask)
    logger.info(f"  - Regime {regime_id} samples: {n_samples:,}")
    
    if n_samples < 100:
        logger.warning(f"  ⚠️  Too few samples ({n_samples}) for regime {regime_id}, skipping")
        return None
    
    X_regime = X_expert[regime_mask]
    y_regime = y[regime_mask]
    
    # Subsample if needed
    if len(X_regime) > max_samples:
        logger.info(f"  - Subsampling: {len(X_regime):,} → {max_samples:,} points")
        indices = np.random.choice(len(X_regime), size=max_samples, replace=False)
        X_regime = X_regime[indices]
        y_regime = y_regime[indices]
    
    # Configure PySR
    config = ModelConfig()
    
    # Parse custom operators
    if operators:
        unary_ops = [op.strip() for op in operators.split(',')]
        logger.info(f"  - Using custom operators: {unary_ops}")
    else:
        unary_ops = config.pysr_unary_operators
    
    try:
        from pysr import PySRRegressor
        
        pysr_config = {
            "niterations": iterations,
            "populations": 31,
            "binary_operators": config.pysr_binary_operators,
            "unary_operators": unary_ops,
            "maxsize": 25,
            "maxdepth": maxdepth,
            "parsimony": config.pysr_complexity_penalty,
            "random_state": 42,
            "temp_equation_file": True,
            "delete_tempfiles": True,
            "verbosity": 1,
            "progress": True,
            "batching": False,
        }
        
        logger.info(f"\n🚀 Starting symbolic regression...")
        logger.info(f"   ETA: ~{iterations * 0.04:.1f} minutes")
        logger.info(f"   Started: {time.strftime('%H:%M:%S')}")
        start_time = time.time()
        
        model = PySRRegressor(**pysr_config)
        model.fit(X_regime, y_regime, variable_names=variable_names)
        
        # Extract results
        best = model.get_best()
        equation = str(best.equation)
        score = float(best.score)
        complexity = int(best.complexity)
        
        elapsed = time.time() - start_time
        logger.info(f"\n✓ Optimization completed in {elapsed/60:.1f} minutes")
        logger.info(f"   Finished: {time.strftime('%H:%M:%S')}")
        
        # Evaluate performance
        y_pred = model.predict(X_regime)
        from sklearn.metrics import r2_score, mean_squared_error
        r2 = r2_score(y_regime, y_pred)
        rmse = np.sqrt(mean_squared_error(y_regime, y_pred))
        
        logger.info(f"\n{'='*70}")
        logger.info(f"Regime {regime_id} Results:")
        logger.info(f"  Equation: {equation}")
        logger.info(f"  Complexity: {complexity}")
        logger.info(f"  Score: {score:.4f}")
        logger.info(f"  R²: {r2:.4f}")
        logger.info(f"  RMSE: {rmse:.2f} μatm")
        logger.info(f"{'='*70}")
        
        # Save results
        results_path = Path(results_dir)
        results_path.mkdir(exist_ok=True)
        
        output_file = results_path / f"regime{regime_id}_optimized.txt"
        with open(output_file, "w") as f:
            f.write(f"Regime {regime_id} Optimization Results\n")
            f.write("="*70 + "\n\n")
            f.write(f"Runtime: {elapsed/60:.1f} minutes\n")
            f.write(f"Settings:\n")
            f.write(f"  - Iterations: {iterations}\n")
            f.write(f"  - Max Depth: {maxdepth}\n")
            f.write(f"  - Operators: {', '.join(unary_ops)}\n\n")
            f.write(f"Discovered Equation:\n")
            f.write(f"  {equation}\n\n")
            f.write(f"Performance Metrics:\n")
            f.write(f"  - Complexity: {complexity}\n")
            f.write(f"  - Score: {score:.4f}\n")
            f.write(f"  - R² Score: {r2:.4f}\n")
            f.write(f"  - RMSE: {rmse:.2f} μatm\n")
        
        logger.info(f"\n💾 Results saved to: {output_file}")
        
        return {
            'regime_id': regime_id,
            'equation': equation,
            'complexity': complexity,
            'score': score,
            'r2': r2,
            'rmse': rmse,
            'runtime_minutes': elapsed / 60
        }
        
    except Exception as e:
        logger.error(f"\n❌ Error optimizing regime {regime_id}: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return None


def main():
    parser = argparse.ArgumentParser(
        description='Optimize specific regimes with custom PySR settings',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Regime selection
    regime_group = parser.add_mutually_exclusive_group(required=True)
    regime_group.add_argument('--regime', type=int,
                             help='Single regime ID to optimize')
    regime_group.add_argument('--regimes', type=int, nargs='+',
                             help='Multiple regime IDs to optimize')
    
    # PySR settings
    parser.add_argument('--iterations', type=int, default=1000,
                       help='PySR iterations per regime')
    parser.add_argument('--maxdepth', type=int, default=6,
                       help='Maximum tree depth (prevents overfitting)')
    parser.add_argument('--operators', type=str, default=None,
                       help='Custom unary operators (comma-separated, e.g., "abs,tanh,cos")')
    
    # Data settings
    parser.add_argument('--max-samples', type=int, default=11000,
                       help='Maximum samples for PySR (subsampling)')
    parser.add_argument('--results-dir', type=str, default='results',
                       help='Directory with existing results and for saving outputs')
    
    args = parser.parse_args()
    
    # Determine which regimes to optimize
    if args.regime is not None:
        regime_ids = [args.regime]
    else:
        regime_ids = args.regimes
    
    logger.info("\n" + "="*70)
    logger.info("SD-MoSE: Generalized Regime Optimizer")
    logger.info("="*70)
    logger.info(f"Regimes to optimize: {regime_ids}")
    logger.info(f"Total regimes: {len(regime_ids)}")
    logger.info("="*70 + "\n")
    
    # Load data
    logger.info("📊 Loading fused dataset...")
    ds = xr.open_dataset(FUSED_NC)
    df = ds.to_dataframe().reset_index()
    df = df.dropna(subset=FEATURES_EXPERT + [TARGET])
    logger.info(f"   {len(df):,} valid samples")
    
    # Load regime assignments
    logger.info("📋 Loading existing regime assignments...")
    regime_assignments = load_regime_assignments(args.results_dir)
    
    # Align data
    if len(regime_assignments) != len(df):
        logger.warning(
            f"Size mismatch: {len(regime_assignments)} assignments vs {len(df)} samples"
        )
        min_len = min(len(regime_assignments), len(df))
        regime_assignments = regime_assignments[:min_len]
        df = df.iloc[:min_len]
        logger.info(f"   Using {min_len:,} samples")
    
    # Extract features
    X_expert = df[FEATURES_EXPERT].values
    y = df[TARGET].values
    
    # Optimize each regime
    all_results = []
    for regime_id in regime_ids:
        result = optimize_regime(
            regime_id=regime_id,
            X_expert=X_expert,
            y=y,
            regime_assignments=regime_assignments,
            variable_names=FEATURES_EXPERT,
            iterations=args.iterations,
            maxdepth=args.maxdepth,
            operators=args.operators,
            max_samples=args.max_samples,
            results_dir=args.results_dir
        )
        
        if result:
            all_results.append(result)
    
    # Summary
    logger.info("\n" + "="*70)
    logger.info("OPTIMIZATION SUMMARY")
    logger.info("="*70)
    logger.info(f"✓ Successfully optimized {len(all_results)} / {len(regime_ids)} regimes\n")
    
    if all_results:
        summary_df = pd.DataFrame(all_results)
        logger.info(summary_df.to_string(index=False))
        
        # Save summary
        summary_path = Path(args.results_dir) / "optimized_regimes_summary.csv"
        summary_df.to_csv(summary_path, index=False)
        logger.info(f"\n💾 Summary saved to: {summary_path}")
    
    logger.info("="*70 + "\n")


if __name__ == "__main__":
    main()
