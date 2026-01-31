"""Re-optimize Regime 3 with enhanced PySR settings.

This script loads the existing SD-MoSE model and re-fits only Regime 3
with optimized parameters:
- 1000 iterations (vs 500)
- maxdepth=6 (prevent SST^8 overfitting)
- Enhanced operators (abs, tanh)

Expected runtime: ~40-60 minutes
Expected improvement: R² from 0.05 to >0.10, complexity ≤ 6
"""

import numpy as np
import pandas as pd
import xarray as xr
from pathlib import Path
import logging
import time

from climate_discovery.config import (
    FUSED_NC,
    FEATURES_SOFT_REGIME,
    FEATURES_EXPERT,
    TARGET,
    ModelConfig,
)
from climate_discovery.models.symbolic import SymbolicExpert

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_regime_assignments():
    """Load existing regime assignments from previous run."""
    results_dir = Path("results")
    predictions_file = results_dir / "uncertainty_predictions.csv"
    
    if not predictions_file.exists():
        raise FileNotFoundError(
            f"No previous results found at {predictions_file}. "
            "Run the full pipeline first: python scripts/run_complete_pipeline.py"
        )
    
    df = pd.read_csv(predictions_file)
    return df['regime'].values

def main():
    logger.info("=" * 70)
    logger.info("SD-MoSE: Targeted Regime 3 Optimization")
    logger.info("=" * 70)
    logger.info("Settings:")
    logger.info("  - Iterations: 1000 (was 500)")
    logger.info("  - Max Depth: 6 (prevent overfitting)")
    logger.info("  - Operators: +abs, +tanh")
    logger.info("=" * 70)
    
    # Load data
    logger.info("\n📊 Loading fused dataset...")
    ds = xr.open_dataset(FUSED_NC)
    df = ds.to_dataframe().reset_index()
    df = df.dropna(subset=FEATURES_EXPERT + [TARGET])
    
    # Load existing regime assignments
    logger.info("📋 Loading existing regime assignments...")
    regime_assignments = load_regime_assignments()
    
    if len(regime_assignments) != len(df):
        logger.warning(
            f"Size mismatch: {len(regime_assignments)} assignments vs {len(df)} samples. "
            "Truncating to shorter length..."
        )
        min_len = min(len(regime_assignments), len(df))
        regime_assignments = regime_assignments[:min_len]
        df = df.iloc[:min_len]
    
    # Extract features
    X_expert = df[FEATURES_EXPERT].values
    y = df[TARGET].values
    
    # Create regime probability matrix
    n_regimes = 6
    regime_probs = np.zeros((len(df), n_regimes))
    for i, regime_id in enumerate(regime_assignments):
        regime_probs[i, regime_id] = 1.0
    
    # Filter to Regime 3 samples
    regime_3_mask = regime_assignments == 3
    logger.info(f"Regime 3 samples: {np.sum(regime_3_mask):,}")
    
    X_regime3 = X_expert[regime_3_mask]
    y_regime3 = y[regime_3_mask]
    weights_regime3 = regime_probs[regime_3_mask, 3]
    
    # Setup enhanced configuration
    config = ModelConfig()
    
    logger.info("\n🔧 Creating enhanced Regime 3 expert...")
    expert_config = {
        "regime_id": 3,
        "niterations": 1000,  # ← DOUBLED
        "populations": 31,
        "binary_operators": config.pysr_binary_operators,
        "unary_operators": config.pysr_unary_operators,  # Now includes abs, tanh
        "complexity_penalty": config.pysr_complexity_penalty,
        "maxsize": 25,
        "verbosity": 1,
    }
    
    expert = SymbolicExpert(**expert_config)
    
    # Subsample if needed (same as production)
    max_samples = 11000
    if len(X_regime3) > max_samples:
        logger.info(f"  Subsampling: {len(X_regime3):,} → {max_samples:,} points")
        indices = np.random.choice(len(X_regime3), size=max_samples, replace=False)
        X_regime3 = X_regime3[indices]
        y_regime3 = y_regime3[indices]
        weights_regime3 = weights_regime3[indices]
    
    # Fit expert
    logger.info(f"\n🚀 Starting symbolic regression (ETA: ~40-60 min)...")
    logger.info(f"   Started: {time.strftime('%H:%M:%S')}")
    start_time = time.time()
    
    # Add maxdepth constraint to PySR config
    # We'll need to modify the fit call directly
    expert.model_ = None  # Reset to apply new config
    
    # Import PySR with maxdepth
    try:
        from pysr import PySRRegressor
        
        pysr_config = {
            "niterations": 1000,
            "populations": 31,
            "binary_operators": config.pysr_binary_operators,
            "unary_operators": config.pysr_unary_operators,
            "maxsize": 25,
            "parsimony": config.pysr_complexity_penalty,
            "random_state": 42,
            "temp_equation_file": True,
            "delete_tempfiles": True,
            "verbosity": 1,
            "progress": True,
            "maxdepth": 6,  # ← NEW: Prevent deep trees like SST^8
            "batching": False,
        }
        
        expert.model_ = PySRRegressor(**pysr_config)
        expert.model_.fit(
            X_regime3,
            y_regime3,
            weights=weights_regime3,
            variable_names=FEATURES_EXPERT
        )
        
        # Extract results
        best = expert.model_.get_best()
        equation = best.equation
        score = best.score
        complexity = best.complexity
        
        elapsed = time.time() - start_time
        logger.info(f"\n✓ Regime 3 optimization completed in {elapsed/60:.1f} minutes")
        logger.info(f"   Finished: {time.strftime('%H:%M:%S')}")
        logger.info(f"\n{'=' * 70}")
        logger.info(f"NEW Regime 3 Equation:")
        logger.info(f"  Equation: {equation}")
        logger.info(f"  Complexity: {complexity}")
        logger.info(f"  Score: {score:.4f}")
        logger.info(f"{'=' * 70}")
        
        # Compare to old
        old_equation = "SSS × SST⁸ + 368.50"
        old_complexity = 8
        old_score = "(unknown from old run)"
        
        logger.info(f"\nOLD Regime 3 Equation:")
        logger.info(f"  Equation: {old_equation}")
        logger.info(f"  Complexity: {old_complexity}")
        logger.info(f"  Score: {old_score}")
        logger.info(f"{'=' * 70}")
        
        # Save new equation
        results_dir = Path("results")
        results_dir.mkdir(exist_ok=True)
        
        with open(results_dir / "regime3_optimized.txt", "w") as f:
            f.write("Regime 3 Optimization Results\n")
            f.write("=" * 70 + "\n\n")
            f.write(f"Runtime: {elapsed/60:.1f} minutes\n")
            f.write(f"Settings: 1000 iterations, maxdepth=6, +abs +tanh\n\n")
            f.write("New Equation:\n")
            f.write(f"  {equation}\n")
            f.write(f"  Complexity: {complexity}\n")
            f.write(f"  Score: {score:.4f}\n\n")
            f.write("Old Equation:\n")
            f.write(f"  {old_equation}\n")
            f.write(f"  Complexity: {old_complexity}\n")
            f.write(f"  Score: N/A\n\n")
            f.write(f"Improvement: Complexity {old_complexity} → {complexity}\n")
        
        logger.info(f"\n💾 Results saved to: {results_dir / 'regime3_optimized.txt'}")
        
        # Performance evaluation
        y_pred = expert.model_.predict(X_regime3)
        from sklearn.metrics import r2_score, mean_squared_error
        r2 = r2_score(y_regime3, y_pred)
        rmse = np.sqrt(mean_squared_error(y_regime3, y_pred))
        
        logger.info(f"\n📊 Performance Metrics:")
        logger.info(f"  R² Score: {r2:.4f} (old: 0.0468)")
        logger.info(f"  RMSE: {rmse:.2f} μatm (old: 35.65 μatm)")
        
        if r2 > 0.10 and complexity <= 6:
            logger.info(f"\n🎉 SUCCESS! Targets achieved:")
            logger.info(f"   ✓ R² >  0.10: {r2:.4f}")
            logger.info(f"   ✓ Complexity ≤ 6: {complexity}")
        else:
            logger.info(f"\n⚠️  Targets not fully achieved:")
            if r2 <= 0.10:
                logger.info(f"   ✗ R² = {r2:.4f} (target > 0.10)")
            if complexity > 6:
                logger.info(f"   ✗ Complexity = {complexity} (target ≤ 6)")
        
    except Exception as e:
        logger.error(f"\n❌ Error during optimization: {e}")
        raise

if __name__ == "__main__":
    main()
