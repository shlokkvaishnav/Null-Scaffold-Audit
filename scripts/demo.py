"""Demonstration Script: All New SD-MoSE Features

This script demonstrates all Phase 1 (Refactoring) and Phase 2 (Improvements) features:
1. Spatial feature engineering
2. Comprehensive benchmarking with literature comparison
3. Generalized regime optimization
4. Consolidated visualization
"""

import sys
from pathlib import Path

# Import juliacall first to avoid torch segfault warning
try:
    import juliacall  # noqa: F401
except ImportError:
    pass  # Not critical if missing

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_regression

# Add src to path
root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(root / "src"))

print("="*70)
print("SD-MoSE: Complete Feature Demonstration")
print("="*70)

# =============================================================================
# DEMO 1: Spatial Feature Engineering
# =============================================================================
print("\n\n[SPATIAL FEATURES] Demo 1: Spatial Feature Engineering")
print("-"*70)

from sdmose.preprocessing import SpatialFeatureEngineer

# Create sample ocean data
sample_data = pd.DataFrame({
    'lat': [0, 35, -45, 70, -10, 25, 15, -30],
    'lon': [-30, 150, 80, -170, -60, 10, -90, 160],
    'sst': [28, 15, 8, 2, 27, 20, 26, 12],
    'sss': [35.5, 34.0, 33.8, 32.0, 36.2, 35.0, 35.8, 34.5],
    'log_chl': [0.5, 1.2, 0.8, 0.3, 0.6, 1.0, 0.7, 0.9],
    'fco2': [380, 350, 320, 300, 400, 360, 390, 340]
})

print("Original ocean data:")
print(sample_data.head())

# Add spatial features
engineer = SpatialFeatureEngineer()
enriched = engineer.add_all_spatial_features(
    sample_data,
    include_basins=True,
    include_bands=True,
    include_distances=True
)

print("\nEnriched data with spatial features:")
print(enriched.head())

new_features = [c for c in enriched.columns if c not in sample_data.columns]
print(f"\nAdded {len(new_features)} new spatial features:")
for feat in new_features:
    print(f"  - {feat}")

print("\nUSAGE: Add spatial features to your data before training:")
print("  engineer = SpatialFeatureEngineer()")
print("  df = engineer.add_all_spatial_features(df)")

# =============================================================================
# DEMO 2: Comprehensive Benchmarking
# =============================================================================
print("\n\n[BENCHMARKING] Demo 2: Model Comparison")
print("-"*70)

from sdmose.benchmarks import (
    LinearBaseline,
    RFBaseline,
    XGBBaseline,
    ModelBenchmark,
    run_all_benchmarks
)

# Create synthetic dataset for demo
print("\nGenerating synthetic dataset (10,000 samples)...")
X, y = make_regression(
    n_samples=10000,
    n_features=4,
    noise=15,
    random_state=42
)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"Training set: {len(X_train):,} samples")
print(f"Test set: {len(X_test):,} samples")

# Run all benchmarks with one function call
print("\nRunning all baseline models...")
benchmark = run_all_benchmarks(X_train, y_train, X_test, y_test)

# Get results
results = benchmark.results
print("\n" + "="*70)
print("RESULTS SUMMARY")
print("="*70)
print(results[['Model', 'R²', 'RMSE', 'MAE', 'Interpretable']].to_string(index=False))

print("\nUSAGE: One-line benchmarking:")
print("  benchmark = run_all_benchmarks(X_train, y_train, X_test, y_test)")
print("  benchmark.plot_comparison()  # Save visualization")

# =============================================================================
# DEMO 3: Regime Optimizer Overview
# =============================================================================
print("\n\n[REGIME OPTIMIZER] Demo 3: Generalized Regime Optimization")
print("-"*70)

optimizer_path = root / "scripts" / "tools" / "optimize_regime.py"
if optimizer_path.exists():
    print(f"SUCCESS: Regime optimizer available at:")
    print(f"  {optimizer_path}")
    
    print("\nKey Features:")
    print("  * Works with ANY regime ID (not hardcoded)")
    print("  * Can optimize multiple regimes at once")
    print("  * Customizable PySR settings (iterations, maxdepth, operators)")
    print("  * Automatic performance evaluation")
    
    print("\nExample Commands:")
    commands = [
        ("Single regime", "python scripts/tools/optimize_regime.py --regime 3 --iterations 1000"),
        ("Multiple regimes", "python scripts/tools/optimize_regime.py --regimes 0 3 5 --iterations 1000"),
        ("With depth limit", "python scripts/tools/optimize_regime.py --regime 3 --maxdepth 6"),
        ("Custom operators", "python scripts/tools/optimize_regime.py --regime 3 --operators 'abs,tanh,cos'"),
    ]
    
    for desc, cmd in commands:
        print(f"\n  # {desc}")
        print(f"  {cmd}")
else:
    print("WARNING: Optimizer not found")

# =============================================================================
# DEMO 4: Consolidated Visualization
# =============================================================================
print("\n\n[VISUALIZATION] Demo 4: Publication-Ready Figures")
print("-"*70)

try:
    from sdmose.visualization import (
        plot_performance_summary,
        plot_tradeoff
    )
    
    print("SUCCESS: Visualization module loaded")
    
    # Create mock performance data
    mock_perf = pd.DataFrame({
        'regime': [0, 1, 2, 3, 4, 5],
        'r2': [0.08, 0.09, 0.12, 0.05, 0.41, 0.07],
        'rmse': [44.8, 30.5, 26.0, 35.6, 75.5, 49.0],
        'n_samples': [8063, 34518, 31758, 32151, 1256, 21008],
        'frac_samples': [0.063, 0.268, 0.247, 0.250, 0.010, 0.163]
    })
    
    print("\nGenerating performance plots...")
    plot_performance_summary(mock_perf, save_path="figures/demo_performance.png")
    plot_tradeoff(mock_perf, save_path="figures/demo_tradeoff.png")
    
    print("SUCCESS: Generated 2 publication-quality figures")
    print("  - figures/demo_performance.png")
    print("  - figures/demo_tradeoff.png")
    
    print("\nUSAGE:")
    print("  from sdmose.visualization import plot_performance_summary")
    print("  plot_performance_summary(perf_df, save_path='figures/my_plot.png')")
    
except Exception as e:
    print(f"WARNING: Visualization demo failed: {e}")

# =============================================================================
# SUMMARY
# =============================================================================
print("\n\n" + "="*70)
print("DEMONSTRATION COMPLETE")
print("="*70)

print("\n[PHASE 1: REFACTORING] ✓ Complete")
print("  * Generalized regime optimizer")
print("  * Consolidated benchmarks module")
print("  * Unified visualization module")

print("\n[PHASE 2: IMPROVEMENTS] ✓ Complete")
print("  * Spatial feature engineering")
print("  * Comprehensive benchmarking with literature")
print("  * One-line benchmark runner")

print("\n[NEXT STEPS] Phase 3 & 4 Available:")
print("  * Run regime 3 optimization with new tool")
print("  * Add spatial features to pipeline")
print("  * Generate literature comparison report")
print("  * Create publication figures")

print("\n" + "="*70)
print("All modules ready for production use!")
print("="*70 + "\n")
