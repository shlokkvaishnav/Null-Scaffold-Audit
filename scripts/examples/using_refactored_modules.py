"""Example: Using Refactored SD-MoSE Modules

This file demonstrates how to use the newly refactored modules.
"""
# -*- coding: utf-8 -*-

import sys
from pathlib import Path
import numpy as np
import pandas as pd

# Add src to path
root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(root / "src"))

print("=" * 70)
print("SD-MoSE Refactored Modules - Usage Examples")
print("=" * 70)

# Example 1: Using Benchmark Models
print("\n[BENCHMARKING] Example 1")
print("-" * 70)

try:
    from climate_discovery.benchmarks import (
        LinearBaseline,
        RFBaseline,
        XGBBaseline,
        ModelBenchmark,
        run_all_benchmarks
    )
    
    print("SUCCESS: Benchmark models imported successfully")

    print("\nAvailable baseline models:")
    print("  - LinearBaseline")
    print("  - RFBaseline")
    print("  - XGBBaseline")
    print("\nUsage:")
    print("  model = LinearBaseline()")
    print("  model.fit(X_train, y_train)")
    print("  predictions = model.predict(X_test)")
    
    print("\nFor full benchmarking:")
    print("  benchmark = run_all_benchmarks(X_train, y_train, X_test, y_test)")
    print("  benchmark.plot_comparison()")
    
except ImportError as e:
    print(f"WARNING: Import failed: {e}")

# Example 2: Using Visualization Module
print("\n\n[VISUALIZATION] Example 2")
print("-" * 70)

try:
    from climate_discovery.visualization import (
        plot_performance_summary,
        plot_tradeoff,
        generate_all_figures
    )
    
    print("SUCCESS: Visualization functions imported successfully")
    print("\nAvailable plot functions:")
    print("  - plot_performance_summary(perf_df): 4-panel performance plot")
    print("  - plot_tradeoff(perf_df): R^2 vs RMSE scatter")
    print("  - generate_all_figures(): Publication-ready figures")
    
    print("\nUsage:")
    print("  perf_df = pd.read_csv('results/regime_performance.csv')")
    print("  plot_performance_summary(perf_df, save_path='figures/performance.png')")
    
except ImportError as e:
    print(f"WARNING: Import failed: {e}")

# Example 3: Regime Optimizer
print("\n\n[REGIME OPTIMIZER] Example 3")
print("-" * 70)

optimizer_path = root / "scripts" / "tools" / "optimize_regime.py"
if optimizer_path.exists():
    print(f"SUCCESS: Regime optimizer available at:")
    print(f"  {optimizer_path}")
    
    print("\nUsage examples:")
    print("\n  # Optimize single regime:")
    print("  python scripts/tools/optimize_regime.py --regime 3 --iterations 1000")
    
    print("\n  # Optimize multiple regimes:")
    print("  python scripts/tools/optimize_regime.py --regimes 0 3 5 --iterations 1000")
    
    print("\n  # With custom operators:")
    print("  python scripts/tools/optimize_regime.py --regime 3 --operators 'abs,tanh,cos'")
    
    print("\n  # Help:")
    print("  python scripts/tools/optimize_regime.py --help")
else:
    print(f"WARNING: Optimizer not found at {optimizer_path}")

# Summary
print("\n\n" + "=" * 70)
print("[SUCCESS] Refactoring Summary")
print("=" * 70)
print("\n* All modules use clean, consolidated imports")
print("* Benchmarking: climate_discovery.benchmarks")
print("* Visualization: climate_discovery.visualization")
print("* Regime optimizer: scripts/tools/optimize_regime.py")
print("\n" + "=" * 70 + "\n")
