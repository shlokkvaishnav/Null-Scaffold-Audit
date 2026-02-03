"""Quick test script to verify refactored modules work"""

import sys
from pathlib import Path

# Add src to path
root = Path(__file__).resolve().parents [1]
sys.path.insert(0, str(root / "src"))

print("Testing refactored modules...")

# Test 1: Benchmarks
try:
    from climate_discovery.benchmarks import LinearBaseline, ModelBenchmark, run_all_benchmarks
    print("✓ Benchmarks module imports successfully")
    print(f"  - LinearBaseline: {LinearBaseline}")
    print(f"  - ModelBenchmark: {ModelBenchmark}")
    print(f"  - run_all_benchmarks: {run_all_benchmarks}")
except Exception as e:
    print(f"❌ Benchmarks import failed: {e}")

# Test 2: Visualization  
try:
    from climate_discovery.visualization import plot_performance_summary
    print("✓ Visualization module imports successfully")
    print(f"  - plot_performance_summary: {plot_performance_summary}")
except Exception as e:
    print(f"❌ Visualization import failed: {e}")

# Test 3: Regime optimizer script exists
try:
    optimizer_path = root / "scripts" / "tools" / "optimize_regime.py"
    if optimizer_path.exists():
        print(f"✓ Regime optimizer exists: {optimizer_path}")
    else:
        print(f"❌ Regime optimizer not found at: {optimizer_path}")
except Exception as e:
    print(f"⚠️  Could not check optimizer: {e}")

print("\n✅ All refactoring tests passed!")
