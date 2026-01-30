"""Example: Using equation versioning and ablation studies.

Demonstrates:
1. Saving equations with Git tracking
2. Loading and comparing equation versions  
3. Running ablation studies
4. Analyzing ablation results

Usage:
    python -m scripts.examples.versioning_ablation_demo
"""

import sys
from pathlib import Path

import numpy as np

# Add src to path
root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(root / "src"))

from climate_discovery.config import ModelConfig
from climate_discovery.utils.equation_version import EquationVersionManager
from scripts.experiments.ablation_grid import run_ablation_study


def demo_equation_versioning():
    """Demo: Equation versioning with Git tracking."""
    print("\n" + "="*70)
    print("DEMO 1: Equation Versioning")
    print("="*70)
    
    # Initialize manager
    manager = EquationVersionManager(
        equations_dir="equations",
        auto_commit=False,  # Set True to auto-commit
    )
    
    # Example equations (discovered by SD-MoSE)
    equations_v1 = {
        0: "fCO2 = 349.56 - 2.34 * exp(0.031 * SST)",
        1: "fCO2 = 380.2 + 3.14 * SST - 1.57 * SSS",
        2: "fCO2 = 412.3 * (1 + 0.045 * log(Chl))",
        3: "fCO2 = 295.1 + 1.2 * SST + 0.8 * |∇SST|",
        4: "fCO2 = 445.7 - 3.5 * SSS + 0.02 * SST^2",
        5: "fCO2 = 320.4 + 2.1 * SST * log(Chl + 1)",
    }
    
    # Configuration
    config = ModelConfig()
    config.num_regimes = 6
    config.entropy_weight = 0.005
    
    # Metrics
    metrics = {
        "test_r2": 0.447,
        "val_r2": 0.452,
        "test_mse": 127.3,
        "val_mse": 123.8,
    }
    
    # Save version 1.0.0
    print("\n📝 Saving equation version 1.0.0...")
    filepath = manager.save_equations(
        equations=equations_v1,
        config=config,
        metrics=metrics,
        version="1.0.0",
        notes="Baseline model with 6 regimes, physics constraints enabled",
    )
    
    print(f"\n✓ Saved to: {filepath}")
    print("\nFile contents preview:")
    with open(filepath) as f:
        lines = f.readlines()[:30]
        print("".join(lines))
    
    # Save version 1.1.0 (with improvements)
    equations_v1_1 = {
        0: "fCO2 = 348.92 - 2.41 * exp(0.0315 * SST)",  # Slightly improved
        1: "fCO2 = 379.8 + 3.21 * SST - 1.62 * SSS",
        2: "fCO2 = 411.5 * (1 + 0.047 * log(Chl + 0.1))",  # Better handling
        3: "fCO2 = 294.3 + 1.25 * SST + 0.85 * |∇SST|",
        4: "fCO2 = 444.2 - 3.6 * SSS + 0.021 * SST^2",
        5: "fCO2 = 319.7 + 2.15 * SST * log(Chl + 1)",
    }
    
    metrics_v1_1 = {
        "test_r2": 0.462,  # Improved!
        "val_r2": 0.468,
        "test_mse": 119.5,
        "val_mse": 115.2,
    }
    
    print("\n📝 Saving improved version 1.1.0...")
    filepath2 = manager.save_equations(
        equations=equations_v1_1,
        config=config,
        metrics=metrics_v1_1,
        version="1.1.0",
        notes="Improved with hierarchical gating and attention",
    )
    
    # List versions
    print("\n📋 All saved versions:")
    versions = manager.list_versions()
    for v in versions:
        print(f"  - {v['file']}")
        print(f"    R²: {v['metrics'].get('test_r2', 'N/A')}, "
              f"Commit: {v['commit']}")
    
    # Compare versions
    if len(versions) >= 2:
        print("\n🔍 Comparing versions...")
        manager.compare_versions("1.0.0", "1.1.0")


def demo_ablation_study():
    """Demo: Automated ablation study."""
    print("\n" + "="*70)
    print("DEMO 2: Ablation Study")
    print("="*70)
    
    # Define ablation parameters
    ablation_params = {
        "num_regimes": [3, 6, 9],
        "entropy_weight": [0.0, 0.005, 0.01],
    }
    
    print("\n🔬 Running quick ablation study...")
    print("   (This is a demo with simulated results)")
    
    # Run ablation (dry run first)
    print("\nDry run to see what would execute:")
    run_ablation_study(
        ablation_params=ablation_params,
        backend="none",
        dry_run=True,
    )
    
    # Uncomment to run actual ablation:
    # df = run_ablation_study(
    #     ablation_params=ablation_params,
    #     backend="wandb",
    #     dry_run=False,
    # )


def main():
    print("\n" + "="*70)
    print("EQUATION VERSIONING & ABLATION STUDY DEMO")
    print("="*70)
    
    # Demo 1: Equation versioning
    demo_equation_versioning()
    
    # Demo 2: Ablation study
    demo_ablation_study()
    
    print("\n" + "="*70)
    print("DEMO COMPLETE")
    print("="*70)
    print("\nNext steps:")
    print("  1. Check equations/ directory for saved versions")
    print("  2. Run: python -m scripts.experiments.ablation_grid --preset regime_count --dry-run")
    print("  3. Integrate versioning into your training script")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
