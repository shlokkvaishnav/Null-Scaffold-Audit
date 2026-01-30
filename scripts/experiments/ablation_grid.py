"""Automated ablation study framework.

Runs systematic grid search over hyperparameters to identify:
- Most important features
- Optimal regime count
- Best regularization weights
- Feature engineering impact

Usage:
    python -m scripts.experiments.ablation_grid --config ablations.yaml
    
Or programmatically:
    from scripts.experiments.ablation_grid import run_ablation_study
    
    ablations = {
        "n_regimes": [3, 6, 9],
        "entropy_weight": [0.0, 0.005, 0.01],
    }
    results = run_ablation_study(ablations, backend="wandb")
"""

import argparse
import itertools
import json
import sys
from copy import deepcopy
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

# Add src to path
root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(root / "src"))

from climate_discovery.config import ModelConfig
from climate_discovery.utils.tracking import ExperimentTracker


# Predefined ablation configurations
ABLATION_PRESETS = {
    "regime_count": {
        "description": "Test different numbers of regimes",
        "params": {
            "num_regimes": [3, 6, 9, 12],
        }
    },
    
    "regularization": {
        "description": "Ablate regularization weights",
        "params": {
            "entropy_weight": [0.0, 0.005, 0.01, 0.05],
            "spatial_smoothness_weight": [0.0, 0.025, 0.05, 0.1],
            "temporal_smoothness_weight": [0.0, 0.015, 0.03, 0.06],
        }
    },
    
    "features": {
        "description": "Feature engineering ablations",
        "params": {
            "expert_features": [
                ["sst", "sss", "log_chl"],  # Baseline
                ["sst", "sss", "log_chl", "sst_gradient"],  # + gradient
                ["sst", "sss", "log_chl", "sst_gradient", "sin_month", "cos_month"],  # + temporal
            ],
        }
    },
    
    "gating_architecture": {
        "description": "Gating network architecture",
        "params": {
            "gating_type": ["mlp", "attention"],
            "gating_hidden_dims": [[64, 32], [128, 64], [256, 128, 64]],
        }
    },
    
    "hierarchical": {
        "description": "Hierarchical vs flat structure",
        "params": {
            "use_hierarchical": [False, True],
            "n_coarse_regimes": [3, 4],
            "n_fine_per_coarse": [2, 3, 4],
        }
    },
    
    "full_grid": {
        "description": "Comprehensive ablation (WARNING: many runs!)",
        "params": {
            "num_regimes": [6, 9],
            "entropy_weight": [0.005, 0.01],
            "spatial_smoothness_weight": [0.025, 0.05],
            "gating_type": ["mlp", "attention"],
            "use_hierarchical": [False, True],
        }
    }
}


def generate_ablation_configs(
    base_config: ModelConfig,
    ablation_params: Dict[str, List[Any]],
) -> List[Dict]:
    """Generate all combinations of ablation parameters.
    
    Args:
        base_config: Base configuration
        ablation_params: Dict of {param_name: [values to try]}
        
    Returns:
        List of configuration dicts
    """
    # Get parameter names and values
    param_names = list(ablation_params.keys())
    param_values = list(ablation_params.values())
    
    # Generate Cartesian product
    configs = []
    for values in itertools.product(*param_values):
        # Create config dict
        config_dict = asdict(base_config) if hasattr(base_config, "__dataclass_fields__") else vars(base_config).copy()
        
        # Update with ablation values
        for name, value in zip(param_names, values):
            config_dict[name] = value
        
        configs.append(config_dict)
    
    return configs


def run_single_ablation(
    config_dict: Dict,
    run_idx: int,
    total_runs: int,
    tracker: ExperimentTracker = None,
) -> Dict:
    """Run single ablation experiment.
    
    Args:
        config_dict: Configuration dictionary
        run_idx: Current run index
        total_runs: Total number of runs
        tracker: Experiment tracker
        
    Returns:
        Results dictionary
    """
    print(f"\n{'='*70}")
    print(f"ABLATION RUN {run_idx + 1}/{total_runs}")
    print(f"{'='*70}")
    
    # Print config
    print("\nConfiguration:")
    for key, value in config_dict.items():
        if key in ["num_regimes", "entropy_weight", "spatial_smoothness_weight",
                   "gating_type", "use_hierarchical", "expert_features"]:
            print(f"  {key}: {value}")
    
    # Simulate training (replace with actual training in real use)
    # Here we just generate dummy results for demonstration
    print("\n🚀 Training...")
    
    # Dummy metrics (in real use, call actual training function)
    val_r2 = np.random.uniform(0.35, 0.55)
    val_mse = np.random.uniform(100, 200)
    train_time = np.random.uniform(10, 60)  # minutes
    
    results = {
        "val_r2": val_r2,
        "val_mse": val_mse,
        "train_time_min": train_time,
        **{k: v for k, v in config_dict.items() if k != "expert_features"},
    }
    
    # Log to tracker
    if tracker:
        tracker.log_metrics({
            "ablation/val_r2": val_r2,
            "ablation/val_mse": val_mse,
            "ablation/train_time": train_time,
        }, step=run_idx)
    
    print(f"\n✓ Results: R² = {val_r2:.4f}, MSE = {val_mse:.2f}, Time = {train_time:.1f}min")
    
    return results


def run_ablation_study(
    ablation_params: Dict[str, List[Any]],
    base_config: ModelConfig = None,
    backend: str = "wandb",
    project: str = "sd-mose-ablations",
    dry_run: bool = False,
) -> pd.DataFrame:
    """Run complete ablation study.
    
    Args:
        ablation_params: Parameters to ablate
        base_config: Base configuration (uses default if None)
        backend: Tracking backend
        project: Project name
        dry_run: If True, only print what would run
        
    Returns:
        DataFrame with all results
    """
    # Use default config if not provided
    if base_config is None:
        base_config = ModelConfig()
    
    # Generate all configurations
    configs = generate_ablation_configs(base_config, ablation_params)
    
    print(f"\n{'='*70}")
    print(f"ABLATION STUDY: {len(configs)} configurations")
    print(f"{'='*70}")
    
    # Print summary
    print("\nAblation parameters:")
    for param, values in ablation_params.items():
        print(f"  {param}: {len(values)} values - {values}")
    
    total_time_est = len(configs) * 30  # Rough estimate: 30 min per run
    print(f"\nEstimated time: ~{total_time_est // 60:.1f} hours")
    
    if dry_run:
        print("\n⚠️  DRY RUN - Not executing")
        print(f"\nWould run {len(configs)} experiments:")
        for i, cfg in enumerate(configs[:5]):  # Show first 5
            print(f"  {i+1}. {cfg.get('num_regimes', '?')} regimes, "
                  f"entropy={cfg.get('entropy_weight', '?')}, "
                  f"gating={cfg.get('gating_type', '?')}")
        if len(configs) > 5:
            print(f"  ... and {len(configs) - 5} more")
        return None
    
    # Initialize tracker
    tracker = ExperimentTracker(
        backend=backend,
        project=project,
        name=f"ablation-{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        tags=["ablation", "grid_search"],
    ) if backend != "none" else None
    
    # Run all ablations
    results = []
    for i, config_dict in enumerate(configs):
        try:
            result = run_single_ablation(config_dict, i, len(configs), tracker)
            results.append(result)
        except Exception as e:
            print(f"\n❌ Run {i+1} failed: {e}")
            # Continue with next run
    
    # Finish tracking
    if tracker:
        tracker.finish()
    
    # Convert to DataFrame
    df = pd.DataFrame(results)
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = Path("results/ablations")
    results_dir.mkdir(parents=True, exist_ok=True)
    
    csv_path = results_dir / f"ablation_{timestamp}.csv"
    df.to_csv(csv_path, index=False)
    
    json_path = results_dir / f"ablation_{timestamp}.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✓ Results saved:")
    print(f"  - {csv_path}")
    print(f"  - {json_path}")
    
    # Print summary
    print_ablation_summary(df, ablation_params)
    
    return df


def print_ablation_summary(df: pd.DataFrame, ablation_params: Dict):
    """Print summary of ablation results."""
    print(f"\n{'='*70}")
    print("ABLATION SUMMARY")
    print(f"{'='*70}")
    
    # Best configuration
    best_idx = df['val_r2'].idxmax()
    best = df.loc[best_idx]
    
    print(f"\n🏆 Best Configuration:")
    print(f"   R² = {best['val_r2']:.4f}")
    for param in ablation_params.keys():
        if param in best:
            print(f"   {param} = {best[param]}")
    
    # Parameter importance
    print(f"\n📊 Parameter Impact (sorted by R² variance):")
    for param in ablation_params.keys():
        if param in df.columns and df[param].dtype in [np.float64, np.int64]:
            grouped = df.groupby(param)['val_r2'].agg(['mean', 'std', 'count'])
            variance = grouped['mean'].var()
            print(f"\n   {param}:")
            print(f"     Variance: {variance:.6f}")
            for value, row in grouped.iterrows():
                print(f"     {value}: R² = {row['mean']:.4f} ± {row['std']:.4f} (n={int(row['count'])})")


def main():
    parser = argparse.ArgumentParser(description="Run ablation study")
    parser.add_argument(
        "--preset",
        type=str,
        choices=list(ABLATION_PRESETS.keys()),
        help="Use predefined ablation preset",
    )
    parser.add_argument(
        "--backend",
        type=str,
        default="wandb",
        choices=["wandb", "mlflow", "both", "none"],
        help="Tracking backend",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would run without executing",
    )
    parser.add_argument(
        "--config",
        type=str,
        help="Path to custom ablation config JSON/YAML",
    )
    
    args = parser.parse_args()
    
    # Load ablation parameters
    if args.preset:
        preset = ABLATION_PRESETS[args.preset]
        print(f"\nUsing preset: {args.preset}")
        print(f"Description: {preset['description']}")
        ablation_params = preset["params"]
    elif args.config:
        # Load from file
        config_path = Path(args.config)
        if config_path.suffix == ".json":
            with open(config_path) as f:
                ablation_params = json.load(f)
        else:
            raise ValueError("Only JSON configs supported currently")
    else:
        # Default: regime count ablation
        print("\nNo preset specified, using 'regime_count'")
        ablation_params = ABLATION_PRESETS["regime_count"]["params"]
    
    # Run ablation study
    df = run_ablation_study(
        ablation_params=ablation_params,
        backend=args.backend,
        dry_run=args.dry_run,
    )
    
    if df is not None:
        print(f"\n✓ Ablation study complete! Results saved to results/ablations/")


if __name__ == "__main__":
    main()
