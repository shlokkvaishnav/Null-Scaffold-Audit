"""Run all SD-MoSE scripts sequentially in the correct order."""

import subprocess
import sys
from pathlib import Path

# __file__ is scripts/run_all.py -> parents[1] is project root
root = Path(__file__).resolve().parents[1]

# List of (category, script_name)
scripts = [
    ("data", "download_data"),
    ("data", "preprocess_data"),
    ("eval", "eval_baselines"),
    ("train", "train_gating"),
    ("train", "discover_laws"),
    ("eval", "eval_mixture"),
    ("train", "train_sdmose"),
    ("eval", "eval_ablations"),
    ("viz", "plot_regimes"),
]

if __name__ == "__main__":
    print("=" * 60)
    print("SD-MoSE: Running all scripts sequentially")
    print("=" * 60)

    for category, script in scripts:
        # Construct module path: scripts.data.download_data
        module_name = f"scripts.{category}.{script}"

        print(f"\n{'='*60}")
        print(f"Running module: {module_name}")
        print("=" * 60)

        # Run as a module (-m) to ensure 'src' imports resolve correctly
        result = subprocess.run(
            [sys.executable, "-m", module_name],
            cwd=str(root),
        )

        if result.returncode != 0:
            print(f"\n❌ Failed: {module_name} (exit code {result.returncode})")
            print("Stopping pipeline.")
            sys.exit(result.returncode)

        print(f"✅ Completed: {module_name}")

    print("\n" + "=" * 60)
    print("✅ All scripts completed successfully!")
    print("=" * 60)
