"""Run all SD-MoSE scripts sequentially in the correct order.

Usage:
    python -m scripts.run_all                    # Run everything
    python -m scripts.run_all --skip-download    # Skip data download
    python -m scripts.run_all --start-from train_gating  # Resume from specific step
    python -m scripts.run_all --only train       # Only run training scripts
"""

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

# Project root
root = Path(__file__).resolve().parents[1]
progress_file = root / ".pipeline_progress.json"

# Pipeline configuration
PIPELINE = [
    {
        "category": "data",
        "script": "download_data",
        "description": "Download SOCAT + Copernicus data",
        "time_estimate": "20 min",
        "skippable": True,
    },
    {
        "category": "data",
        "script": "preprocess_data",
        "description": "Preprocess and feature engineering",
        "time_estimate": "5 min",
        "skippable": False,
    },
    {
        "category": "eval",
        "script": "eval_baselines",
        "description": "Evaluate baseline models",
        "time_estimate": "30 min",
        "skippable": True,
    },
    {
        "category": "train",
        "script": "train_gating",
        "description": "Train gating network",
        "time_estimate": "1 hour",
        "skippable": False,
    },
    {
        "category": "train",
        "script": "discover_laws",
        "description": "Discover symbolic laws with PySR",
        "time_estimate": "2 hours",
        "skippable": False,
    },
    {
        "category": "train",
        "script": "train_sdmose",
        "description": "Full SD-MoSE alternating optimization",
        "time_estimate": "4 hours",
        "skippable": True,
    },
    {
        "category": "eval",
        "script": "eval_mixture",
        "description": "Evaluate SD-MoSE mixture",
        "time_estimate": "5 min",
        "skippable": True,
    },
    {
        "category": "eval",
        "script": "eval_ablations",
        "description": "Run ablation studies",
        "time_estimate": "3 hours",
        "skippable": True,
    },
    {
        "category": "viz",
        "script": "plot_regimes",
        "description": "Generate publication figures",
        "time_estimate": "10 min",
        "skippable": True,
    },
]


def save_progress(step_index: int, status: str):
    """Save pipeline progress for resumption."""
    progress = {
        "last_completed_step": step_index,
        "status": status,
        "timestamp": datetime.now().isoformat(),
    }
    with open(progress_file, "w") as f:
        json.dump(progress, f, indent=2)


def load_progress():
    """Load previous pipeline progress."""
    if progress_file.exists():
        with open(progress_file, "r") as f:
            return json.load(f)
    return None


def run_script(category: str, script: str, root_dir: Path) -> bool:
    """Run a single pipeline script.
    
    Returns:
        True if successful, False otherwise
    """
    module_name = f"scripts.{category}.{script}"
    
    print(f"\n{'='*60}")
    print(f"▶️  Running: {module_name}")
    print("=" * 60)
    
    start_time = time.time()
    
    result = subprocess.run(
        [sys.executable, "-m", module_name],
        cwd=str(root_dir),
    )
    
    elapsed = time.time() - start_time
    elapsed_str = f"{int(elapsed // 60)}m {int(elapsed % 60)}s"
    
    if result.returncode != 0:
        print(f"\n❌ FAILED: {module_name} (exit code {result.returncode})")
        print(f"   Time: {elapsed_str}")
        return False
    
    print(f"\n✅ SUCCESS: {module_name}")
    print(f"   Time: {elapsed_str}")
    return True


def print_pipeline_summary(pipeline: list, start_from: int = 0):
    """Print summary of what will be run."""
    print("\n" + "=" * 60)
    print("PIPELINE SUMMARY")
    print("=" * 60)
    
    total_time = 0
    for i, step in enumerate(pipeline[start_from:], start=start_from):
        status = "⏭️  SKIP" if i < start_from else "▶️  RUN"
        print(f"{i+1:2d}. [{status}] {step['category']:5s}/{step['script']:20s}")
        print(f"     {step['description']}")
        print(f"     Est. time: {step['time_estimate']}")
    
    print("\n" + "=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="Run SD-MoSE pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m scripts.run_all                          # Full pipeline
  python -m scripts.run_all --skip-download          # Skip data download
  python -m scripts.run_all --start-from train_gating  # Resume from step
  python -m scripts.run_all --only train             # Only training
  python -m scripts.run_all --only eval              # Only evaluation
        """
    )
    
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Skip data download (assumes data already exists)"
    )
    parser.add_argument(
        "--start-from",
        type=str,
        help="Resume from specific script (e.g., 'train_gating')"
    )
    parser.add_argument(
        "--only",
        type=str,
        choices=["data", "train", "eval", "viz"],
        help="Only run scripts from this category"
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from last failed step"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be run without executing"
    )
    
    args = parser.parse_args()
    
    # Build execution list
    to_execute = PIPELINE.copy()
    
    # Filter by category
    if args.only:
        to_execute = [s for s in to_execute if s["category"] == args.only]
    
    # Skip download
    if args.skip_download:
        to_execute = [s for s in to_execute if s["script"] != "download_data"]
    
    # Resume from checkpoint
    start_index = 0
    if args.resume:
        progress = load_progress()
        if progress:
            start_index = progress["last_completed_step"] + 1
            print(f"\n📍 Resuming from step {start_index + 1}")
    
    # Start from specific script
    if args.start_from:
        for i, step in enumerate(to_execute):
            if step["script"] == args.start_from:
                start_index = i
                print(f"\n📍 Starting from: {args.start_from}")
                break
        else:
            print(f"❌ Script '{args.start_from}' not found in pipeline")
            sys.exit(1)
    
    # Filter execution list
    to_execute = to_execute[start_index:]
    
    if not to_execute:
        print("❌ No scripts to run!")
        sys.exit(1)
    
    # Print summary
    print("\n" + "=" * 60)
    print("SD-MoSE PIPELINE")
    print("=" * 60)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Scripts to run: {len(to_execute)}")
    print("=" * 60)
    
    print_pipeline_summary(PIPELINE, start_from=start_index)
    
    if args.dry_run:
        print("\n🔍 DRY RUN - No scripts executed")
        sys.exit(0)
    
    # Confirm if long pipeline
    if len(to_execute) > 5:
        response = input("\nThis will take several hours. Continue? (y/n): ")
        if response.lower() != 'y':
            print("Cancelled")
            sys.exit(0)
    
    # Execute pipeline
    pipeline_start = time.time()
    completed = 0
    
    for i, step in enumerate(to_execute, start=start_index):
        success = run_script(step["category"], step["script"], root)
        
        if success:
            completed += 1
            save_progress(i, "completed")
        else:
            save_progress(i, "failed")
            
            print("\n" + "=" * 60)
            print("❌ PIPELINE FAILED")
            print("=" * 60)
            print(f"Failed at step {i+1}: {step['script']}")
            print(f"Completed: {completed}/{len(to_execute)} scripts")
            print("\nTo resume:")
            print(f"  python -m scripts.run_all --resume")
            print(f"  python -m scripts.run_all --start-from {step['script']}")
            sys.exit(1)
    
    # Success summary
    pipeline_time = time.time() - pipeline_start
    hours = int(pipeline_time // 3600)
    minutes = int((pipeline_time % 3600) // 60)
    
    print("\n" + "=" * 60)
    print("✅ PIPELINE COMPLETED SUCCESSFULLY!")
    print("=" * 60)
    print(f"Completed: {completed}/{len(to_execute)} scripts")
    print(f"Total time: {hours}h {minutes}m")
    print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("\n📊 Check results:")
    print(f"  - Figures: {root / 'figures'}")
    print(f"  - Results: {root / 'results'}")
    print(f"  - Checkpoints: {root / 'checkpoints'}")
    print("=" * 60)
    
    # Clean up progress file
    if progress_file.exists():
        progress_file.unlink()


if __name__ == "__main__":
    main()