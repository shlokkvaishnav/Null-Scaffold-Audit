"""
Run an ensemble of gating networks with different random seeds.
"""

import os
import subprocess
from pathlib import Path

ENSEMBLE_SEEDS = [0, 1, 2, 3, 4]

ROOT = Path(__file__).resolve().parents[2]
CHECKPOINT_ROOT = ROOT / "checkpoints" / "ensemble"
CHECKPOINT_ROOT.mkdir(parents=True, exist_ok=True)

for seed in ENSEMBLE_SEEDS:
    print(f"\n=== Training ensemble member (seed={seed}) ===")

    env = os.environ.copy()
    env["PYTHONHASHSEED"] = str(seed)

    out_dir = CHECKPOINT_ROOT / f"seed_{seed}"
    out_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        "python",
        "-m",
        "scripts.train.train_gating",
        "--seed",
        str(seed),
        "--out_dir",
        str(out_dir),
    ]

    subprocess.run(cmd, env=env, check=True)

print("\nAll ensemble members trained successfully.")