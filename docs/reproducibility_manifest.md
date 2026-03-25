# Reproducibility Manifest

This repository now includes runtime reproducibility artifacts and guardrails:

- Locked dependency snapshot: `requirements-lock.txt` (generated with `pip freeze`).
- Python version requirement: `>=3.9` (from `setup.py`).
- Runtime machine + accelerator metadata is logged at run start via `scripts/reproducibility.py`.
- CUDA/CPU notes are captured in each run manifest under `platform_hardware`:
  - `cuda_available`
  - `cuda_device_count`
  - `cuda_devices` (if available)

## Required deterministic config fields

Runs fail at startup unless these config fields exist and are valid:

- `seed_policy.deterministic` (must be `true`)
- `seed_policy.seeds` (must be a non-empty list)
- `deterministic_runtime.python_hash_seed` (int)
- `deterministic_runtime.torch_use_deterministic_algorithms` (bool)
- `deterministic_runtime.cudnn_deterministic` (bool)
- `deterministic_runtime.cudnn_benchmark` (bool)

## Startup logging

Runner scripts log a reproducibility manifest containing:

- Git commit hash
- Selected package versions
- Platform/hardware metadata
- Seed policy and deterministic runtime policy

For benchmark runs (`scripts/reproduce_benchmarks.py`), the manifest is also saved to:

- `results/reproducibility/<experiment_name>_manifest.json`
