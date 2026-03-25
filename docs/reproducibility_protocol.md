# Reproducibility Protocol

## Deterministic seed policy
- Default seed list: `[7, 11, 23, 47, 101]`
- Extended list for final runs: add `[131, 181, 223, 269, 307]`
- Seed all RNGs used by Python, NumPy, and framework backends.

## Fixed config files
All table/figure jobs are pinned in `configs/paper/`.

## One-script reproduction
```bash
python scripts/reproduce_benchmarks.py --config configs/paper/benchmark_minimal.yaml
```

## Runtime budget table
Provide both:
- small reproducible budget (`benchmark_minimal.yaml`)
- full budget (`benchmark_full.yaml`)

## Hyperparameter search protocol
Log:
- search space,
- sampled/tried configuration IDs,
- objective metric,
- selected config and rationale.

Artifacts are written to `results/reproducibility/`.
