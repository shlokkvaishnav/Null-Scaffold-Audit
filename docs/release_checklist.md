# Release Checklist

Use this checklist before a release freeze to ensure benchmark artifacts are reproducible and sanity checks pass.

## 0) Environment prep

```bash
pip install -r requirements.txt
```

## 1) Regenerate all benchmark results

### Minimal mode

```bash
python scripts/reproduce_benchmarks.py --config configs/paper/benchmark_minimal.yaml
```

**Expected outputs**
- `results/reproducibility/benchmark_minimal_results.json`
- `results/reproducibility/runtime_budget_table.csv`

**Approximate runtime**
- ~2 seconds on this repo's default environment

### Full mode

```bash
python scripts/reproduce_benchmarks.py --config configs/paper/benchmark_full.yaml
```

**Expected outputs**
- `results/reproducibility/benchmark_full_results.json`
- `results/reproducibility/runtime_budget_table.csv` (overwritten with full-budget row)

**Approximate runtime**
- ~2 seconds on this repo's default environment

## 2) Sanity checks (pass/fail gates)

### A. Artifact existence check

```bash
test -f results/reproducibility/benchmark_minimal_results.json \
  -a -f results/reproducibility/benchmark_full_results.json \
  -a -f results/reproducibility/runtime_budget_table.csv
```

- **Pass**: command exits with status `0`
- **Fail**: any expected file is missing

### B. Result schema check

```bash
python - <<'PY'
import json
from pathlib import Path

paths = [
    Path('results/reproducibility/benchmark_minimal_results.json'),
    Path('results/reproducibility/benchmark_full_results.json'),
]
required_top = {'experiment', 'config', 'aggregate', 'runs', 'search_protocol'}

for p in paths:
    data = json.loads(p.read_text())
    missing = required_top - set(data)
    if missing:
        raise SystemExit(f'{p}: missing keys {sorted(missing)}')
    if not data['runs']:
        raise SystemExit(f'{p}: runs is empty')
print('schema checks passed')
PY
```

- **Pass**: prints `schema checks passed` and exits `0`
- **Fail**: missing keys, empty `runs`, or JSON parse errors

### C. Smoke test for agent pipeline

```bash
PYTHONPATH=. python scripts/test_agent.py
```

- **Pass**: exits `0` and prints `[SUCCESS] Agent architecture test complete!`
- **Fail**: any traceback or nonzero exit code

## 3) Freeze policy (tested-at-least-once requirement)

Before freeze, record one successful run for **every command above** in the release notes or CI logs.
Use this table as a sign-off template:

| Command | Last run (UTC) | Operator | Result |
|---|---|---|---|
| `python scripts/reproduce_benchmarks.py --config configs/paper/benchmark_minimal.yaml` |  |  |  |
| `python scripts/reproduce_benchmarks.py --config configs/paper/benchmark_full.yaml` |  |  |  |
| `test -f results/reproducibility/benchmark_minimal_results.json -a -f results/reproducibility/benchmark_full_results.json -a -f results/reproducibility/runtime_budget_table.csv` |  |  |  |
| `python - <<'PY' ...` (schema check block above) |  |  |  |
| `PYTHONPATH=. python scripts/test_agent.py` |  |  |  |

Release freeze is blocked if any command has not been run successfully at least once.
