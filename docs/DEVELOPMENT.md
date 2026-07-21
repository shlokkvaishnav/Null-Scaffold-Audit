# Development

## Setup

**Docker (recommended, no local Python needed):**

```bash
docker compose up --build
```

Runs the FastAPI service at `http://localhost:8000` (interactive docs at
`/docs`). Everything — symbolic regression backend, ML baselines, the CLI —
runs inside the container.

```bash
docker compose run --rm api pytest tests/ -v
docker compose run --rm api sde list-plugins
docker compose run --rm api sde run configs/run/coulomb_symbolic.yaml
docker compose run --rm api sde benchmark configs/paper/benchmark_minimal.yaml
docker compose run --rm api python -m physics_discovery.evaluation.benchmark_runner --subset smoke
```

**Local (without Docker):**

```bash
pip install -e ".[dev,gbm]"
pytest tests/ -v
sde list-plugins
```

> **Note on sandboxed/locked-down environments:** some Windows environments
> under an Application Control / AppLocker-style policy block `sklearn`'s
> compiled extensions at import time (`DLL load failed ... An Application
> Control policy has blocked this file`), which cascades into anything that
> imports `physics_discovery.evaluation.metrics` (and thus most of the
> Feynman plugin). This is an environment restriction, not a project bug --
> `tests/test_api_endpoints.py` and `tests/test_feynman_plugin.py` both guard
> against it (the latter via `pytest.importorskip("sklearn.metrics")`). If
> you hit this, verify via Docker (Linux, unaffected) instead of chasing it
> locally.

## Project layout

```
engine/             the platform: plugin contract, registry, orchestrator, config schemas
cli/                 the `sde` command-line entry point
physics_discovery/  the first plugin: physics/Feynman equation-discovery agent + FastAPI service
configs/
  run/               engine.config.RunConfig YAML files (one orchestrator run)
  paper/             engine.config.BaselineExperimentConfig YAML files (paper benchmark tables)
scripts/             standalone entry points (reproduce_benchmarks, run_ablations, run_baselines, ...)
tests/               pytest suite -- engine tests use fakes, physics_discovery/plugin tests use real code
docs/                this directory
```

See [ARCHITECTURE.md](ARCHITECTURE.md) for how the pieces fit together and
[PLUGIN_GUIDE.md](PLUGIN_GUIDE.md) for adding a new algorithm or domain.

## Tests

```bash
pytest tests/ -v                                  # everything
pytest tests/test_engine_orchestrator.py -v       # engine seams only (fast, no ML deps)
pytest tests/test_feynman_plugin.py -v            # the real plugin, through the orchestrator
pytest tests/test_cli.py -v                       # the `sde` CLI
```

## CI (`.github/workflows/ci.yml`)

Four jobs, all on `ubuntu-latest`:

| Job | What it does |
|---|---|
| `lint` | `ruff check .` |
| `typecheck` | `mypy physics_discovery engine` |
| `test` | `pytest tests/ -v` |
| `smoke-benchmark` | `python -m physics_discovery.evaluation.benchmark_runner --subset smoke --backend gplearn` |

Two other workflows: `docker-build.yml` (builds the image on push/PR) and
`full-benchmark.yml` (the full, slower Feynman benchmark — not run on every
PR).

## Reproducing paper benchmark results

```bash
python scripts/reproduce_benchmarks.py --config configs/paper/benchmark_minimal.yaml
# or, equivalently, via the CLI once configs pass validation:
sde benchmark configs/paper/benchmark_minimal.yaml
```

Use `configs/paper/benchmark_full.yaml` for the full 10-seed run. Both
compute real metrics (RMSE, MAE, calibration error, symbolic complexity) from
actual model fits against `physics_discovery/experiments/contract.py`'s
shared baseline contract — no placeholder numbers. Results land in
`results/reproducibility/`.

## Symbolic regression backend

Symbolic regression defaults to **`gplearn`** (pure Python, no external
runtime dependency) both in Docker and CI. `PySR` is available as an opt-in
backend (`SYMBOLIC_BACKEND=pysr`, or pass `backend="pysr"` in an
`AlgorithmPlugin`'s config) for higher-quality search, but it depends on a
Julia runtime and a longer image rebuild — not the default, to keep
`docker compose up` fast and reliable.

## Adding a dependency

Add it to `dependencies` (always needed) or the appropriate
`[project.optional-dependencies]` extra (`pysr`, `torch`, `gbm`, `dev`) in
`pyproject.toml`. If a new top-level package needs to be importable (like
`engine`, `cli`, `physics_discovery` today), add its glob to
`[tool.setuptools.packages.find].include` and to the relevant `COPY` lines in
`Dockerfile` (both the `builder` and `runtime` stages).
