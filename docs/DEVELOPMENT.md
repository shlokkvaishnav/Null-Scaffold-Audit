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
docker compose run --rm api python -m plugins.physics.benchmark_runner --subset smoke
```

**Local (without Docker):**

```bash
pip install -e ".[dev,gbm]"
pytest tests/ -v
sde list-plugins
```

> **Plugin discovery requires an actual install.** `sde list-plugins` (and
> `_default_registry()` generally) discovers plugins via the `"sde.plugins"`
> Python packaging entry-point group (`engine/discovery.py`) — entry points
> are metadata written at install time, not something importable from a bare
> source checkout. Skip `pip install -e .` (e.g. because you're only running
> from source and verifying everything via Docker/CI instead) and
> `sde list-plugins` will report an empty registry, not an error. CI and the
> Docker image both run `pip install`, so this only affects ad hoc local
> runs against an uninstalled checkout.

> **Note on sandboxed/locked-down environments:** some Windows environments
> under an Application Control / AppLocker-style policy block `sklearn`'s
> compiled extensions at import time (`DLL load failed ... An Application
> Control policy has blocked this file`), which cascades into anything that
> imports `engine.evaluation.metrics` (and thus most of the
> Feynman plugin). This is an environment restriction, not a project bug --
> `tests/test_api_endpoints.py` and `tests/test_feynman_plugin.py` both guard
> against it. If you hit this, verify via Docker (Linux, unaffected) instead
> of chasing it locally.

## Project layout

```
engine/             the platform: plugin contract, registry, orchestrator, config schemas
cli/                 the `sde` command-line entry point
plugins/physics/    the first domain plugin: the DiscoveryAgent loop, Feynman data, FastAPI service
configs/run/         engine.config.RunConfig YAML files (one orchestrator run)
scripts/             the audit runner, the selection-ceiling runner, and the summariser
tests/               pytest suite -- engine tests use fakes, plugin tests use real code
docs/                this directory
```

See [ARCHITECTURE.md](ARCHITECTURE.md) for how the pieces fit together and
[PLUGIN_GUIDE.md](PLUGIN_GUIDE.md) for adding a new algorithm or domain.

## Tests

```bash
pytest tests/ -v                                  # everything
pytest tests/test_engine_orchestrator.py -v       # engine seams only (fake plugins, no ML deps)
pytest tests/test_engine_discovery.py -v          # entry-point discovery mechanism (mocked, no install needed)
pytest tests/test_feynman_plugin.py -v            # the real plugin, through the orchestrator
pytest tests/test_synthetic_plugin.py -v          # the second (synthetic) plugin
pytest tests/test_cli.py -v                       # the `sde` CLI (requires `pip install -e .` -- see note above)
```

## CI (`.github/workflows/ci.yml`)

Four jobs, all on `ubuntu-latest`:

| Job | What it does |
|---|---|
| `lint` | `ruff check .` |
| `typecheck` | `mypy engine algorithms validators plugins cli` |
| `test` | `pytest tests/ -v` |
| `smoke-benchmark` | `python -m plugins.physics.benchmark_runner --subset smoke --backend gplearn` |

Two other workflows: `docker-build.yml` (builds the image on push/PR) and
`full-benchmark.yml` (the full, slower Feynman benchmark — not run on every
PR).

## Reproducing the audit results

```bash
python scripts/measure_selection_ceiling.py --domain physics   --scaffold plugins.physics.audit_adapter:DiscoveryAgentScaffold --subset all --seeds 10

python scripts/run_null_scaffold_audit.py --domain physics   --scaffold plugins.physics.audit_adapter:DiscoveryAgentScaffold --subset smoke --seeds 20

python scripts/summarize_audit.py
```

Run the ceiling first. It costs one arm and reports, per problem, whether any
selection-only wrapper could clear the pre-registered margin at all -- so it can
rule a problem out before a two-arm sweep spends hours discovering the same thing
one INCONCLUSIVE at a time.

Margins are pre-registered at the top of `scripts/run_null_scaffold_audit.py`
and are not to be adjusted after seeing an interval. Artifacts land in
`results/null_scaffold_audit/` and `results/selection_ceiling/`, and those are
tracked in git: every number this project reports is read from them.

A `reproduce_benchmarks.py` / `configs/paper/` path used to be documented here.
It was removed along with the paper framing it served; see ARCHITECTURE.md.

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
`engine`, `cli`, `algorithms`, `validators`, `plugins` today), add its glob to
`[tool.setuptools.packages.find].include` and to the relevant `COPY` lines in
`docker/Dockerfile` (both the `builder` and `runtime` stages).

To add a *plugin* rather than a dependency of this repo, you generally don't
edit this repo at all -- see [PLUGIN_GUIDE.md](PLUGIN_GUIDE.md). This repo's
own plugins are declared as `"sde.plugins"` entry points in this
`pyproject.toml`; that table is the one place to touch if you're adding a
plugin *inside* this repo rather than in a separate installable package.
