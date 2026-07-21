# Scientific Discovery Engine (SDE)

SDE is a small, domain-agnostic engine (`engine/`) that orchestrates a fixed
workflow — load data, split, fit, predict, validate, score — while
delegating every step that touches actual scientific content to a plugin.
The engine doesn't know what a "physics equation" is; it only knows the
shapes of a dataset and of an algorithm/domain plugin.

**`physics_discovery/`** is the first plugin: an agentic system that
automates the scientist's hypothesize → test → refine loop for closed-form
equation discovery, using symbolic regression as the hypothesis-generation
engine. Given any tabular dataset (features → target), it proposes candidate
equations, scores them for fit and validity, tracks confidence across
competing hypotheses, and iteratively refines until the hypothesis set
converges. It's evaluated against the **AI-Feynman symbolic regression
benchmark** — 36 real physics equations with known ground truth — reporting
equation-rediscovery rate alongside predictive accuracy (RMSE) against
standard ML baselines (gradient-boosted trees, small neural ensembles).

Read [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for how the engine and
plugin fit together, and [docs/PLUGIN_GUIDE.md](docs/PLUGIN_GUIDE.md) if
you're adding a new algorithm or scientific domain.

## Quick start

```bash
docker compose up --build
```

No local Python install, no host `pip install` — everything (symbolic
regression backend, ML baselines, API server, CLI) runs inside the
container. The physics_discovery API is then available at
`http://localhost:8000` (interactive docs at `/docs`).

```bash
docker compose run --rm api pytest tests/ -v
docker compose run --rm api sde list-plugins
docker compose run --rm api sde run configs/run/coulomb_symbolic.yaml
```

See [docs/DEVELOPMENT.md](docs/DEVELOPMENT.md) for local (non-Docker) setup,
the full test/CI breakdown, and reproducing paper benchmark results.

## The `sde` CLI

```bash
sde list-plugins                                    # what's registered
sde run configs/run/coulomb_symbolic.yaml           # run one domain+algorithm plugin pair
sde benchmark configs/paper/benchmark_minimal.yaml  # validated paper-style multi-model comparison
```

## physics_discovery API

The FastAPI service is a separate, direct way to drive the physics_discovery
plugin (upload a CSV, get back a discovered equation) — independent of the
engine/CLI path above.

| Endpoint | Description |
|---|---|
| `POST /datasets` | Upload a CSV (multipart `file`, optional `target_column` form field, defaults to the last column) |
| `GET /datasets/{id}` | Dataset metadata (feature names, row count) |
| `POST /jobs` | Submit a discovery job for a dataset (runs in the background) |
| `GET /jobs/{id}` | Poll job status; once done, returns the discovered equation, RMSE, and a confidence score |
| `GET /benchmark/feynman?subset=smoke` | Run a quick Feynman rediscovery check synchronously and return the results table |
| `GET /health` | Liveness check |

## Repository layout

```
engine/             the platform: plugin contract, registry, orchestrator, config schemas
cli/                 the `sde` command-line entry point
physics_discovery/  the first plugin: physics/Feynman equation-discovery agent + FastAPI service
configs/            run/ (single orchestrator runs) and paper/ (benchmark comparison tables)
scripts/             standalone entry points (reproduce_benchmarks, run_ablations, ...)
tests/               pytest suite
docs/                architecture, plugin guide, development guide
```

Full breakdown in [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

## License

MIT License — see [LICENSE](LICENSE) for details.
