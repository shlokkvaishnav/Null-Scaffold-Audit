# Equation Discovery Agent

An agentic system that automates the scientist's hypothesize → test → refine loop for closed-form equation discovery, using symbolic regression as the hypothesis-generation engine. Given any tabular dataset (features → target), the agent proposes candidate equations, scores them for fit and validity, tracks confidence across competing hypotheses, and iteratively refines until the hypothesis set converges.

The system is evaluated against the **AI-Feynman symbolic regression benchmark** — a set of real physics equations with known ground truth — reporting equation-rediscovery rate alongside predictive accuracy (RMSE) against standard ML baselines (gradient-boosted trees, small neural ensembles). It's exposed as a small FastAPI service (submit a dataset, get back a discovered equation) and ships fully containerized.

## Quick start

The only setup step is:

```bash
docker compose up --build
```

No local Python install, no host `pip install` — everything (symbolic regression backend, ML baselines, API server) runs inside the container. The API is then available at `http://localhost:8000` (interactive docs at `/docs`).

Run the test suite or the benchmark inside the container:

```bash
docker compose run --rm api pytest tests/ -v
docker compose run --rm api python -m equation_discovery.evaluation.benchmark_runner --subset smoke
```

## API

| Endpoint | Description |
|---|---|
| `POST /datasets` | Upload a CSV (multipart `file`, optional `target_column` form field, defaults to the last column) |
| `GET /datasets/{id}` | Dataset metadata (feature names, row count) |
| `POST /jobs` | Submit a discovery job for a dataset (runs in the background) |
| `GET /jobs/{id}` | Poll job status; once done, returns the discovered equation, RMSE, and a confidence score |
| `GET /benchmark/feynman?subset=smoke` | Run a quick Feynman rediscovery check synchronously and return the results table |
| `GET /health` | Liveness check |

## Architecture

```
Observe → Retrieve → Reason → Verify → Learn → Converge
  data     priors    generate  score/   update   stable
           +archive  hypothesis validate confidence hypothesis
                                                     set
```

- **`equation_discovery/core/`** — the agent loop itself: `DiscoveryAgent` orchestrates observation, hypothesis generation, scoring, and confidence tracking; `Hypothesis` is a first-class equation object (formula, fit score, complexity); `HypothesisScorer` combines data fit with validity/complexity penalties; `HypothesisArchive` retains the top hypotheses and prunes the rest; `ConfidenceTracker` (and its geodesic/factor-graph variants) tracks belief over competing hypotheses via Bayesian-style updates; `ConvergenceController` decides when the loop has stabilized.
- **`equation_discovery/generators/`** — hypothesis generators: symbolic regression (`gplearn` by default, optional `PySR` backend), gradient-boosted tree baselines, and a small neural ensemble.
- **`equation_discovery/validation/`** — `EquationValidator` rejects equations with structural issues (division by zero, log/sqrt of negatives); `LyapunovScreener` screens for dynamically unstable candidates via finite-difference Jacobian analysis.
- **`equation_discovery/data/`** — synthetic regression data, a CSV loader for arbitrary datasets, and a self-contained loader for 36 AI-Feynman physics equations (metadata committed as JSON, sample data generated deterministically at runtime — no network access required).
- **`equation_discovery/evaluation/`** — fit metrics, confidence intervals and paired significance tests, and symbolic/numeric equivalence checking against ground-truth equations (the rediscovery benchmark).
- **`equation_discovery/api/`** — the FastAPI service.

## Symbolic regression backend

Symbolic regression defaults to **`gplearn`** (pure Python, no external runtime dependency) both in Docker and CI. `PySR` is available as an opt-in backend (`SYMBOLIC_BACKEND=pysr`) for higher-quality search, but it depends on a Julia runtime and requires a longer image rebuild — not the default, to keep `docker compose up` fast and reliable.

## Development (without Docker)

```bash
pip install -e ".[dev,gbm]"
pytest tests/ -v
python -m equation_discovery.evaluation.benchmark_runner --subset smoke
```

## Reproducing benchmark results

```bash
python scripts/reproduce_benchmarks.py --config configs/paper/benchmark_minimal.yaml
```

Use `configs/paper/benchmark_full.yaml` for the full 10-seed run. Both compute real metrics (RMSE, MAE, calibration error, symbolic complexity) from actual model fits — no placeholder numbers.

## License

MIT License — see [LICENSE](LICENSE) for details.
