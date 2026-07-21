# Architecture

## What this project is

**Scientific Discovery Engine (SDE)** is a small, domain-agnostic engine that
orchestrates a fixed workflow — *load data → split → fit → predict → validate
→ score* — while delegating every step that touches scientific content to a
plugin. The engine never knows what a "physics equation" or a "climate
variable" is; it only knows the shapes of `Dataset`, `AlgorithmPlugin`, and
`DomainPlugin`.

**`physics_discovery`** is the first plugin: an agentic equation-discovery
system (symbolic regression + a hypothesize→verify→learn loop) benchmarked
against 36 AI-Feynman physics equations. It used to be the entire project
(under the name `sdmose`, then `equation_discovery`). It is now one
implementation behind the engine's interfaces — real, working science code,
just no longer assumed to be the only domain the project will ever support.

If you're adding a second domain (chemistry, biology, a different physics
benchmark) or a second algorithm (PySR as its own plugin, a Bayesian search
method), you write it against the interfaces described here, not against
`physics_discovery` internals.

## The three packages

```
engine/             <- the platform. Domain-agnostic. Never imports a plugin.
  plugin.py           Dataset, AlgorithmPlugin, DomainPlugin (the contract)
  registry.py         PluginRegistry: name -> factory lookup
  discovery.py         discover_plugins: finds installed "sde.plugins" entry points, calls each one's register()
  orchestrator.py     ExperimentConfig, RunResult, DiscoveryOrchestrator
  config.py           pydantic schemas for YAML configs (RunConfig, BaselineExperimentConfig)

cli/                <- the one CLI entry point over the engine (`sde ...`)
  main.py             list-plugins / run / benchmark commands

physics_discovery/  <- the first plugin (physics/Feynman equation discovery)
  core/               DiscoveryAgent loop: perception, retrieval, belief, generator, scorer, archive
  generators/         SymbolicHypothesisGenerator, BaselineModel, Ensemble, SymbolicGate
  data/                feynman_loader, tabular CSV loader, synthetic regression data
  evaluation/          fit metrics, significance tests, symbolic/numeric equivalence checks
  validation/          EquationValidator, LyapunovStabilityScreener
  experiments/         the shared baseline experiment contract (see below)
  api/                 the FastAPI service (unrelated to engine/ — a separate way to drive physics_discovery directly)
  plugins/feynman.py    the adapter layer: wraps everything above as an AlgorithmPlugin/DomainPlugin
  plugins/synthetic.py  a second DomainPlugin (no ground truth), reusing feynman.py's algorithms
```

**Dependency direction is one-way**: `physics_discovery/plugins/feynman.py`
imports from `engine/`. Nothing in `engine/` or `cli/` imports from
`physics_discovery/`, or any other plugin package, **at all** — not even by
name. Plugins are found dynamically at runtime via `engine/discovery.py`
(Python packaging entry points, see below), not hardcoded into `cli/main.py`
or anywhere else. This is what makes the engine usable for a domain nobody
building this repo has thought of: a third party can add a plugin by
installing their own package, without a PR against this repo. If you ever
see `engine/*.py` or `cli/*.py` importing a concrete plugin module by name,
that's a bug — file it.

## The plugin contract (`engine/plugin.py`)

```python
class Dataset:
    X: np.ndarray
    y: np.ndarray
    feature_names: list[str]
    metadata: dict          # e.g. {"ground_truth": {...}} for benchmarks with known answers

class AlgorithmPlugin(Protocol):
    name: str
    def fit(self, X, y) -> "AlgorithmPlugin": ...
    def predict(self, X) -> np.ndarray: ...
    @property
    def equation(self) -> str | None: ...   # None is valid -- not every algorithm is symbolic

class DomainPlugin(Protocol):
    name: str
    def load_dataset(self, **kwargs) -> Dataset: ...
    def validate(self, equation: str | None) -> dict: ...        # constraint violations, {} if none/valid
    def score(self, y_true, y_pred, equation=None) -> dict[str, float]:  # fit-quality metrics
```

These are `typing.Protocol` classes (structural typing) — a plugin doesn't
need to inherit from anything, it just needs matching methods/attributes.
See `physics_discovery/plugins/feynman.py` for five real implementations
(`SymbolicRegressionAlgorithm`, `GBMBaselineAlgorithm`,
`NeuralEnsembleAlgorithm`, `DiscoveryAgentAlgorithm`, `FeynmanDomainPlugin`).

## The orchestrator (`engine/orchestrator.py`)

```python
registry = PluginRegistry()
feynman.register(registry)             # or any other module's register(registry)

orchestrator = DiscoveryOrchestrator(registry)
result = orchestrator.run(ExperimentConfig(
    domain="feynman_physics",
    algorithm="symbolic_regression",
    domain_kwargs={"equation_id": "coulomb_force", "n_samples": 200, "seed": 0},
    algorithm_kwargs={"backend": "gplearn", "generations": 10},
))
# result: RunResult(domain, algorithm, equation, metrics, constraints)
```

`DiscoveryOrchestrator.run` does exactly this and nothing else:
1. `registry.build_domain(...)` and `domain.load_dataset(...)`
2. split into train/test by `train_fraction`
3. `registry.build_algorithm(...)`, `.fit(...)`, `.predict(...)`
4. `domain.validate(algorithm.equation)`
5. `domain.score(y_test, y_pred, algorithm.equation)`

No science, no metric computation, no equation validity logic lives in
`engine/` — that discipline is what keeps a second plugin from requiring
engine changes.

## Two config shapes, on purpose (`engine/config.py`)

- **`RunConfig`** — one orchestrator run (one domain + one algorithm). See
  `configs/run/coulomb_symbolic.yaml`. This is what `sde run <path>` loads.
- **`BaselineExperimentConfig`** — mirrors the pre-existing
  `configs/paper/*.yaml` shape: a multi-model, multi-seed comparison with
  ablations and paired-significance testing, used for the actual paper
  benchmark tables (`scripts/reproduce_benchmarks.py`,
  `physics_discovery/experiments/contract.py`). `sde benchmark <path>`
  validates against `BaselineExperimentConfig` (which re-checks the same
  shared contract `validate_baseline_contract` already enforces) and then
  runs `scripts/reproduce_benchmarks.run(...)`.

These are deliberately *not* unified into one schema: they drive different
things (a single plugin run vs. a full paper-reproducibility comparison), and
forcing one shape onto both would make the paper pipeline's contract
(`physics_discovery/experiments/contract.py`) harder to reason about, not
easier.

## The CLI (`cli/main.py`)

```bash
sde list-plugins                                  # what's registered
sde run configs/run/coulomb_symbolic.yaml          # one orchestrator run
sde benchmark configs/paper/benchmark_minimal.yaml # paper-style comparison
```

`_default_registry()` in `cli/main.py` builds an empty `PluginRegistry` and
calls `engine.discovery.discover_plugins(registry)` — it does not import
`physics_discovery` (or anything else) by name. Discovery works via Python
packaging entry points: any installed package that declares one under the
`"sde.plugins"` group gets its `register(registry)` called automatically.
This repo's own two plugins are registered the same way, via entry points in
*this repo's* `pyproject.toml`:

```toml
[project.entry-points."sde.plugins"]
feynman_physics = "physics_discovery.plugins.feynman:register"
synthetic_regression = "physics_discovery.plugins.synthetic:register"
```

Adding a plugin from a separate package requires **no change to this repo**:
add the same kind of entry point to your own package's `pyproject.toml`,
`pip install` it, and `sde list-plugins` picks it up next run — see
[PLUGIN_GUIDE.md](PLUGIN_GUIDE.md). Entry points are packaging metadata, so
they only exist once a package is actually installed; see
[DEVELOPMENT.md](DEVELOPMENT.md) for what that means for local dev without
Docker.

## What's deliberately not built yet

Sandboxed/permissioned plugin loading (right now any installed entry point
is trusted and executed), a report generator, a validation-strategy plugin
type (OOD/bootstrap/sensitivity), distributed execution, a third real plugin
(e.g. PySR as its own `AlgorithmPlugin`). These are true next steps, not
oversights — see [PLUGIN_GUIDE.md](PLUGIN_GUIDE.md) for what the second
plugin (`synthetic_regression`) already validated about the interfaces, and
why that's a data point rather than a permanent guarantee.
