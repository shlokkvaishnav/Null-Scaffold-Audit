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
  plugins/feynman.py   the adapter layer: wraps everything above as an AlgorithmPlugin/DomainPlugin
```

**Dependency direction is one-way**: `physics_discovery/plugins/feynman.py`
imports from `engine/`. Nothing in `engine/` or `cli/` imports from
`physics_discovery/` except by name, through the registry, at runtime. If you
ever see `engine/*.py` importing a concrete plugin module directly, that's a
bug — file it.

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

`_default_registry()` in `cli/main.py` is the one place that wires concrete
plugin modules (currently just `physics_discovery.plugins.feynman`) into the
CLI. Adding a second plugin means adding one `register(registry)` call there
— see [PLUGIN_GUIDE.md](PLUGIN_GUIDE.md).

## What's deliberately not built yet

Dynamic/sandboxed plugin loading, a report generator, a validation-strategy
plugin type (OOD/bootstrap/sensitivity), distributed execution, a second
real plugin. These are true next steps, not oversights — see
[PLUGIN_GUIDE.md](PLUGIN_GUIDE.md) for why a second plugin should land before
any of the interfaces above are treated as stable.
