# Architecture

## What this project is

**Scientific Discovery Engine (SDE)** is a small, domain-agnostic engine that
orchestrates a fixed workflow — *load data → split → fit → predict → validate
→ score* — while delegating every step that touches scientific content to a
plugin. The engine never knows what a "physics equation" or a "climate
variable" is; it only knows the shapes of `Dataset`, `AlgorithmPlugin`, and
`DomainPlugin`.

**`plugins/physics/`** is the first plugin: an agentic equation-discovery
system (symbolic regression + a hypothesize→verify→learn loop) benchmarked
against 36 AI-Feynman physics equations. It used to be the entire project
(under the name `sdmose`, then `equation_discovery`). It is now one
implementation behind the engine's interfaces — real, working science code,
just no longer assumed to be the only domain the project will ever support.

If you're adding a second domain (chemistry, biology, a different physics
benchmark) or a second algorithm (PySR as its own plugin, a Bayesian search
method), you write it against the interfaces described here, not against
`plugins/physics/` internals.

## The packages

```
engine/             <- the platform. Domain-agnostic. Never imports a plugin.
  plugin.py           Dataset, AlgorithmPlugin, DomainPlugin (the contract)
  registry.py         PluginRegistry: name -> factory lookup
  discovery.py         discover_plugins: finds installed "sde.plugins" entry points, calls each one's register()
  orchestrator.py     ExperimentConfig, RunResult, DiscoveryOrchestrator
  config.py           pydantic schemas for YAML configs (RunConfig)
  audit/              null-scaffold audit: arms, statistics, degeneracy, AuditProblem
  expressions/        Hypothesis, safe evaluation of candidate equation strings
  evaluation/         fit metrics, SRBench-rule symbolic/numeric equivalence
  scoring.py          HypothesisScorer: ranks by fit + violations + complexity
  experiments/        the shared baseline experiment contract (see below)

cli/                <- the one CLI entry point over the engine (`sde ...`)
  main.py             list-plugins / run / benchmark commands

algorithms/         <- search algorithms behind AlgorithmPlugin (gplearn, sklearn)
  symbolic.py, ensemble.py, baselines.py

validators/         <- constraint rules behind ConstraintValidator
  equation_validity.py, dynamical_stability.py

plugins/            <- domain plugins; the engine may not import from here
  physics/            plugin.py (entry point), feynman_loader.py, feynman_equations.json
    scaffold/         DiscoveryAgent loop: perception, retrieval, belief, generator, archive
    audit_adapter.py  exposes that loop to engine.audit
    inference/        variational EM apparatus (needs torch)
    api/              the FastAPI service (a separate way to drive the loop)
  synthetic/          plugin.py, data.py — a second domain with a known formula
```

**Dependency direction is one-way**: `plugins/physics/plugin.py`
imports from `engine/`. Nothing in `engine/` or `cli/` imports from
`plugins/`, or any other plugin package, **at all** — not even by
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
See `plugins/physics/plugin.py` for five real implementations
(`SymbolicRegressionAlgorithm`, `GBMBaselineAlgorithm`,
`NeuralEnsembleAlgorithm`, `DiscoveryAgentAlgorithm`, `FeynmanDomainPlugin`).

## The orchestrator (`engine/orchestrator.py`)

```python
registry = PluginRegistry()
feynman.register(registry)  # or any other module's register(registry)

orchestrator = DiscoveryOrchestrator(registry)
result = orchestrator.run(
    ExperimentConfig(
        domain="feynman_physics",
        algorithm="symbolic_regression",
        domain_kwargs={"equation_id": "coulomb_force", "n_samples": 200, "seed": 0},
        algorithm_kwargs={"backend": "gplearn", "generations": 10},
    )
)
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

## One config shape (`engine/config.py`)

- **`RunConfig`** — one orchestrator run (one domain + one algorithm). See
  `configs/run/coulomb_symbolic.yaml`. This is what `sde run <path>` loads.

There used to be a second, `BaselineExperimentConfig`, mirroring a
`configs/paper/*.yaml` shape for multi-model multi-seed comparisons with
ablations and paired-significance testing. It, its contract module
(`engine/experiments/`), its configs and its runner were removed together: they
served a paper framing this project retracted, and one of the configs keyed a
figure on the belief-entropy of the agent loop the audit refuted. By the end
nothing outside its own tests referenced the package.

What replaced it is not a config schema at all. The experiment protocol lives in
`research/AUDIT_METHODOLOGY.md` and the pre-registered margins at the
top of `scripts/run_null_scaffold_audit.py`. That is a better home: a margin
agreed in advance is a claim about method, and keeping it in a YAML file anyone
can edit between runs is the loophole the RFC exists to close.

## The CLI (`cli/main.py`)

```bash
sde list-plugins                                  # what's registered
sde run configs/run/coulomb_symbolic.yaml          # one orchestrator run
```

`_default_registry()` in `cli/main.py` builds an empty `PluginRegistry` and
calls `engine.discovery.discover_plugins(registry)` — it does not import
a plugin (or anything else) by name. Discovery works via Python
packaging entry points: any installed package that declares one under the
`"sde.plugins"` group gets its `register(registry)` called automatically.
This repo's own two plugins are registered the same way, via entry points in
*this repo's* `pyproject.toml`:

```toml
[project.entry-points."sde.plugins"]
feynman_physics = "plugins.physics.plugin:register"
synthetic_regression = "plugins.synthetic.plugin:register"
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
