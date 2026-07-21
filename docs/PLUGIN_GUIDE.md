# Adding a plugin

This is the practical, step-by-step version of [ARCHITECTURE.md](ARCHITECTURE.md).
Follow it when adding a new algorithm, a new scientific domain, or both.

## Before you start: read the reference implementation

`physics_discovery/plugins/feynman.py` is a complete, working example of
every pattern below. When in doubt, match what it does rather than inventing
a new pattern.

## Adding a new `AlgorithmPlugin`

An algorithm plugin needs exactly four things:

```python
class MyAlgorithm:
    name = "my_algorithm"                    # unique, used as the registry key

    def __init__(self, **config):            # constructor takes only kwargs
        ...

    def fit(self, X: np.ndarray, y: np.ndarray) -> "MyAlgorithm":
        ...
        return self                          # must return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        ...

    @property
    def equation(self) -> str | None:
        return None                          # None is correct if not symbolic
```

If you're wrapping an existing fit/predict-style model (most sklearn-shaped
APIs), the adapter is usually 15-20 lines — see `GBMBaselineAlgorithm` and
`NeuralEnsembleAlgorithm` in `physics_discovery/plugins/feynman.py` for the
"no equation" case, and `SymbolicRegressionAlgorithm` for the "has an
equation" case.

## Adding a new `DomainPlugin`

```python
class MyDomain:
    name = "my_domain"

    def load_dataset(self, **kwargs) -> Dataset:
        # kwargs come straight from ExperimentConfig.domain_kwargs / RunConfig.domain_kwargs
        return Dataset(X=..., y=..., feature_names=[...], metadata={...})

    def validate(self, equation: str | None) -> dict:
        # constraint violations for a candidate equation; {} means "no violations" or "not applicable"
        if equation is None:
            return {}
        ...

    def score(self, y_true, y_pred, equation=None) -> dict[str, float]:
        # fit-quality metrics -- this is where domain-specific scoring choices live,
        # NOT in engine/orchestrator.py
        ...
```

`metadata` on `Dataset` is where domain-specific extras go (e.g. the Feynman
plugin puts the ground-truth formula there for benchmarks with a known
answer). The orchestrator never reads `metadata` itself — only your domain
plugin and whatever you build downstream of a `RunResult` should.

## Registering your plugin

Every plugin module exposes one function:

```python
def register(registry: PluginRegistry) -> None:
    registry.register_domain(MyDomain.name, MyDomain)
    registry.register_algorithm(MyAlgorithm.name, MyAlgorithm)
```

Then wire it into the CLI's default registry in `cli/main.py::_default_registry`:

```python
def _default_registry() -> PluginRegistry:
    from physics_discovery.plugins import feynman
    from my_package.plugins import my_plugin_module   # add this

    registry = PluginRegistry()
    feynman.register(registry)
    my_plugin_module.register(registry)                # and this
    return registry
```

## Testing your plugin

Two layers, both demonstrated in the existing test suite:

1. **Fake-plugin orchestrator tests** (`tests/test_engine_orchestrator.py`) —
   prove the *engine* seams work with trivial in-memory fakes, no real
   science, no heavy deps. You generally don't need to add to these unless
   you're changing `engine/` itself.
2. **Real-plugin integration tests** (`tests/test_feynman_plugin.py`) — build
   a `PluginRegistry`, register your real plugin, run it through
   `DiscoveryOrchestrator`, assert on `RunResult`. Copy this file's structure
   for a new plugin. Keep runs fast (small `n_samples`, small
   `population_size`/`generations` for anything search-based) — these should
   run in seconds, not minutes.

If your plugin's dependencies might not be importable everywhere (e.g. they
need a compiled extension that's blocked in some sandboxed environments),
wrap the import in `try/except ImportError: pytest.skip(..., allow_module_level=True)`
at the top of the test module, the way `tests/test_feynman_plugin.py` and
`tests/test_synthetic_plugin.py` do — skip, don't error collection, when the
environment genuinely can't run it. Plain `pytest.importorskip("sklearn.metrics")`
is not enough here: the actual failure can come from a *different* sklearn
submodule (e.g. `sklearn.ensemble`) imported transitively later in the same
module, so guard the real downstream import, not a proxy for it.

## What the second plugin already validated

`physics_discovery/plugins/synthetic.py` (`SyntheticRegressionDomainPlugin`)
is the second plugin the interfaces were checked against, per the principle
below — a domain with no `equation_id`, no known ground-truth formula, and
different `domain_kwargs` entirely, run through the *same* algorithm plugins
`physics_discovery/plugins/feynman.py` registers. It required **no changes**
to `Dataset`, `AlgorithmPlugin`, or `DomainPlugin` — `metadata` being a plain
`dict` and `validate`/`score` not assuming a ground-truth comparison was
already enough. See `tests/test_synthetic_plugin.py`.

That's one data point, not a permanent stamp of approval — a third plugin
(PySR as its own `AlgorithmPlugin`, already a project dependency via the
`pysr` extra, or a domain with categorical/non-numeric features) may still
surface a real gap. Treat this section as "known to work as of the second
plugin," not "proven forever."

## Why a second (and third) plugin matters

The interfaces in `engine/plugin.py` were designed against exactly one real
implementation (physics_discovery) and validated against a second
(`synthetic_regression`, above). A contract shaped by too few examples is a
guess dressed up as a decision. **Before treating these interfaces as fully
stable**, keep stress-testing with plugins that are meaningfully different
from what's already there. If a new plugin needs a change to `Dataset`,
`AlgorithmPlugin`, or `DomainPlugin`, that's the interface doing its job:
better to find it early than after a dozen plugins are built against a
contract that was wrong.
