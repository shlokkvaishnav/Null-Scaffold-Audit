# Scientific Discovery Engine (SDE)

SDE is a small, domain-agnostic engine (`engine/`) that orchestrates a fixed
workflow — load data, split, fit, predict, validate, score — while
delegating every step that touches actual scientific content to a plugin.
The engine doesn't know what a "physics equation" or a "neural architecture"
is; it only knows the shapes of a dataset and of an algorithm/domain plugin.
Plugins are found at runtime via Python packaging entry points
(`engine/discovery.py`), not hardcoded into the engine or CLI — anyone can
add a plugin for their own domain by installing a package that declares one,
with no change to this repo. See [docs/PLUGIN_GUIDE.md](docs/PLUGIN_GUIDE.md).

**`engine/audit/`** is the platform's other half, and the current research
direction: a budget-matched, pre-registered audit of whether a search
pipeline's scaffold contributes *anything* that its own bare base searcher
would not have produced at the same compute. It compares a wrapped pipeline
against independent restarts of its own unwrapped searcher, using two
one-sided equivalence tests (TOST) rather than significance testing — because
`p > 0.05` cannot license "the scaffold does nothing," only "we couldn't
tell." An oracle-ceiling measurement and a feasibility/minimum-detectable-effect
pre-check decide, before a sweep runs, whether the design can resolve
anything at all. See
[docs/rfc/RFC-0001-null-scaffold-audit.md](docs/rfc/RFC-0001-null-scaffold-audit.md)
for the full statistical design.

## Current study: Does the Scaffold Earn Its Keep

Applying that audit to Neural Architecture Search: do NAS controllers (via
[NASLib](https://github.com/automl/NASLib) — `DARTSOptimizer`,
`GDASOptimizer`, `RegularizedEvolution`, `BANANAS`, and others) actually beat
budget-matched independent restarts of random search, on real (non-uniform)
NAS-Bench-style benchmark distributions
([NATS-Bench](https://arxiv.org/abs/2009.00437))? This question goes back to
Li & Talwalkar's *"Random Search and Reproducibility for Neural Architecture
Search"* (2019) and is still actively contested. Zero training compute is
required — NATS-Bench is a tabular lookup of precomputed architecture
accuracies, so the whole study runs from downloaded tables, not GPUs.

`plugins/nas_search/` is not yet implemented — see the migration report for
this pivot for current status and next steps.

## The archived direction: physics equation discovery

The project's original plugin, `plugins/physics/` — an agentic
hypothesize→verify→refine loop wrapping symbolic regression, benchmarked
against the 36-equation AI-Feynman set — is preserved under
[`archive/physics_equation_discovery/`](archive/physics_equation_discovery/README.md),
not deleted. It's kept because it's the reason `engine/audit/` exists: its
scaffold was found, by accident, to produce character-identical output across
all three of its "refine" iterations (a fixed random seed meant the loop
never actually looped), and nothing in the pipeline would have noticed
without the audit that was subsequently built to catch exactly that failure
mode. That finding — and the audit results showing the scaffold was
statistically indistinguishable from, or worse than, plain restarts — is the
one piece of verified evidence this project has produced to date. It is not
under active development.

Read [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for how the engine and a
plugin fit together, and [docs/PLUGIN_GUIDE.md](docs/PLUGIN_GUIDE.md) if
you're adding a new algorithm or domain.

## Quick start

```bash
docker compose up --build
```

No local Python install, no host `pip install` — everything runs inside the
container.

```bash
docker compose run --rm api pytest tests/ -v
docker compose run --rm api sde list-plugins
```

See [docs/DEVELOPMENT.md](docs/DEVELOPMENT.md) for local (non-Docker) setup
and the full test/CI breakdown.

## The `sde` CLI

```bash
sde list-plugins    # what's registered -- currently empty until plugins/nas_search/ lands
sde run <config>    # run one domain+algorithm plugin pair
```

## Repository layout

```
engine/                          the platform: plugin contract, registry, orchestrator,
                                  config schemas, and the audit mechanism (engine/audit/)
cli/                              the `sde` command-line entry point
plugins/nas_search/              NOT YET IMPLEMENTED -- the current research direction
scripts/                          audit runner, selection-ceiling runner, feasibility
                                  pre-check, research queue, summariser
tests/                            pytest suite
docs/                              architecture, plugin guide, development guide, RFC/ADR
archive/physics_equation_discovery/  the original, now-archived research direction --
                                  see its own README for what's there and why
```

Full breakdown in [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

## License

MIT License — see [LICENSE](LICENSE) for details.
