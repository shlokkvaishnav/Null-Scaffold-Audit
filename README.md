<div align="center">

# Null-Scaffold-Audit

[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![Build](https://img.shields.io/github/actions/workflow/status/shlokkvaishnav/Null-Scaffold-Audit/ci.yml?style=flat-square&label=build)](https://github.com/shlokkvaishnav/Null-Scaffold-Audit/actions)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue?style=flat-square)](LICENSE)
[![Docker](https://img.shields.io/badge/docker-ready-2496ED?style=flat-square&logo=docker&logoColor=white)](docker-compose.yml)

**Does a search pipeline's scaffold actually contribute anything over its own bare base searcher at matched compute — and can that be answered mechanically, before a paper is published, instead of by a skeptic years later?**

</div>

---

## Research question

Does a pre-registered, budget-matched equivalence-testing audit — comparing a wrapped search pipeline against independent restarts of its own unwrapped searcher, at equal compute — correctly and cheaply distinguish scaffolds that contribute from scaffolds that don't? Applied here to Neural Architecture Search: do NASLib controllers (`DARTSOptimizer`, `GDASOptimizer`, `RegularizedEvolution`, `BANANAS`) actually beat budget-matched restarts of random search, on real NAS-Bench-style benchmark distributions ([NATS-Bench](https://arxiv.org/abs/2009.00437))?

## Why this matters

A scaffolded method is usually published as a whole — a search procedure wrapped in an iterative-refine loop, a belief state, a hypothesis archive — and the reported result is for the whole. Almost nothing in current practice establishes that the wrapper contributed to the number. This isn't hypothetical: this project's own first research direction (an agentic equation-discovery system, now archived) had a scaffold that, for a real stretch of its development, produced character-identical output across all three of its "refine" iterations — a fixed random seed meant the loop never actually looped — and nothing in the pipeline noticed. The literature has begun to notice the general pattern (a 2026 study across ~25,000 agent runs attributed 41.4% of explained variance to the base model and 1.5% to the scaffold), but existing scaffold-ablation studies compare against *no scaffold*, not against *the same searcher given the same compute to use as independent restarts* — which conflates search-budget effects with the scaffold's own contribution. Full positioning against prior work below.

## What we investigate

`engine/audit/` runs two arms at matched compute — the pipeline as submitted, and its own base searcher given the same budget as independent restarts — and decides between them using two one-sided equivalence tests (TOST), not significance testing, because `p > 0.05` cannot license "the scaffold does nothing," only "we couldn't tell." A selection-ceiling measurement (`scripts/measure_selection_ceiling.py`) isolates how much of any gain could come from *selecting* the best of already-generated candidates rather than *generating* better ones, and a feasibility/minimum-detectable-effect pre-check (`scripts/audit_feasibility.py`) decides, before a sweep runs, whether the design could resolve anything at all. Full statistical design: [`docs/rfc/RFC-0001-null-scaffold-audit.md`](docs/rfc/RFC-0001-null-scaffold-audit.md).

## Current findings

**ESTABLISHED** — supported directly by evidence already in this repo:
> On the archived physics-equation-discovery pipeline (`archive/physics_equation_discovery/`), the audit found the scaffold's three "refine" iterations were, for a documented stretch of development, provably identical — a degeneracy the audit's pre-check exists to catch. Across the full `null_scaffold_audit` sweep on that pipeline, no problem/metric combination returned a clean `CONTRIBUTES` verdict; most returned `INCONCLUSIVE` or `HARMFUL`. The equivalence-testing machinery itself (BCa bootstrap, a Tango score interval for paired-binary metrics, Holm correction, achieved-power reporting on every verdict) is implemented and passing its own test suite (165 tests, 98.23% coverage against a 95% floor on `engine/audit/` and `tools/`).

**HYPOTHESIS** — under active investigation, not yet tested:
> That NASLib controllers beat budget-matched random-search restarts on NATS-Bench-style distributions. This question goes back to Li & Talwalkar's *"Random Search and Reproducibility for Neural Architecture Search"* (2019) and is still actively contested as of 2023–2025 follow-up work. Nothing in this repository has tested it yet — `plugins/nas_search/` does not exist.

**OPEN** — unresolved, blocking further progress:
> Whether `naslib` can be installed at all in this environment. It has no PyPI release (last upstream push 2024-11-11) and its git-source install currently fails outright — `uv lock`/`pip`'s submodule init errors on a broken reference (`sphinx_source_code` has no URL in `.gitmodules`), with no known upstream fix found. This is the actual next blocker, not a documentation gap. Also open: whether the audit's specific combination (equivalence testing + budget-matched restart control + oracle-ceiling gate + feasibility pre-check, as one reusable platform capability) is genuinely undemonstrated elsewhere, or whether a closer match exists that three literature-search passes didn't surface.

**DO NOT CLAIM** — statements this evidence does not support:
> "First to apply equivalence testing to scaffold ablation" — false; *"Safety Under Scaffolding"* (arXiv:2603.10044) got there first, in the same year, uncited until this line. "Our audit generalizes across domains" — tested on exactly two in-house domains (physics, one synthetic problem) before this pivot, zero external pipelines. "Our [archived] agent discovers physics equations" — the archived results don't support it, and "agent" overstates a pipeline whose hypothesis generator is verified non-LLM (`archive/physics_equation_discovery/plugins/physics/scaffold/generator.py` is pure gplearn/PySR genetic programming; the belief state is passed in and explicitly ignored). "NAS controllers don't beat random search" or the reverse — nothing has been run yet.

## Methodology, in brief

- **Equivalence testing, not significance testing.** `p > 0.05` on a difference test licenses no conclusion; TOST establishes equivalence positively, against a margin agreed *before* the data is seen.
- **The control is budget-matched independent restarts of the pipeline's own base searcher** — not "no scaffold," and not a single unmatched baseline run. A scaffold that spends 3× the compute and is compared against one base-searcher run is being credited for the compute it consumed.
- **A degeneracy pre-check** runs before any statistics: are a scaffold's own proposals distinct from each other? A scaffold whose iterations repeat is null by construction, and this is the check that would have caught (and later did catch) the archived pipeline's defect at the cost of one run.
- **The margin is pre-registered, per metric**, in code (`scripts/run_null_scaffold_audit.py`), not in an editable config file anyone could tune after seeing results.
- **Paired-binary metrics use a different interval** (Tango score, not the bootstrap) — a metric like exact-recovery rate can produce a collapsed, all-agreeing sample that would otherwise report false certainty; this project's own statistics module documents the bug this fixes.

Full statistical design and every alternative considered: [`docs/rfc/RFC-0001-null-scaffold-audit.md`](docs/rfc/RFC-0001-null-scaffold-audit.md).

## Repository structure

```
engine/                          the platform: plugin contract, registry, orchestrator,
                                  config schemas -- and the audit mechanism, engine/audit/
                                  (arms, statistics, calibration, degeneracy, verdict)
cli/                              the `sde` command-line entry point
plugins/nas_search/              NOT YET IMPLEMENTED -- the current research direction,
                                  blocked on the naslib install issue above
scripts/                          audit runner, selection-ceiling runner, feasibility
                                  pre-check, autonomous research queue, summariser
tests/                            pytest suite for the active tree (165 tests)
docs/                              architecture, plugin guide, development guide, the
                                  RFC/ADR for the audit's statistical design
results/                          fresh output for this study -- currently empty
archive/physics_equation_discovery/  the original research direction (equation discovery
                                  via symbolic regression, AI-Feynman benchmark), preserved
                                  for provenance -- see its own README for what and why
```

## Reproducing the audit apparatus

```bash
docker compose up --build          # no local Python install, no host pip install
docker compose run --rm --entrypoint pytest api tests/ -v   # the image's ENTRYPOINT is
                                                              # `sde`, so pytest needs an
                                                              # explicit override to run
docker compose run --rm api sde list-plugins   # currently empty -- see below
```

Or locally: `uv sync --frozen --extra dev`, then `uv run pytest tests/ -v`. See [`docs/DEVELOPMENT.md`](docs/DEVELOPMENT.md) for the full test/CI breakdown.

**The NAS study itself cannot yet be reproduced end to end** — `plugins/nas_search/` is unimplemented, blocked on the `naslib` install failure documented above. What *can* be run today is the audit apparatus against the archived physics/synthetic domains, or against any new domain that implements the `BaseSearcher`/`Scaffold` protocols in `engine/audit/arms.py`:

```bash
python scripts/audit_feasibility.py --domain <your-domain> ...   # feasibility gate, run first
python scripts/measure_selection_ceiling.py --domain <your-domain> ...
python scripts/run_null_scaffold_audit.py --domain <your-domain> ...
```

## Current results

The **established** finding above comes from the archived physics pipeline's audit sweep: [`archive/physics_equation_discovery/results/null_scaffold_audit/summary.csv`](archive/physics_equation_discovery/results/null_scaffold_audit/summary.csv). **No NAS-domain results exist in this repository yet** — `results/` is currently empty, honestly, rather than seeded with placeholder or projected numbers. Nothing here is backfilled or estimated.

## Limitations

Single audit design, validated pre-pivot on exactly two in-house domains (physics, one synthetic problem) — not yet shown to generalize to a pipeline this team didn't build, which is the single biggest open question (see Related work). The pre-registered margin is a judgement call that requires domain knowledge the engine itself does not have (RFC-0001 §7); a poorly chosen margin produces confident nonsense in either direction. `unwrap()` — exposing a scaffold's bare base searcher — is a genuine structural constraint some pipelines cannot satisfy; those report `NOT_SEPARABLE` rather than being silently exempted, but a determined author could still hide scaffold work inside a nominal "base" searcher. NASLib itself is ~2 years stale (last push 2024-11-11) and currently fails to install via its documented method, independent of anything in this repository.

## Open research questions / next experiments

1. **Resolve the `naslib` install blocker** (highest priority — nothing downstream can run until this is fixed): fork-and-patch the broken submodule reference, a manual non-recursive clone, or an alternative NAS-controller library.
2. **Implement `plugins/nas_search/`**, starting with the cheapest possible sanity check — audit `RandomSearch` against itself, which should return `NULL` — before adding a real controller.
3. **Run the actual audit**: DARTS / GDAS / RegularizedEvolution / BANANAS against budget-matched random-search restarts on NATS-Bench.
4. **External pipeline validation** (highest-value, not yet started): run this audit against a scaffolded method this team did not build, and compare its verdict directly to what the closest existing comparators (ScaffoldSafety, the GAIA controlled comparison) would have concluded on the same case. This is the experiment that would turn a measurement of one project's own pipeline into evidence about the audit method itself.
5. **Margin-sensitivity analysis** — how much does the verdict move as the pre-registered margin moves, holding data fixed?

## Related work

The closest prior art is *"Safety Under Scaffolding: How Evaluation Conditions Shape Measured Safety"* (arXiv:2603.10044) — TOST equivalence testing, pre-registered margins, Holm correction, shipped as a reusable framework, for LLM-agent safety. Its control is "no scaffold at all," not a budget-matched restart of the pipeline's own searcher, which is the specific gap this project's design targets. *"Scaffold Effects on GAIA: A Controlled Comparison"* (arXiv:2606.08529) is pre-registered and controlled but not equivalence-tested and not budget-matched. *"Oracle Gap and Signal Fidelity"* (arXiv:2607.17531) is the closest match to this project's selection-ceiling idea, in LLM best-of-N sampling rather than evolutionary/architecture search. The genre this belongs to — reported gains vanishing under a fairer comparison — includes Dacrema et al. (RecSys'19, recommender systems), Musgrave et al. (ECCV 2020, metric learning), and Melis et al. (ICLR 2018, neural language model evaluation); none of them shipped a reusable mechanism, which is the gap this project targets. On the NAS side specifically: Li & Talwalkar (2019) is the founding result; the GECCO 2025 *Call for Action* (arXiv:2505.03977) and SRBench++ describe adjacent open problems in search-method benchmarking more broadly.

## License

MIT — see [`LICENSE`](LICENSE) for details.
