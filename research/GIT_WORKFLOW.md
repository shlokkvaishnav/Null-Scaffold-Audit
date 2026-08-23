# Git workflow for this research

`main` is the validated research state, not just working code. Branches are where uncertainty gets explored. This document says how those two things stay separate on purpose.

This document is the policy. [`AGENT_PIPELINE.md`](AGENT_PIPELINE.md) is how three separate Claude Code sessions (researcher, implementer, reviewer) carry it out, coordinating through GitHub issues/PRs/labels instead of live messaging.

A branch is not merged because it works or produces a better number. It is merged when it is validated, reproducible, documented, and useful to the research — and a branch that disproves a hypothesis, rules out a mechanism, or produces a negative result can satisfy all four of those without ever changing a line of the implementation. `engine/audit/`'s own verdict enum makes exactly this distinction at the statistical level (`NULL` is a positive finding, not a failure) — this document makes the same distinction at the workflow level.

## Branch types

| Prefix | For | Example |
|---|---|---|
| `research/<topic>` | A substantial research implementation or investigation, usually spanning several experiments | `research/nas-controller-audit` |
| `experiment/<name>` | One specific, narrowly-scoped experiment | `experiment/margin-sensitivity` |
| `analysis/<name>` | Analysis of an existing result — no new data collection | `analysis/ceiling-as-inconclusive-predictor` |
| `method/<name>` | A new methodological component (a metric, a detector, a protocol) | `method/oracle-ceiling-predictor` |
| `reproduction/<target>` | Reproducing an external paper, benchmark, or system's behavior | `reproduction/scaffoldsafety-comparison` |

Don't create a branch for a cosmetic or purely-editorial change — those go straight to `main` via normal CI (lint, typecheck, tests, domain-independence). Do create one whenever a change could affect an experimental conclusion (see **Isolation**, below).

## Before writing code: the spec

A substantial branch (`research/*`, most `experiment/*`) starts with a filled-out copy of [`SPEC_TEMPLATE.md`](SPEC_TEMPLATE.md), committed before the implementation that answers it. The point of committing the spec first is that it timestamps the hypothesis — a hypothesis written after seeing the result is not a hypothesis, it's a caption. If a branch turns exploratory partway through (an unexpected observation redirects it), say so explicitly in the spec and in the final writeup; do not backfill a clean hypothesis once the answer is known.

A small `experiment/*` branch (e.g. re-running an existing sweep at a different seed count) can skip the full template and state the one-line question + expected outcome in the first commit message instead — use judgement, but when in doubt, write the spec.

## Isolation

If a change touches any of: the base searcher, the controller/scaffold set, the benchmark distribution (NATS-Bench vs. anything else), the audit's pre-registered margin, the statistical procedure, or the experimental protocol — put it on its own branch. Mixing two of these in one branch makes "what caused the change" unanswerable, which defeats the point of running the audit at all. This is the same principle `docs/rfc`'s successor, [`AUDIT_METHODOLOGY.md`](AUDIT_METHODOLOGY.md), argues for at the statistical level (why the control is budget-matched restarts, not a single unmatched run) — applied here to how the repository's history should be organized.

## Lifecycle

1. Research question → 2. Literature check → 3. Hypothesis → 4. Experimental design → 5. Implementation → 6. Validation → 7. Experiment → 8. Analysis → 9. Interpretation → 10. **Decision**

The decision is one of:

- **MERGE** — sufficiently validated; becomes part of the main research codebase.
- **ARCHIVE** — scientifically useful, doesn't belong in the main implementation. Keep the branch (or a documented summary + the branch ref) rather than deleting it — the pattern already established by `archive/physics_equation_discovery/`.
- **REVISE** — promising, but the experiment or implementation needs another pass.
- **ABANDON** — the question or approach is no longer useful. Still not deleted without inspection — see below.
- **REPRODUCE** — should be repeated under better controls before a decision can be made.

"Not merged" is not a synonym for "failed." A branch that establishes a scaffold is null, or that a NAS controller doesn't beat random search, is a successful branch even if nothing from it reaches `main`.

## Merge criteria

Before merging into `main`, check all nine — not every one needs to be perfect, but any real weakness has to be written down, not glossed over:

**Scientific relevance** (addresses an approved research question) · **Correctness** (implementation does what it claims) · **Experimental validity** (controls/baselines/metrics are appropriate — for this project, that means budget-matched, not just any control) · **Reproducibility** (another researcher could rerun it — provenance is already captured per `AuditReport`: git SHA, config, environment) · **Documentation** (purpose and methodology are written down) · **Interpretation** (we know what the result does and doesn't establish) · **Research integrity** (negative results and limitations are honestly recorded, matching the README's own "DO NOT CLAIM" standard) · **Integration** (doesn't make the codebase harder to understand without justification) · **Evidence** (enough of it to justify moving from "branch" to "validated state")

A positive result is not a merge criterion. A negative result is not a merge blocker.

## When *not* to merge

Irreproducible. Unstable implementation. Fundamentally flawed methodology. Result depends on an uncontrolled confound. Multiple important variables changed at once with no way to attribute the effect. Result uninterpretable. Substantial technical debt with no research justification. Purely exploratory with no validated conclusion yet. Contradicts the established methodology without an approved reason. Based on cherry-picked runs (re-running a sweep until `CONTRIBUTES` appears, or reporting only the best of several seeds, are both this). Makes an unsupported scientific claim.

Any of these → archive the branch and write down what was learned instead of merging or deleting.

## Negative results

Don't hide them. A hypothesis being false, a controller not beating random search, an expected effect not appearing — these are findings, and get recorded the same way a positive result would. A negative-result branch merges into `main` only if the *implementation itself* becomes validated infrastructure. This already happened once, before this workflow existed: the archived physics scaffold's degeneracy was a negative result about that scaffold specifically, and what merged into `main` as a result wasn't a fix to that scaffold — it was `engine/audit/`, the general mechanism the negative result motivated. Otherwise the branch and its writeup are preserved, not deleted, and the conclusion is what travels forward.

## Tags for milestones

Use tags for validated research milestones, not every commit. First one: `research-v0.1-audit-mechanism-validated`, marking the point where the audit mechanism (TOST, BCa bootstrap, Tango score interval, degeneracy pre-check, selection ceiling, feasibility gate) is implemented and validated — on the archived physics pipeline — but the NAS-controller study itself has not yet started.

## Decision log

Significant decisions — why a baseline was chosen, why a name was picked, why a branch was archived instead of merged, why a result should not be read as general evidence — go in [`DECISION_LOG.md`](DECISION_LOG.md), newest first, so the project doesn't depend on anyone's memory of why something is the way it is.
