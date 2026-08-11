# Scientific Discovery Engine (SDE)

**Bootstrap Specification**

| | |
|---|---|
| Version | `v0.0.0-foundation` |
| Status | Draft |
| Owner | Shlok Vaishnav |
| Supersedes | — |
| Superseded by | — |

---

## 0. About this document

This document defines the engineering, architectural, and research foundations of the Scientific Discovery Engine.

It is the highest-level technical authority in the project. No implementation may contradict it. Where reality demands a change, the change is made **here first**, via an Architecture Decision Record that explicitly names the section it supersedes — never by writing code that disagrees with it and leaving the document stale.

A specification that the code has quietly outgrown is worse than no specification, because it lies with authority. Keeping this document true is a standing obligation, not a milestone.

---

## 1. Vision

The Scientific Discovery Engine is an engineering platform for computational scientific discovery.

Rather than solving one scientific problem, SDE provides the reusable infrastructure by which researchers discover, validate, interpret, benchmark, and reproduce scientific knowledge — across any domain, without modifying the core engine.

The long-form argument for why this platform should exist lives in [`VISION.md`](VISION.md). This section states only the commitment: **the engine is domain-independent, permanently.** Scientific content lives exclusively in plugins.

---

## 2. Mission

Build a scientific discovery platform that demonstrates genuine excellence across:

- Software engineering
- Research engineering
- Scientific machine learning
- Experiment management
- Reproducibility
- MLOps and infrastructure
- Developer experience

The measure is not breadth of features. It is whether a researcher outside this project can add a scientific domain, get a trustworthy result, and reproduce it a year later.

---

## 3. Long-term goal

A framework capable of supporting discovery in biology, medicine, physics, chemistry, economics, finance, materials, and climate — with **zero changes to the core engine** for any of them.

If adding a domain requires touching `engine/`, the architecture has failed and the failure is the engine's, not the domain's.

---

## 4. Core principles

These are immutable. Changing one requires an ADR that argues the case explicitly and is reviewed as an amendment, not an edit.

### 4.1 Architecture first

Architecture is designed and documented before the subsystem it governs is implemented. Not the reverse, and not concurrently.

### 4.2 Plugin first

Scientific domains, discovery algorithms, validators, reports, and visualizations are all replaceable through stable interfaces. Nothing scientific is hardcoded into the engine.

### 4.3 Research first

Scientific correctness outranks engineering convenience. When they conflict, correctness wins and the inconvenience gets documented.

### 4.4 Reproducibility first

Every experiment, every figure, and every reported number must be regenerable from a recorded configuration, a recorded environment, and a recorded seed.

### 4.5 Evidence over assertion

**No claim appears in any document unless a command in this repository regenerates the evidence for it.**

This is the principle with teeth. Benchmark tables, performance figures, accuracy claims, and comparison results are *generated artifacts*, never hand-authored prose. A number typed by a human into a README is a defect, regardless of whether it happens to be correct today.

The corollary: **negative results are first-class.** A method that fails to beat its own baseline is reported as failing to beat its own baseline. There is no version of this project in which the documentation is more optimistic than the evidence.

### 4.6 Production quality

Every module meets professional engineering standards for testing, typing, logging, configuration, and documentation. No prototype code on the main branch.

### 4.7 Documentation as implementation

A feature without documentation is not complete. Documentation is not a follow-up task; it is part of the deliverable.

---

## 5. What SDE is

- A scientific workflow engine
- A research platform
- An experiment orchestration framework
- A benchmarking laboratory
- A publication support system
- A plugin architecture with stable contracts

## 6. What SDE is not

- A wrapper around any single symbolic regression library
- An AutoML library
- A notebook collection
- A domain-specific application
- A machine learning tutorial
- A university assignment

Where a proposed feature would move the project toward any item in this list, the feature is rejected or redesigned. This list is a scope defence, and it is meant to be used.

---

## 7. Target users

**Primary**

- Research scientists
- Research engineers
- Machine learning researchers
- Faculty and graduate students

**Secondary**

- Undergraduate researchers
- Scientific software developers
- Industrial R&D teams

The primary users share one property that drives most design decisions: **they will not trust a result they cannot regenerate.** Design for that reader.

---

## 8. Repository philosophy

- Every engineering decision is understandable from the documents in the repository, without reading the code.
- Every module has exactly one responsibility.
- Every interface is documented at its boundary.
- Every subsystem is replaceable.
- Every feature has a measurable purpose, and the measurement exists.

No directory exists without a documented purpose. A directory whose `README.md` cannot explain why it exists gets deleted, not tolerated.

---

## 9. Repository structure

```
scientific-discovery-engine/
├── .github/            CI workflows, issue and PR templates, automation
├── docs/               all human-readable documentation
│   ├── design/         subsystem design documents
│   ├── rfc/            request-for-comments: proposed architecture
│   ├── adr/            architecture decision records: decided architecture
│   ├── eng/            engineering tasks: ENG-xxx, executable without invention
│   └── research/       mathematical background, literature, methodology
├── engine/             the domain-independent core; knows no science
├── algorithms/         discovery algorithm implementations behind stable interfaces
├── plugins/            scientific domains; the only place science lives
├── constraints/        search-space constraints (dimensional, structural, physical)
├── validators/         hypothesis validation independent of how it was generated
├── reports/            publication-quality artifact generation
├── visualization/      figure generation; every figure regenerable
├── sdk/                the surface external plugin authors build against
├── cli/                the command-line entry point
├── configs/            experiment and run configurations, versioned
├── experiments/        experiment definitions and recorded results
├── benchmarks/         benchmark suites and their generated result artifacts
├── papers/             publication drafts and their regenerable artifacts
├── docker/             container definitions
├── devcontainer/       reproducible development environment
├── scripts/            operational entry points
├── tools/              repository tooling and code generation
├── tests/              the test suite
└── examples/           worked examples for plugin and algorithm authors
```

The load-bearing boundary is between `engine/` and `plugins/`. `engine/` may never import from `plugins/`, may never branch on a plugin's identity, and may never contain the name of a scientific domain. This is enforced in CI, not by convention.

---

## 10. Technology philosophy

Languages are chosen by technical suitability, and each addition must justify the cost it imposes on contributors.

| Language | Role | Status |
|---|---|---|
| Python | Orchestration, ML, scientific computing | Current |
| Rust | Performance-critical computation, storage, parsing | Planned |
| Go | Distributed services, workers, scheduling | Future |
| TypeScript | Dashboards, documentation UI | Future |

**A second language enters the repository only when an ADR demonstrates a measured bottleneck that the first language cannot address.** Polyglot repositories are a cost paid by every future contributor; that cost is justified by benchmark evidence, never by preference.

Tooling decisions — dependency management, configuration, containerization — are each recorded as their own ADR, with the alternatives considered and the reason for the choice.

---

## 11. Engineering standards

Every subsystem ships with:

- Unit tests, with a stated and CI-enforced coverage floor
- Documentation at its interface boundary
- Complete type hints, checked in CI
- Structured logging
- Externalized configuration
- Benchmarks, where the subsystem has performance characteristics that matter

Experimental code does not bypass these standards. It lives on a branch until it meets them.

### 11.1 Reproducibility requirements

These are not aspirational; they are acceptance criteria.

- All dependencies pinned via a committed lock file
- Container base images pinned by digest, not tag
- Every experiment records: configuration, git SHA, seed, resolved environment, and hardware
- Rerunning a recorded experiment reproduces its recorded metrics

---

## 12. Documentation standards

Every subsystem's documentation states:

1. **Purpose** — what problem it solves
2. **Architecture** — how it is built
3. **Interfaces** — what it exposes and what it requires
4. **Extension points** — how someone else extends it
5. **Limitations** — what it does not do and where it will break
6. **Future work** — known gaps

Section 5 is mandatory and is the one most often skipped. A document with no stated limitations is treated as incomplete, not as evidence of a flawless subsystem.

---

## 13. Research standards

Every research module documents:

- Literature references
- Mathematical background
- Baseline comparisons, including the trivial baseline
- Evaluation methodology, including data splits and their justification
- Reproduction instructions
- Limitations and threats to validity
- Future research directions

**Every method is compared against the simplest thing that could work.** A method that does not beat its own trivial baseline is reported as such, prominently, in the same table.

---

## 14. Definition of done

A feature is complete only when all of the following hold:

- [ ] Implementation complete
- [ ] Tests pass and coverage floor is met
- [ ] Type checking passes
- [ ] Documentation written, including limitations
- [ ] Benchmarks recorded, where applicable
- [ ] Every claim made about the feature is backed by a regenerable artifact
- [ ] Governing architecture documents remain accurate
- [ ] CI passes
- [ ] Review checklist passes

---

## 15. Development workflow

```
Vision
  ↓
Specification
  ↓
RFC                  proposed architecture, open for challenge
  ↓
ADR                  the decision, and why the alternatives lost
  ↓
Engineering Task     ENG-xxx, unambiguous and executable
  ↓
Implementation
  ↓
Tests
  ↓
Benchmarks
  ↓
Documentation
  ↓
Review
  ↓
Merge
```

No production implementation skips this workflow.

The RFC stage exists to be *argued with*. An RFC that no one challenged is not thereby validated — it may simply not have been read. Where an RFC records no considered alternatives, review returns it.

---

## 16. Repository governance

Every major subsystem requires an RFC, an ADR, engineering tasks, and a design review before implementation begins.

**On the use of AI coding assistants.** AI assistants implement against specifications; they do not author architecture, RFCs, ADRs, or vision documents. The intellectual ownership of this project's design stays with its authors. This is a governance rule, not a preference — the value of the specification lies in it having been reasoned through, and a specification generated by the same system that implements it provides no independent check.

The corollary: **every specification handed to an implementer must be unambiguous enough to execute without invention.** If an implementer has to guess, the specification is defective and gets fixed before implementation proceeds.

---

## 17. Versioning strategy

| Version | Scope |
|---|---|
| `v0.0.x` | Foundation |
| `v0.1.x` | Platform infrastructure |
| `v0.2.x` | Plugin runtime |
| `v0.3.x` | Workflow engine |
| `v0.4.x` | Experiment engine |
| `v0.5.x` | Discovery engine |
| `v0.6.x` | Validation |
| `v0.7.x` | Reference domain plugin |
| `v0.8.x` | Benchmark laboratory |
| `v0.9.x` | Publication pipeline |
| `v1.0.0` | Stable Scientific Discovery Engine |

The reference domain plugin at `v0.7.x` is a *demonstration that the plugin contract holds*, chosen at that time on evidence. It is not the project's subject, and it does not become the project's identity.

---

## 18. Milestone 0 — `v0.0.0`

Milestone 0 is complete when the repository contains:

1. `BOOTSTRAP.md`, `CONSTITUTION.md`, `VISION.md`
2. The full directory structure of §9, each directory carrying a `README.md` stating its purpose
3. RFC, ADR, and engineering-task templates
4. Dependency management with a committed lock file
5. A container definition with a digest-pinned base image
6. CI running lint, type check, tests, and the `engine/` domain-independence check
7. Issue and pull-request templates encoding the §14 definition of done
8. `CONTRIBUTING.md` describing the §15 workflow

Milestone 0 contains **no scientific implementation whatsoever**. Not a search algorithm, not a metric, not a dataset loader. Any scientific code appearing in `v0.0.0` is a scope violation and is removed.

---

## 19. Success criteria

SDE succeeds when:

1. A researcher outside this project adds a scientific domain without modifying `engine/`
2. A new discovery algorithm integrates behind the existing interface with no core changes
3. Any published result can be regenerated by a single documented command
4. Publication-quality reports generate automatically from recorded experiments
5. The architecture absorbs a genuinely unanticipated requirement without redesign
6. A stranger reading the repository can reconstruct why every major decision was made

Criterion 5 is the real test, and it cannot be verified in advance. Criterion 6 is the reason this document exists.

---

## 20. Motto

> Build infrastructure that accelerates scientific discovery, not just another machine learning application.
