# Scientific Discovery Engine — Constitution

**Version 1.0**

---

## Preamble

This constitution states the commitments that hold regardless of subsystem, language, milestone, or contributor.

It is deliberately short. The [Bootstrap Specification](BOOTSTRAP.md) says what we are building; this document says what we will not compromise while building it. Where the two appear to conflict, this document governs.

Articles are amended, never quietly edited. An amendment requires an Architecture Decision Record that names the article, argues the case, and records what was given up.

---

## Article 1 — Correctness

Scientific correctness takes precedence over implementation convenience.

Where an inconvenient approach is more correct, the inconvenience is accepted and documented, not engineered away.

---

## Article 2 — Recorded decisions

Every architectural decision is documented before it is implemented.

A decision that exists only in the code, in a commit message, or in someone's memory has not been made — it has merely happened.

---

## Article 3 — Reproducibility

Every experiment is reproducible.

An experiment that cannot be regenerated from its recorded configuration, environment, and seed is not a result. It is an anecdote, and it is reported as one or not at all.

---

## Article 4 — Single responsibility

Every subsystem has one responsibility, and it can be stated in a sentence.

A subsystem whose purpose requires a paragraph is two subsystems that have not yet been separated.

---

## Article 5 — Domain independence

The core engine is domain independent.

Scientific domains live in plugins. The engine may not import from a plugin, branch on a plugin's identity, or contain the name of a scientific field anywhere in its source.

This is the project's central architectural claim. Violating it does not degrade the architecture — it refutes it.

---

## Article 6 — Algorithm independence

Algorithms live behind stable interfaces.

The engine never depends on a specific algorithm, library, or backend. Any algorithm must be removable without the engine noticing.

---

## Article 7 — Completeness

A feature is complete only when it has tests, documentation, benchmarks where applicable, and review.

Partial completion is not a state this project recognizes. Work that meets some of these criteria is unfinished work.

---

## Article 8 — Documentation

Documentation is a first-class artifact, produced with the feature and reviewed with the same rigor as code.

Every subsystem's documentation states its limitations. A document that claims no limitations is incomplete, not exemplary.

---

## Article 9 — Dependency direction

No module depends on the implementation details of another module.

Dependencies flow through stable, documented interfaces in one direction. A cycle between modules is a defect, not a design.

---

## Article 10 — Performance

Performance optimizations never reduce correctness.

An optimization is accepted only with a benchmark showing the gain and a test showing the behavior is unchanged. Speculative optimization is rejected regardless of how plausible it sounds.

---

## Article 11 — Architecture precedes implementation

Architecture is designed before implementation begins.

Code written ahead of its architecture is a prototype. Prototypes are valuable and they do not merge to the main branch.

---

## Article 12 — Contributions

Every contribution improves at least one of: research, engineering, reproducibility, usability, maintainability, or extensibility.

A change that improves none of these is not neutral. It is cost without benefit, and it is declined.

---

## Article 13 — Evidence

No claim is made that the repository cannot regenerate.

Every benchmark number, comparison table, accuracy figure, and performance statistic is a generated artifact produced by a documented command. Numbers are never typed into prose by hand.

Where documentation and generated evidence disagree, **the evidence is correct and the documentation is a defect** — to be fixed immediately, at whatever cost to the narrative.

---

## Article 14 — Honest reporting

Negative results are reported with the same prominence as positive ones.

A method that does not beat its baseline is documented as not beating its baseline, in the same table, without softening. A method that beats its baseline only under specific conditions has those conditions stated alongside the result.

The project has no interest in appearing to work. It has an interest in being known to work, which is a stronger and more difficult claim.

---

## Article 15 — Scope

The engine acquires capability only through the plugin interface.

When a requirement seems to demand a special case in the core, that is evidence the interface is wrong. The interface is fixed; the special case is not added.

---

## Article 16 — Authorship

The architecture, specifications, and research direction of this project are authored by its maintainers.

Automated tooling implements against specifications and does not originate them. A design that no human reasoned through has not been reviewed, whatever process produced it.
