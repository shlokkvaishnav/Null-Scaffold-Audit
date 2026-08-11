# Contributing

This project is unusual in one respect, and it is worth stating before anything
else: **the specification comes first, and it is binding.**

Three documents govern everything here.

| Document | What it settles |
|---|---|
| [`VISION.md`](VISION.md) | Why this project exists and what would prove it wrong |
| [`BOOTSTRAP.md`](BOOTSTRAP.md) | What is being built, and to what standard |
| [`CONSTITUTION.md`](CONSTITUTION.md) | What will not be compromised while building it |

Where code and these documents disagree, the documents are not stale — the code
is wrong, or the document is amended deliberately through the process below.
A specification the code has quietly outgrown is worse than no specification,
because it lies with authority.

---

## Getting set up

Dependencies are resolved from the committed lock file, so your environment
matches CI and the container exactly.

```bash
uv sync --frozen --extra dev
```

`--frozen` is deliberate. It fails rather than silently re-resolving when
`uv.lock` and `pyproject.toml` disagree, because an environment that quietly
repairs itself is an environment nobody can reconstruct.

Run everything CI runs, before you push:

```bash
uv run ruff check . && uv run ruff format --check .
uv run mypy engine tools
uv run pytest tests/ --cov=tools --cov=engine/audit --cov-fail-under=95
uv run python tools/check_domain_independence.py --path engine
```

If you change dependencies, regenerate and commit the lock file:

```bash
uv lock
```

---

## The workflow

Production implementation follows this path (BOOTSTRAP.md §15). No production
implementation skips it.

```
Vision → Specification → RFC → ADR → Engineering Task → Implementation
       → Tests → Benchmarks → Documentation → Review → Merge
```

### 1. RFC — propose

Copy [`docs/rfc/TEMPLATE.md`](docs/rfc/TEMPLATE.md) to
`docs/rfc/RFC-NNNN-short-title.md`.

An RFC proposes; it does not decide. **It exists to be argued with.** Where an
RFC records no considered alternatives, review returns it — not as a formality,
but because a proposal whose alternatives were never written down was never
compared to anything.

One thing worth internalising: an RFC that nobody challenged is not thereby
validated. It may simply not have been read. If your RFC draws no objections,
seek one out.

### 2. ADR — decide

Copy [`docs/adr/TEMPLATE.md`](docs/adr/TEMPLATE.md) to
`docs/adr/ADR-NNNN-short-title.md`.

The ADR records the decision **and why each alternative lost.** That second part
is what stops the same rejected idea returning every six months from someone who
does not know it was already examined.

**An accepted ADR is immutable.** A decision that changes is superseded by a new
ADR naming the one it replaces; the old record keeps its original text. Editing
an accepted ADR destroys the only evidence of what was believed at the time,
which is the entire point of keeping it (Article 2).

### 3. Engineering task — specify

Copy [`docs/eng/TEMPLATE.md`](docs/eng/TEMPLATE.md) to
`docs/eng/ENG-NNNN-short-title.md`, or open an
[engineering-task issue](.github/ISSUE_TEMPLATE/engineering-task.md).

The task must be **unambiguous enough to execute without invention.** If the
implementer has to guess, the task is defective and goes back for
specification. It is not fixed by guessing well.

An ENG task never precedes its ADR. If there is no accepted decision, the work
is still at the RFC stage.

### 4. Implementation

Against the task, and nothing beyond it. Scope discovered mid-implementation
becomes a new task, not a larger diff.

### 5. Review and merge

Open a PR; the template will load. Every box in the definition of done, or it
is not done — partial completion is not a state this project recognizes.

### When the workflow does not apply

Documentation fixes, tooling, defect fixes in an existing implementation, and
Milestone 0 foundation work go straight to a PR. Say which exemption applies in
the description.

---

## The rules that have teeth

Most projects state principles and hope. These four are enforced mechanically,
because a rule that depends on reviewer attention is a rule that erodes.

### The engine knows no science

`engine/` may not import from a plugin, branch on a plugin's identity, or
contain the name of a scientific field anywhere in its source — including in
comments and docstrings.

This is Article 5, the project's central architectural claim. Violating it does
not degrade the architecture; **it refutes it.** `tools/check_domain_independence.py`
fails the build when it stops being true.

If a finding is genuinely a false positive, suppress the line with a reason:

```python
URL = "https://example.org/spec"  # domain-independence: allow -- external URL, not a concept
```

The reason is mandatory, and every suppression is printed on every run, so they
stay auditable instead of accumulating quietly. A suppression is a debt, and it
should have an issue attached.

### No claim the repository cannot regenerate

Every benchmark number, comparison table, accuracy figure, and performance
statistic is a generated artifact produced by a documented command. **A number
typed into prose by a human is a defect, regardless of whether it happens to be
correct today.**

This is not a rule about honesty in the moral sense. It is a rule about entropy:
a number and its evidence that are not mechanically connected will drift apart,
and nobody will notice until a reviewer runs the code.

### Negative results are reported at equal prominence

A method that does not beat its baseline is documented as not beating its
baseline, in the same table, without softening. A method that wins only under
particular conditions has those conditions stated beside the result.

The project has no interest in appearing to work. It has an interest in being
known to work, which is a harder and more valuable claim.

If your change makes something worse, say so in the PR. That is a normal
outcome and reporting it is the contribution, not a failure of one.

### The interface is fixed, not special-cased

When a requirement seems to demand a special case in the core, that is evidence
the interface is wrong. The interface gets fixed. The special case does not get
added (Article 15).

---

## On AI coding assistants

You may use them to implement. They do not author architecture, RFCs, ADRs, or
vision documents.

This is a governance rule rather than a preference. The value of a specification
lies in its having been reasoned through by someone who will be accountable for
it, and a specification generated by the same system that implements it provides
no independent check — it will agree with the implementation by construction,
including where both are wrong.

The corollary cuts the other way and is the more useful half: **if a task is
specified well enough for an assistant to execute without inventing anything, it
is specified well enough.** A task an assistant cannot follow without guessing
is a task a new contributor cannot follow either. Treat that as a defect in the
specification.

---

## Style

Follow the surrounding code. `ruff` settles formatting disputes so that review
can be about substance.

Comments explain **why**, not what. A comment restating the code adds a second
thing to keep in sync and no information. The comments worth writing are the
ones recording a constraint, a rejected alternative, or a surprise — the things
that are invisible in the code and expensive to rediscover.

Every subsystem's documentation states its limitations (Article 8). A document
claiming none is treated as incomplete, not as exemplary.
