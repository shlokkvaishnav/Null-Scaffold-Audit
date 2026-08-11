---
name: Engineering task
about: A unit of implementation work derived from an accepted decision
title: "ENG-NNNN: "
labels: ["engineering-task"]
---

<!-- This mirrors docs/eng/TEMPLATE.md so that an accepted issue can be
     promoted to an ENG document without re-typing it.

     An engineering task must be unambiguous enough to execute without
     invention (BOOTSTRAP.md §16). If the implementer has to guess, the task is
     defective and comes back for specification — it is not resolved by
     guessing well. That holds whether the implementer is a person or an
     automated assistant. -->

## Governing documents

- Originating RFC: `RFC-NNNN`
- Governing ADR: `ADR-NNNN`

> An ENG task never precedes its ADR. If there is no accepted decision, this is
> still at the RFC stage — open the RFC instead.

## Objective

<!-- One sentence. What will be true when this is done that is not true now. -->

## Scope

**In scope:**

**Explicitly out of scope:**

<!-- Name the adjacent, tempting work this deliberately does not do, and where
     it is tracked instead. This is the list that prevents disputes at review. -->

## Files

| Path | Change |
|---|---|
| | |

<!-- If this table cannot be filled in advance, the design is not settled and
     the task is not ready to be worked. -->

## Interfaces

<!-- Exact signatures and error behavior, written out rather than described.
     Include behavior on invalid input, on empty input, and at the boundaries:
     unspecified error behavior is where implementations silently diverge from
     intent. -->

```python
```

## Behavioral specification

| Case | Input | Expected outcome |
|---|---|---|
| | | |

## Tests required

- [ ] `test_<name>` — asserts …

**Coverage floor for the touched modules:** <n>%

## Evidence required

<!-- Article 13. Every claim this will let the project make, and the command
     that regenerates the evidence. "None" is a valid entry — write it, so a
     reviewer knows it was considered rather than skipped. -->

| Claim | Regenerating command | Artifact |
|---|---|---|
| | | |

## Definition of done

<!-- Verbatim from BOOTSTRAP.md §14. -->

- [ ] Implementation complete
- [ ] Tests pass and coverage floor is met
- [ ] Type checking passes
- [ ] Documentation written, including limitations
- [ ] Benchmarks recorded, where applicable
- [ ] Every claim made about the feature is backed by a regenerable artifact
- [ ] Governing architecture documents remain accurate
- [ ] CI passes
- [ ] Review checklist passes
