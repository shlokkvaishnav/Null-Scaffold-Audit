# ENG-NNNN: <short title>

| | |
|---|---|
| ID | `ENG-NNNN` |
| Title | <short title> |
| Originating RFC | `RFC-NNNN` |
| Governing ADR | `ADR-NNNN` |
| Owner | <name> |
| Status | Ready · In Progress · In Review · Blocked · Done · Abandoned |
| Blocked by | — |

> Copy this file to `docs/eng/ENG-NNNN-short-title.md`.
>
> An engineering task is the unit handed to an implementer. **It must be
> unambiguous enough to execute without invention** (BOOTSTRAP.md §16). If the
> implementer has to guess, the task is defective and comes back for
> specification — it is not resolved by guessing well.
>
> That standard applies whether the implementer is a person or an automated
> assistant. Article 16 permits tooling to implement against a specification;
> it does not permit tooling to supply the parts of the specification that were
> left out.
>
> An ENG task never precedes its ADR. If there is no accepted decision, the
> work is still at the RFC stage.
>
> Delete every quoted instruction block before marking the task Ready.

---

## 1. Objective

> One sentence. What will be true when this task is done that is not true now.

---

## 2. Scope

**In scope:**

**Explicitly out of scope:**

> The second list is the one that prevents disputes at review. Name the
> adjacent, tempting work that this task deliberately does not do, and where
> it is tracked instead.

---

## 3. Files

> Every file to be created or modified, with a one-line statement of what
> changes in each. If this list cannot be written in advance, the design is
> not settled and the task is not Ready.

| Path | Change |
|---|---|
| | |

---

## 4. Interfaces

> Exact signatures, types, and error behavior for everything this task adds or
> changes. Written out, not described.
>
> Include what each function does on invalid input, on empty input, and at its
> boundaries. Unspecified error behavior is where implementations diverge from
> intent, silently and permanently.

```python
```

---

## 5. Behavioral specification

> What the code must do, stated so that a disagreement about whether it does
> it can be settled by running something.
>
> Enumerate the cases: normal, boundary, degenerate, and failing. For each,
> the expected observable outcome.

| Case | Input | Expected outcome |
|---|---|---|
| | | |

---

## 6. Tests required

> The specific tests that must exist, named. Not "add tests" — the actual
> assertions, so that review can check the list rather than form an opinion.

- [ ] `test_<name>` — asserts …
- [ ] `test_<name>` — asserts …

**Coverage floor for the touched modules:** <n>%

---

## 7. Evidence required

> Constitution Article 13. Every claim this task will let the project make,
> and the command that regenerates the evidence for it.
>
> If the task makes no external claim, write "none" — that is a real and
> common answer, and stating it is how the reviewer knows it was considered
> rather than skipped.

| Claim | Regenerating command | Artifact path |
|---|---|---|
| | | |

---

## 8. Constraints

> Rules the implementation must observe, including the ones that will feel
> arbitrary without their reason. State the reason.
>
> Standing constraints, always in force:
>
> - `engine/` acquires no scientific content, no domain name, and no plugin
>   import (Article 5, checked by `tools/check_domain_independence.py`).
> - Dependencies flow one way, through documented interfaces (Article 9).
> - No new top-level dependency without an ADR (BOOTSTRAP.md §10).

---

## 9. Definition of done

> Copied verbatim from BOOTSTRAP.md §14. Every box, or the task is not done —
> partial completion is not a state this project recognizes (Article 7).

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

## 10. Notes for the implementer

> Context that helps but is not binding: where similar code already lives,
> which prior task this resembles, which library behaviour to watch for.
>
> Anything in this section that turns out to be *required* belongs in §4, §5,
> or §8 instead. Requirements written as friendly advice get treated as
> friendly advice.
