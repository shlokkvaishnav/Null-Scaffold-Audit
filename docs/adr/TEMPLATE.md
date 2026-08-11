# ADR-NNNN: <short title>

| | |
|---|---|
| Number | `ADR-NNNN` |
| Title | <short title> |
| Status | Proposed · Accepted · Rejected · Deprecated · Superseded |
| Deciders | <names> |
| Date | YYYY-MM-DD |
| Originating RFC | `RFC-NNNN` or — |
| Supersedes | — |
| Superseded by | — |

> Copy this file to `docs/adr/ADR-NNNN-short-title.md`. Numbers are sequential
> and never reused.
>
> **An accepted ADR is immutable.** A decision that changes is superseded by a
> new ADR naming the one it replaces, and the old record keeps its original
> text. Editing an accepted ADR destroys the only evidence of what was believed
> at the time, which is the entire value of the record (Constitution Article 2).
>
> The one permitted edit to an accepted ADR is filling in its **Superseded by**
> row and setting its status.
>
> Delete every quoted instruction block before submitting.

---

## Context

> The forces at play, written as they stood *at the time of the decision* —
> present tense, no hindsight.
>
> Include the constraints that were real: deadlines, existing commitments,
> team size, what was already built. A future reader needs to know not just
> what was chosen but what the choice was made *under*, otherwise a reasonable
> decision will look inexplicable later.
>
> Where the decision rests on measurement rather than judgement, state the
> measurement and the command that produced it (Constitution Article 13).

---

## Decision

> One paragraph, active voice, unambiguous. "We will …"
>
> This is the section people will quote. Write it so that quoting it out of
> context does not mislead.

---

## Alternatives rejected

> Each alternative that was seriously considered, and the specific reason it
> lost. Not a summary of the RFC — the *decisive* reason.
>
> This section is what stops the same alternative being re-proposed every six
> months by someone who does not know it was already examined.

### <Alternative> — rejected because …

### <Alternative> — rejected because …

---

## Consequences

**What becomes easier:**

**What becomes harder:**

**What is now committed to** *(and what it would cost to reverse)*:

**What this constrains in future decisions:**

> Consequences are stated whether or not they favour the decision. An ADR that
> reads as advocacy has failed at its job, which is to inform a reader who does
> not yet know whether the decision was right.

---

## Constitutional impact

> Which articles this decision touches, and how.
>
> **If this decision amends an article or a section of BOOTSTRAP.md, say so
> explicitly and state the amended text.** The amendment happens here, in the
> record, and the governing document is updated in the same change — never by
> letting the code drift and the document go stale (BOOTSTRAP.md §0).

| Article / Section | Effect |
|---|---|
| | |

---

## Compliance

> How adherence to this decision is checked. Name the mechanism.
>
> A CI check, a test, a lint rule, or a review-checklist item. "Reviewers will
> remember" is not a compliance mechanism — a rule that depends on reviewer
> attention is a rule that erodes.

---

## Revisit criteria

> The observation that would make this decision wrong.
>
> State it concretely enough to be noticed: a benchmark number, a contributor
> complaint that recurs, a scaling limit, a dependency going unmaintained. A
> decision with no stated revisit criteria tends to outlive its justification,
> because nobody knows what would count as evidence against it.
