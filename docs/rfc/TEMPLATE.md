# RFC-NNNN: <short title>

| | |
|---|---|
| Number | `RFC-NNNN` |
| Title | <short title> |
| Author | <name> |
| Status | Draft · In Review · Accepted · Rejected · Withdrawn · Superseded |
| Created | YYYY-MM-DD |
| Supersedes | — |
| Superseded by | — |
| Resulting ADR | — |

> Copy this file to `docs/rfc/RFC-NNNN-short-title.md`. Numbers are assigned
> sequentially and never reused, including for withdrawn RFCs.
>
> An RFC **proposes**. It does not decide. The decision, once made, is recorded
> as an ADR that links back here.
>
> Delete every quoted instruction block before opening the RFC for review.

---

## 1. Summary

> Two or three sentences. What is being proposed, and what changes if it is
> accepted. A reader should be able to stop here and know whether the rest of
> the document concerns them.

---

## 2. Motivation

> What problem exists today. Be concrete: describe the situation that prompted
> this RFC, not the abstract desirability of the proposal.
>
> State the cost of doing nothing. If that cost is low, say so — an RFC may
> honestly conclude that the status quo is adequate.

---

## 3. Guide-level explanation

> Explain the proposal as you would to a contributor who will use it but not
> implement it. Use the vocabulary the project already uses. Show the interface
> or workflow as it will appear to them, with an example.
>
> If this section is hard to write, that is evidence about the design, not
> about the writing.

---

## 4. Reference-level explanation

> The precise design. Interfaces, types, data flow, error behavior, and the
> boundaries between this and adjacent subsystems.
>
> Enough detail that an implementer does not have to invent anything
> (BOOTSTRAP.md §16). Where a detail is deliberately left open, say so
> explicitly and record it in §10.

---

## 5. Constitutional review

> Every RFC states its relationship to the Constitution. Answer each; "not
> applicable" is a valid answer with a reason attached.

| Article | Question | Assessment |
|---|---|---|
| 5 — Domain independence | Does this put any scientific content, domain name, or plugin import into `engine/`? | |
| 6 — Algorithm independence | Does this couple the engine to a specific algorithm, library, or backend? | |
| 9 — Dependency direction | Which way do dependencies flow? Does this create a cycle? | |
| 13 — Evidence | What claims will this make, and which command regenerates the evidence for each? | |
| 15 — Scope | Does this add a special case to the core rather than fixing an interface? | |

If any answer is uncomfortable, write the uncomfortable version. An RFC that
passes its own constitutional review too easily has usually not been read
against it.

---

## 6. Alternatives considered

> **Mandatory. An RFC recording no considered alternatives is returned at
> review** (BOOTSTRAP.md §15).
>
> At least two, each with a real argument in its favour before the argument
> against. "Do nothing" is a legitimate alternative and is frequently the one
> that deserves the most honest treatment.

### Alternative A — <name>

**What it is:**

**In its favour:**

**Why it loses:**

### Alternative B — <name>

**What it is:**

**In its favour:**

**Why it loses:**

---

## 7. Drawbacks

> What this proposal costs even if it works exactly as intended. Complexity
> added, flexibility given up, contributors burdened, doors closed.
>
> A proposal with no drawbacks has not been examined closely enough. Article 8
> applies to proposals as well as to subsystems.

---

## 8. Prior art

> How other systems solved this, inside and outside this domain of software.
> What they got right, what they got wrong, and which part of their solution is
> being borrowed. Include the cases where the analogy breaks down.

---

## 9. Impact

**Affected subsystems:**

**Interface changes:** *(breaking / additive / none — and for whom)*

**Migration required:** *(what existing code or data must change, and who does it)*

**Documentation to update:**

**Governing documents to amend:** *(if this contradicts BOOTSTRAP.md or
CONSTITUTION.md, that amendment is part of this proposal, not a follow-up)*

---

## 10. Unresolved questions

> What this RFC deliberately leaves open, and how each will be resolved —
> during review, during implementation, or in a later RFC.
>
> An empty section here is a claim that nothing is uncertain. Make that claim
> only when it is true.

---

## 11. Future possibilities

> What this enables that is explicitly out of scope now. Kept separate so that
> scope discussion stays about the proposal rather than about its descendants.
