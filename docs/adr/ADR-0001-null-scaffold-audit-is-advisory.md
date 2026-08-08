# ADR-0001: The null-scaffold audit is advisory and default-on

| | |
|---|---|
| Number | `ADR-0001` |
| Title | The null-scaffold audit is advisory and default-on |
| Status | Proposed |
| Deciders | Shlok Vaishnav |
| Date | 2026-08-09 |
| Originating RFC | `RFC-0001` |
| Supersedes | — |
| Superseded by | — |

---

## Context

RFC-0001 specifies a procedure for determining whether a discovery pipeline's
scaffolding contributes anything its base searcher would not have produced at
matched compute. The RFC deliberately left one question open (§10): is the
audit advisory, or is it a gate that blocks a method from being reported?

The two positions each have a real constituency.

**Mandatory** is the version with teeth. It is also the version that makes the
project's external argument: the gap in the existing literature is not that
nobody has audited scaffolding — several groups have — but that every such
audit was retrospective, manual, and voluntary, and therefore performed by
skeptics rather than by proposers. A platform that merely *offers* an audit
reproduces the voluntary structure it was built to fix.

**Advisory** is the version that does not deter adoption. BOOTSTRAP.md §19
criterion 1 is a researcher outside this project adding a domain without
modifying `engine/`. That researcher currently has zero investment in this
platform's methodology and a working alternative in every existing library. A
platform whose first interaction is a gate that can declare their contribution
worthless is a platform they will not adopt, and an unadopted platform enforces
nothing.

Two facts constrain the choice at the time of this decision:

- **The audit is unproven.** It has not been implemented, and its statistical
  design has not been validated against a case where the correct answer is
  known. Making an unvalidated procedure blocking would let a defect in the
  audit suppress correct work — a failure mode strictly worse than the one it
  guards against.
- **There are no external adopters yet.** Nothing is being deterred today, but
  nothing is being protected either. Both arguments are currently about a
  hypothetical population.

---

## Decision

We will implement the null-scaffold audit as **advisory and default-on**.

The audit runs automatically whenever a benchmark result is generated, and its
verdict is published alongside that result rather than filed separately. It
does not block a merge, a report, or a submission. A pipeline that cannot be
audited reports `NOT_SEPARABLE`, and that value appears in the results table
like any other verdict.

The verdict enumeration is closed: `CONTRIBUTES`, `NULL`, `HARMFUL`,
`INCONCLUSIVE`, `NOT_SEPARABLE`.

The teeth come from publication, not from prohibition. A verdict that always
appears next to the number it qualifies is difficult to omit and does not
require anyone to volunteer for scrutiny — which was the actual defect in the
prior art, and is addressed here without also making an unvalidated procedure
authoritative.

---

## Alternatives rejected

### Mandatory blocking gate — rejected because the audit is not yet validated

A blocking gate is only as trustworthy as the procedure behind it, and this
procedure has no implementation and no validation against known-answer cases. A
false `NULL` from a blocking audit suppresses correct work and is
indistinguishable, to the person affected, from the platform being wrong on
purpose. This is reconsidered once the revisit criteria below are met; the
decision is against *sequencing*, not against the idea.

### Opt-in only — rejected because it reproduces the defect being fixed

An audit a researcher must choose to run is an audit run by people who expect
to pass it. This is precisely the structure RFC-0001 §2 identifies as the gap
in the existing literature, and adopting it would leave the platform with the
mechanism but not the property that makes the mechanism worth having.

### Default-on but reported separately — rejected because separation is omission

A verdict in a separate artifact is a verdict that does not travel with the
number it qualifies. In practice the number is quoted and the audit is not.
Co-location is doing the actual work here, and it costs nothing extra.

---

## Consequences

**What becomes easier:** External authors can adopt the platform without
risking their contribution being blocked by a procedure they did not design.
The audit can be implemented, run, and corrected in public while its own
defects are still being found.

**What becomes harder:** The platform's external claim is weaker than it would
be under a mandatory gate. "We publish an audit alongside every result" is a
smaller claim than "no unaudited result exists here," and any paper argument
must state the smaller one.

**What is now committed to:** Every generated benchmark result carries a
verdict field. Removing it later would be a breaking change to the results
schema and to every downstream report.

**What this constrains:** The audit must be cheap enough to run by default. A
procedure costing 30 seeds × 2 arms on every benchmark run is at the edge of
acceptable, and this decision effectively forbids a design substantially more
expensive than RFC-0001 §7 describes.

---

## Constitutional impact

| Article / Section | Effect |
|---|---|
| Article 14 — Honest reporting | Strengthened operationally. A `NULL` or `HARMFUL` verdict is a negative result published at the same prominence as the number it qualifies, which is what the article requires and previously depended on remembering. |
| Article 13 — Evidence | Extended. The verdict is itself a generated artifact carrying config, seeds, budget, git SHA, and environment. |
| BOOTSTRAP.md §14 — Definition of done | **Unchanged.** No item is added. Were the audit made mandatory, §14 would gain one, and that amendment would belong in the superseding ADR. |

---

## Compliance

The results schema requires a verdict field; a report generated without one
fails schema validation. This is a mechanical check, not a review convention.

Whether the audit *ran* is therefore visible in every artifact. What this ADR
does not enforce — deliberately — is what anyone does about an unflattering
verdict.

---

## Revisit criteria

This decision is reconsidered, via a superseding ADR, when any of the following
is observed:

1. **The audit has been validated.** It returns the correct verdict on at least
   three constructed cases with known ground truth: a scaffold that provably
   contributes, one that is provably null, and one that is provably harmful.
2. **A published result in this repository carries a `NULL` verdict and is
   quoted anyway** without the verdict. That is direct evidence that
   publication alone is insufficient teeth.
3. **External adoption reaches a point where the deterrence argument is
   testable** — at least three plugins authored outside this project. The
   deterrence concern is currently hypothetical and should not outlive the
   evidence indefinitely.
4. **`NOT_SEPARABLE` becomes the majority verdict.** That would indicate the
   `unwrap()` requirement is the wrong interface, and the problem is in
   RFC-0001 §4.4 rather than in this decision.
