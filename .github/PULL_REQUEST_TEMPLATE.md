# <title>

## What this changes

<!-- One paragraph. What is different after this merges. -->

## Why

<!-- The problem being solved. Link the governing documents. -->

- Engineering task: `ENG-NNNN`
- Governing ADR: `ADR-NNNN`
- Originating RFC: `RFC-NNNN`

> Production implementation follows Vision → Specification → RFC → ADR →
> Engineering Task → Implementation (BOOTSTRAP.md §15). If those links are
> empty, say which exemption applies: documentation-only, tooling, a fix to a
> defect in an existing implementation, or Milestone 0 foundation work.

---

## Definition of done

<!-- Copied verbatim from BOOTSTRAP.md §14. Every box, or this is not done.
     Partial completion is not a state this project recognizes (Article 7).
     Do not tick a box you have not verified — an unticked box is a normal
     state for a draft; a wrongly ticked one is a defect in the record. -->

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

## Evidence

<!-- Constitution Article 13: no claim is made that the repository cannot
     regenerate. If this PR states a number anywhere — in the description, in
     a docstring, in a README — give the command that regenerates it.

     "No claims made" is a valid and common entry. Write it explicitly. -->

| Claim | Regenerating command | Artifact |
|---|---|---|
| | | |

**Negative results:** <!-- Article 14. If something did not work, or did not
beat its baseline, it goes here at the same prominence as what did. A PR that
reports only improvements is asserting that nothing regressed — make sure that
is true before leaving this blank. -->

---

## Constitutional review

- [ ] `engine/` gained no scientific content, domain name, or plugin import (Article 5)
- [ ] No new coupling to a specific algorithm, library, or backend (Article 6)
- [ ] Dependencies flow one way through documented interfaces; no cycles (Article 9)
- [ ] No special case added to the core that should have been an interface fix (Article 15)
- [ ] Any new dependency is justified by an ADR (BOOTSTRAP.md §10)
- [ ] Any `# domain-independence: allow` pragma added carries a stated reason and is tracked

## Limitations

<!-- Article 8: what this does not do, and where it will break. A section
     claiming no limitations is treated as incomplete, not as exemplary. -->

## Reviewer notes

<!-- Where to start, what you are least sure about, what deserves scrutiny.
     Naming your own weakest point is the fastest route to a useful review. -->
