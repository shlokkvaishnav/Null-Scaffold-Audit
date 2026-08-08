<!-- Purpose statement required by BOOTSTRAP.md section 8. -->

# docs/eng/

**Purpose:** Engineering tasks — the executable unit of work, derived from an accepted decision.

**Contains:** `ENG-NNNN-short-title.md`, numbered sequentially and never reused. Use `TEMPLATE.md`.

**Does not contain:** Work whose architecture is unsettled. An ENG task never precedes its ADR; if there is no accepted decision, the work is still at the RFC stage.

**Governing rule:** BOOTSTRAP.md section 16 — every specification handed to an implementer must be unambiguous enough to execute without invention. Where an implementer has to guess, the task is defective and is returned for specification rather than resolved by guessing well. This holds whether the implementer is a person or an automated assistant: Article 16 permits tooling to implement against a specification, never to supply the parts of it that were left out.
