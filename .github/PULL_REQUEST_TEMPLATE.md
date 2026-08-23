<!--
This is the mini peer review from research/GIT_WORKFLOW.md, made concrete.
Answer every question honestly -- an unanswered or hand-waved one is a real
finding for the reviewer, not a formality to clear.

Implementer: fill in everything through "Decision (implementer's self-assessment)."
Reviewer: fill in "Decision (reviewer)" and everything below it; do not edit the
implementer's answers above, comment instead if you disagree with one.
-->

**Closes:** #<issue number>
**Branch type:** `research/` · `experiment/` · `analysis/` · `method/` · `reproduction/` (pick one)

## What question did this branch answer?

## What was the hypothesis?

(Should match the linked issue. Note here if it changed and why.)

## What evidence was collected?

(Point at the actual artifacts -- results files, config, seeds, environment. See research/AUDIT_METHODOLOGY.md's `AuditReport` provenance fields for what "reproducible" means in this project.)

## What does the result actually establish?

## What does it explicitly NOT establish?

## What confounds remain?

## Could another researcher reproduce this from what's in the PR?

## Does this change the project's research thesis ("Does the Scaffold Earn Its Keep")?

## Decision (implementer's self-assessment)

- [ ] MERGE
- [ ] ARCHIVE
- [ ] REVISE
- [ ] ABANDON
- [ ] REPRODUCE

Why:

---

## Decision (reviewer)

Checked against research/GIT_WORKFLOW.md's nine merge criteria and "when not to merge" list:

- [ ] Scientific relevance
- [ ] Correctness
- [ ] Experimental validity (controls/baselines/metrics appropriate -- budget-matched, not just any control)
- [ ] Reproducibility
- [ ] Documentation
- [ ] Interpretation (what the result does/doesn't establish is understood)
- [ ] Research integrity (negative results and limitations honestly recorded)
- [ ] Integration (doesn't make the codebase harder to understand without justification)
- [ ] Evidence (enough to justify moving from branch to validated state)

Any unchecked box needs a comment explaining the weakness, not silence.

**Reviewer's decision:** MERGE / ARCHIVE / REVISE / ABANDON / REPRODUCE

**Reviewer does not merge this PR.** If the decision is MERGE, label it
`stage:approved-pending-merge` and leave it for a human to click merge.
