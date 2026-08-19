<!--
Copy this file into the branch that answers it (e.g. as the first commit on
`research/<topic>`, named `SPEC.md` at the root of whatever directory the
work lives in) and fill it in BEFORE writing the implementation or looking
at any results. See ../GIT_WORKFLOW.md for when a branch needs this versus
a one-line experiment note.

If the work becomes exploratory partway through — an unexpected observation
redirects it — do not rewrite this file to look like it predicted the
detour. Add a dated addendum at the bottom instead.
-->

# Spec: <branch name>

**Branch:** `research/<topic>` or `experiment/<name>`
**Date opened:** YYYY-MM-DD
**Status:** DRAFT / IN PROGRESS / COMPLETE

## Research question

What are we trying to determine? One or two sentences, specific enough to be answerable.

## Hypothesis

What do we currently expect, and why?

## Null / alternative hypothesis

What result would contradict the hypothesis above? State it concretely enough that a result could actually fail to support it — "nothing happens" is rarely a real null hypothesis on its own; say what "nothing happens" would look like in the actual metric (e.g. a `NULL` verdict within the pre-registered margin, not just "no visible difference").

## Motivation

Why does this question matter for the project's research thesis (`../README.md`)? What would change if it were answered either way?

## Experimental design

How will this be tested? Domain/pipeline, scaffold vs. base searcher, budget, seed count, what varies and what's held constant.

## Metrics

What is actually measured, and which of those measurements decide the outcome. Note whether each is continuous (BCa bootstrap) or paired-binary (Tango score interval) — see `AUDIT_METHODOLOGY.md` §4.2 for why the two need different machinery.

## Baselines / controls

What is this compared against? For this project, the default control is budget-matched independent restarts of the pipeline's own base searcher, not "no scaffold" — note explicitly if a branch deviates from that and why.

## Expected outcomes

Enumerate the plausible outcomes, not just the hoped-for one — e.g. "(a) `CONTRIBUTES`, (b) `NULL`, (c) `HARMFUL`, (d) `INCONCLUSIVE` because the design lacked power, (e) `INCONCLUSIVE` because the selection ceiling was already below the margin (run the feasibility pre-check first — see `scripts/audit_feasibility.py`)."

## Interpretation plan

For each outcome in the previous section: what would it mean, and what would it *not* mean? What follow-up would each outcome imply?

## Confounds considered

What could produce a false positive or false negative here, and how (if at all) is it controlled for?

---

## Results

*(Filled in after the experiment runs — this section, and everything below it, does not exist when the spec is first committed.)*

## Interpretation

What did the result actually establish? What does it explicitly not establish?

## Decision

MERGE / ARCHIVE / REVISE / ABANDON / REPRODUCE — and why. See `../GIT_WORKFLOW.md`'s merge criteria before deciding.
