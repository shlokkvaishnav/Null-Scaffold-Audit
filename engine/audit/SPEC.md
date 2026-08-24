<!--
Copied verbatim from GitHub issue #23, per research/GIT_WORKFLOW.md and
research/SPEC_TEMPLATE.md. Section headers below follow SPEC_TEMPLATE.md;
their content is the issue body's corresponding section, unedited.
-->

# Spec: method/superiority-precision-metric

**Branch:** `method/superiority-precision-metric`
**Date opened:** 2026-08-24
**Status:** IN PROGRESS
**Issue:** https://github.com/shlokkvaishnav/Null-Scaffold-Audit/issues/23

## Branch type

method/ -- a new methodological component (a detector/metric, not a domain
or a scaffold)

## Research question

`engine/audit/statistics.py`'s `required_sample_size` (and the
`n_for_target_power`/`power` fields it populates on every `MetricVerdict`)
measures one specific thing, by its own docstring: the sample size (and
achieved power) for the **TOST equivalence claim** -- "the smallest paired
sample size whose TOST power reaches `target_power`," "reported so
`INCONCLUSIVE` stops being a surprise." That is exactly the right question
when the verdict is `NULL` or `INCONCLUSIVE` -- both are claims (or
non-claims) *about equivalence*, and TOST power is the correct lens.

It is a different question when the verdict is `CONTRIBUTES` or `HARMFUL`.
Those verdicts rest on a one-sided superiority/inferiority claim (`ci_low >
margin` or `ci_high < -margin`, per `_resolve`), not an equivalence claim,
and nothing in `engine/audit/` currently reports a power or precision
number that actually describes *that* claim's reliability. `power` and
`n_for_target_power` are still computed and populated on `CONTRIBUTES`/
`HARMFUL` rows -- they just answer a question ("how many seeds would this
row need for adequate power to establish equivalence") that has little to
do with whether the superiority verdict actually reached is trustworthy.

This is not a hypothetical concern -- it already happened, three times, in
this project's own recent history, and was each time worked around by hand
rather than by anything in `engine/audit/` itself:

- Issue #19: Ackley's `INCONCLUSIVE` verdict was correctly diagnosed via
  `n_for_target_power` (135) vs. actual `n` (60) -- appropriate, since
  `INCONCLUSIVE` genuinely is a TOST-power question.
- Issue #21: Rastrigin's `CONTRIBUTES` verdict was re-run at higher `n`
  using the *same* diagnostic (`n_for_target_power=181` vs. `n=60`) --
  but `CONTRIBUTES` is a superiority claim, not an equivalence one, so this
  was arguably applying the wrong statistic. It happened to be harmless
  (the CI was already ~9-10x outside the margin at n=60, and the re-run at
  n=200 confirmed rather than changed the verdict), but the reasoning that
  justified the extra ~140-seed run was not actually diagnosing what it
  was framed as diagnosing.
- `STEPSIZE_SPEC.md` (issue #17 / PR #18) noticed the same gap directly and
  worked around it manually: *"Power for this row is 0.00 for the same
  reason as PR #16's rastrigin row -- this is TOST-for-NULL power, and it
  isn't the number that describes an `INCONCLUSIVE` verdict's own evidence;
  what matters is that the CI now spans zero."* That observation was never
  turned into anything `engine/audit/` itself reports -- it's a comment in
  one plugin's spec file, not a capability available to the next one.

**Question:** should `engine/audit/` report a distinct, superiority-
relevant precision metric on `CONTRIBUTES`/`HARMFUL` rows -- e.g. the
observed CI's distance past the margin boundary as a multiple of the CI's
own half-width, or a proper minimum-detectable-effect-at-achieved-`n`
calculation for a one-sided test -- so that a caller deciding whether a
`CONTRIBUTES`/`HARMFUL` verdict is trustworthy enough at its achieved `n`
(or whether it's worth re-running at higher `n`) is looking at a number
that actually answers that question, instead of reusing (or manually
overriding) a power figure documented as being about a different claim?

## Hypothesis

A superiority-relevant precision metric -- concretely, proposed as
`(|ci_bound_nearer_to_margin| - margin) / (ci_high - ci_low)`, i.e. how many
CI-half-widths past the decision boundary the nearer CI bound sits,
computed the same way for both `CONTRIBUTES` and `HARMFUL` rows using
whichever bound `_resolve` actually used to reach the verdict -- would,
applied retrospectively to this project's own already-collected data:

- Correctly indicate Rastrigin's original 60-seed `CONTRIBUTES` row
  (diff +17.54, CI [+15.84, +19.22], margin ±1.84) as already decisively
  resolved, not needing the re-run issue #21 performed -- a large ratio,
  reflecting that the CI was already comfortably clear of the boundary.
- Correctly indicate the same for Ackley's decisive `HARMFUL` row from PR
  #20 (n=1971) as decisively resolved.
- Show a small or near-zero ratio on rows that were genuinely fragile --
  there is no `CONTRIBUTES`/`HARMFUL` example of this in the current
  dataset (Ackley's fragile row was `INCONCLUSIVE`, which the existing
  TOST-power metric already handles correctly), so this branch should
  construct or identify at least one synthetic case (e.g. via
  `engine/audit/calibration.py`'s existing scaffolds, at a sample size
  chosen to land the CI just barely past the margin) to confirm the metric
  actually distinguishes "barely resolved" from "decisively resolved,"
  not just report a large number on cases already known to be robust.

## Null / alternative hypothesis

**Null (would contradict the hypothesis):** the proposed metric does not
meaningfully distinguish "decisively resolved" from "barely resolved"
`CONTRIBUTES`/`HARMFUL` cases any better than eyeballing the existing CI
against the margin already does -- i.e. this is a real conceptual gap
(the docstring is honestly describing what it computes) but not a
*practically* useful one, since anyone reading a `CONTRIBUTES`/`HARMFUL`
row's CI and margin directly (as `STEPSIZE_SPEC.md`'s writeup already did
by hand) gets the same information without a new field. If so, the
right conclusion is to document the distinction more clearly (e.g. in
`AUDIT_METHODOLOGY.md` §4.2 and in the `MetricVerdict` docstring, noting
`power`/`n_for_target_power` apply specifically to `NULL`/`INCONCLUSIVE`
rows) rather than add a new computed field that doesn't earn its
complexity.

**Alternative (supports the hypothesis):** the metric adds real,
non-redundant decision-relevant information -- e.g. it changes which of
two `CONTRIBUTES` rows with similar-looking CIs a researcher would trust
without a re-run, in a way eyeballing the raw numbers doesn't make
obvious, or it correctly flags a constructed near-boundary case that a
naive glance at the CI would treat as fine. If so, add it to
`engine/audit/statistics.py` and `MetricVerdict` as a genuinely new,
reusable capability -- not just documentation.

## Motivation

`AUDIT_METHODOLOGY.md` §9 explicitly commits to "achieved power" being
part of every verdict's evidentiary record, and this project's own
practice (three separate branches now) shows that commitment is currently
being fulfilled correctly for `NULL`/`INCONCLUSIVE` and only informally,
by manual CI-reading, for `CONTRIBUTES`/`HARMFUL`. That gap is currently
low-stakes because every `CONTRIBUTES`/`HARMFUL` row this project has
produced so far turned out decisive on inspection -- but the whole reason
this audit exists is to not rely on a result "looking" decisive. The next
time a real controller is audited (README item 3, once `naslib` resolves)
or another external pipeline is validated (the deferred ScaffoldSafety/GAIA
half of item 4), a `CONTRIBUTES`/`HARMFUL` verdict that sits closer to its
boundary than Rastrigin's or Ackley's did would currently have no
engine-reported number to say so -- exactly the situation issue #13's
margin-sensitivity analysis exists to catch for `NULL` verdicts, with no
equivalent for the other two verdict types.

## Experimental design

**No new NATS-Bench or scipy evaluations.** This is a statistics/tooling
branch operating on:

1. Already-collected data: `results/basinhopping_audit/`,
   `results/basinhopping_audit_stepsize_scaling/`,
   `results/basinhopping_audit_ackley_power/`,
   `results/basinhopping_audit_rastrigin_power/` -- recomputing the
   proposed metric against each `CONTRIBUTES`/`HARMFUL` row already on
   record, both at their original (sometimes underpowered) `n` and their
   re-run `n` where applicable, to show the metric's value moves as
   expected as `n` increases on a known case (Rastrigin: 60 -> 200;
   Ackley: 60 -> 1971).
2. A small number of constructed cases using
   `engine/audit/calibration.py`'s existing `OracleScaffold` (which the
   module docstring already establishes should read `CONTRIBUTES`) and
   `WastefulScaffold` (`HARMFUL`), run at a couple of different `n` chosen
   specifically to produce at least one CI that lands close to (not deep
   past) its margin -- the "barely resolved" comparison case the real data
   doesn't currently contain. This mirrors `calibration.py`'s own stated
   purpose: known-answer constructions used to test the audit's own
   instruments, not a new scientific claim about scaffolds.

**Implementation, if the hypothesis holds:** add the metric as a new field
on `MetricVerdict` (e.g. `boundary_clearance_ratio` or similar, computed
only when `verdict in (CONTRIBUTES, HARMFUL)`, `None` otherwise) inside
`engine/audit/statistics.py`, alongside (not replacing) the existing
`power`/`n_for_target_power` fields -- those remain correct and useful for
`NULL`/`INCONCLUSIVE` rows and should not be removed or repurposed.

## Metrics

- **The proposed boundary-clearance ratio itself**, computed on every
  `CONTRIBUTES`/`HARMFUL` row in the four existing result artifacts, plus
  the constructed calibration cases.
- **Comparison against the existing `power`/`n_for_target_power` numbers on
  the same rows**, to show explicitly how they diverge (e.g. Rastrigin's
  60-seed row: `n_for_target_power=181` reads as "underpowered" by that
  metric, while the proposed ratio should read as "decisively resolved").
- No statistical hypothesis test in the TOST sense is being run here --
  this branch evaluates a proposed *reporting* metric's behavior on known
  cases, closer in spirit to `calibration.py`'s validation approach than
  to a scaffold-contribution audit.

## Baselines / controls

The "baseline" being compared against is the status quo: reading a
`CONTRIBUTES`/`HARMFUL` row's raw `observed_difference`, `ci_low`,
`ci_high`, and `margin` directly, the way `STEPSIZE_SPEC.md`'s writeup
already did by hand. The question this branch answers is whether a
computed metric adds anything over that manual reading, not whether either
approach beats a different control arm in the usual audit sense.

## Expected outcomes

- **Metric confirmed useful** (hypothesis): implement it in
  `engine/audit/statistics.py`, document the `power`/`n_for_target_power`
  vs. this new field distinction in both the `MetricVerdict` docstring and
  `AUDIT_METHODOLOGY.md` §4.2, and note explicitly that this does not
  retroactively change any verdict already reported (Rastrigin's,
  Ackley's, or Griewank's) -- it changes what's reported going forward.
- **Metric not meaningfully useful** (null): document the
  power-vs-verdict-type distinction in `AUDIT_METHODOLOGY.md` and the
  `MetricVerdict` docstring anyway (the conceptual gap is real even if a
  new computed field isn't warranted), and record in `DECISION_LOG.md` why
  a new field was considered and not added -- this keeps the next person
  from re-discovering the same gap and re-proposing the same fix without
  knowing it was already considered.
- **Metric partially useful** (e.g. useful as a rough heuristic but not
  precise enough to replace manual CI-reading): report as such; a
  `method/` branch doesn't need to produce a merge-worthy feature to be a
  successful branch, per `GIT_WORKFLOW.md`.

## Interpretation plan

- Useful: this becomes a real, reusable capability any future
  `CONTRIBUTES`/`HARMFUL`-producing audit branch can check without
  reinventing the by-hand reasoning `STEPSIZE_SPEC.md` did.
- Not useful / redundant with manual reading: still valuable as a
  documentation fix, closing a real gap between what
  `AUDIT_METHODOLOGY.md` §9 promises ("achieved power... on every
  verdict") and what's actually informative for two of the four verdict
  types.
- Either way: does not touch or reinterpret any already-merged basinhopping
  or NAS results -- this is purely about what gets reported on *future*
  audit runs.
- Says nothing about `naslib`-blocked README item 3 or the deferred
  ScaffoldSafety/GAIA comparison, though it directly prepares for both --
  a real controller audit or an external-pipeline validation is exactly
  where a `CONTRIBUTES`/`HARMFUL` verdict's reliability will next matter
  under real stakes.

## Confounds considered

- **Retrofitting a metric to already-known-good cases risks confirming
  itself trivially.** All four real `CONTRIBUTES`/`HARMFUL` rows in this
  project's history turned out decisive -- a metric could look useful
  merely by agreeing with "yes, decisive" on cases everyone already knows
  are decisive. Mitigation: the constructed near-boundary calibration case
  (via `OracleScaffold`/`WastefulScaffold` at a deliberately chosen small
  `n`) is required specifically to test whether the metric can also say
  "not yet decisive" on a case built to be borderline, not just "yes" on
  cases built to be obvious.
- **Choice of the specific formula is a design decision, not a proven-
  optimal one.** `(clearance) / (CI half-width)` is one reasonable
  normalization; a different one (e.g. distance in raw units, or a formal
  minimum-detectable-effect calculation via `_tost_power`'s superiority-
  test analogue) could be defensible instead. State the choice and its
  rationale rather than presenting it as the only possible design.
- **This does not audit whether `_tost_power`/`required_sample_size` are
  themselves correct** (`engine/audit/`'s test suite already covers that,
  per the README's "165 tests, 98.23% coverage" figure) -- this is about
  whether the *right* power/precision concept is being surfaced for the
  verdict type actually reached, a reporting-layer question, not a
  correctness bug in the existing statistics.

---

## Results

*(Filled in after the experiment runs.)*

## Interpretation

*(Filled in after the experiment runs.)*

## Decision

*(Filled in after the experiment runs.)*
