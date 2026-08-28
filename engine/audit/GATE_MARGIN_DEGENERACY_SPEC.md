<!--
Copied verbatim from GitHub issue #29, per research/GIT_WORKFLOW.md and
research/SPEC_TEMPLATE.md. Section headers below follow SPEC_TEMPLATE.md;
their content is the issue body's corresponding section, unedited.

Named GATE_MARGIN_DEGENERACY_SPEC.md, not SPEC.md: this branch shares
engine/audit/ with issue #23's SPEC.md and issue #27's
MARGIN_DEGENERACY_SPEC.md, neither of which may be overwritten -- per
GIT_WORKFLOW.md's "a completed branch's pre-registered result stands."

STACKED ON PR #28 (issue #27), not on main: this branch's own commits sit
on top of `method/audit-margin-degeneracy`'s branch history because this
question is only answerable once `MetricVerdict.margin_degeneracy` exists,
which is not yet on `main`. Do not merge this branch before PR #28 merges.
-->

# Spec: method/audit-margin-degeneracy-gating

**Branch:** `method/audit-margin-degeneracy-gating`
**Date opened:** 2026-08-28
**Status:** IN PROGRESS
**Issue:** https://github.com/shlokkvaishnav/Null-Scaffold-Audit/issues/29
**Depends on:** PR #28 (issue #27) -- unmerged at time of writing; this
branch stacks on `method/audit-margin-degeneracy`.

## Branch type

method/ -- a new methodological component (verdict-computation logic in
`engine/audit/`, not a new scaffold or domain claim)

## Research question

Issue #27/PR #28 added `MetricVerdict.margin_degeneracy`, computed inside
`arms.audit()` and reported alongside the verdict -- diagnostic only,
explicitly not wired into `_resolve` (`engine/audit/statistics.py`). A
margin-degenerate row today still reports whatever `NULL`/`HARMFUL`/
`INCONCLUSIVE` its floating-point-scale noise happened to produce; only the
label changes. `MARGIN_DEGENERACY_SPEC.md`'s Interpretation section named
this gap explicitly and left it open rather than deciding it inside that
branch.

**Question:** should `margin_degenerate=True` change what a verdict
*asserts*, not just how it's labeled -- e.g. withdraw the verdict to
`INCONCLUSIVE` the way `_guard_vacuous_comparison` (`arms.py`) already
withdraws a verdict for a different structural failure (both arms
identical) -- or is diagnostic-only correct, on the reasoning that a
degenerate margin is still evidence about *something* (the control arm has
converged, which the treatment arm demonstrably has not) and gating would
throw that away?

## Hypothesis

Gating is the more defensible default, mirroring `_guard_vacuous_comparison`'s
own precedent: a verdict computed against a margin with no domain meaning is
not evidence about the wrapper's actual behavior (this is exactly what issue
#25/PR #26's Griewank incident demonstrated -- the same design produced
opposite verdicts across two seed draws). Reporting `HARMFUL` or `NULL`
from noise, even with a diagnostic flag attached, risks being read as an
ordinary finding by anyone who doesn't check `margin_degeneracy` explicitly
-- which is the same risk issue #27 was filed to close for the *detection*
half of this problem, still open for the *reporting* half.

## Null / alternative hypothesis

**Null (would contradict the hypothesis):** gating to `INCONCLUSIVE` throws
away real information in at least one case this project's own history
contains -- e.g. if a margin-degenerate row's *direction* (not magnitude) is
still informative, or if `_guard_vacuous_comparison`'s precedent doesn't
actually transfer (that check fires on a different, arguably stronger,
structural failure -- both arms identical -- not merely one arm's spread
being small relative to the other). Diagnostic-only would then be the
correct design, and this issue should close as "confirmed as-is," not
revised.

**Alternative (supports the hypothesis):** re-labeling every margin-degenerate
row in this project's own history (both Griewank `run_audit.py` readings) to
`INCONCLUSIVE` loses nothing this project has actually relied on -- neither
reading was ever cited as an established finding (`DECISION_LOG.md` already
says so explicitly for both) -- and gating removes a live footgun for future
readers who don't check the diagnostic field.

## Motivation

Closes the one deliberately-open question from issue #27/PR #28, which this
project's own review process flagged rather than silently deciding either
way inside that branch (`MARGIN_DEGENERACY_SPEC.md`'s Interpretation:
"Whether it *should* gate ... is a separate, deliberately unanswered
question"). Matters because the diagnostic-only design currently ships a
verdict a reader has to know to distrust -- exactly the failure mode
`AUDIT_METHODOLOGY.md`'s own standard (`NULL` as a positive finding,
`INCONCLUSIVE` as a refusal to overclaim) exists to prevent.

## Experimental design

No new scaffold or scientific claim -- a design decision about existing
verdict-computation logic, validated against data this project already has.
Two candidate designs to compare directly:

1. **Gate:** when `margin_degeneracy.degenerate` is `True` for a metric,
   withdraw that metric's verdict to `INCONCLUSIVE` inside `arms.py`'s
   `audit()` (after `assess_margin_degeneracy`, mirroring
   `_guard_vacuous_comparison`'s own withdrawal pattern -- keep the interval,
   p-value, and margin on the record, add a `test` string saying why, per
   that function's own docstring precedent).
2. **Stay diagnostic-only:** no change to `_resolve`; instead strengthen
   how the flag surfaces (e.g. require plugin scripts to print/report it
   prominently, or fail CI on an unacknowledged margin-degenerate row in a
   committed result) rather than changing the verdict itself.

Re-run both candidate designs against every row already covered by
`tests/test_audit_margin_degeneracy.py` (14 rows, no new searcher/scaffold
runs) and check which one matches this project's own retrospective judgment
per `DECISION_LOG.md` (both Griewank `run_audit.py` readings already
declared untrustworthy; every other row already trusted).

**Held constant:** `assess_margin_degeneracy`'s own mechanism and threshold
(issue #27) are not touched by this branch -- this decides what happens
*after* it fires, not how it fires.

## Metrics

Whichever verdicts each candidate design would produce for the two known-bad
rows (currently `HARMFUL`/`HARMFUL` for `run_audit.py` Griewank in the
seed-cap-fix reading and PR #16's `INCONCLUSIVE`/`HARMFUL` pair) versus what
this project's own `DECISION_LOG.md` already says should be trusted.

## Baselines / controls

Retrospective, the same way issue #27 validated: every already-published
`plugins/basinhopping_audit/` row is the test set, with trust labels already
established.

## Expected outcomes

- **Gating matches this project's own retrospective judgment cleanly**: ship
  it, update `plugins/basinhopping_audit`'s two affected historical rows'
  interpretation to note the now-mechanical `INCONCLUSIVE` withdrawal.
- **Gating loses real information on at least one row**: report as a
  negative result for gating specifically -- diagnostic-only stays the
  design, and this issue's own value is having checked rather than left the
  question open indefinitely.
- **Neither design is clearly better**: report both trade-offs honestly and
  recommend the more conservative one (gate), consistent with this
  project's general bias toward `INCONCLUSIVE` over an unearned positive
  claim (`AUDIT_METHODOLOGY.md`'s own stated preference).

## Interpretation plan

- Gating validated: closes the reporting half of the gap issue #27 closed
  the detection half of.
- Gating loses information: valuable negative result, and this project's
  general audit design principle (favor `INCONCLUSIVE` over an overclaim)
  gets a documented, tested exception rather than an assumed one.
- No clear winner: state the trade-off plainly rather than picking a default
  by convenience.

## Confounds considered

The known-bad label set is still small (same two Griewank readings issue
#27 validated against) -- any conclusion here inherits that same
overfitting risk, stated explicitly rather than re-litigated as new
evidence. Changing `_resolve`'s output for even one already-published row
is a real behavior change (unlike issue #27's purely additive field) -- must
be called out prominently in the PR, not treated as equivalent in risk to
adding a diagnostic.
