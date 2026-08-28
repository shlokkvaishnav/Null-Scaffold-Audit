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

## Results

**Outcome: the Alternative hypothesis, cleanly -- same pattern as issue #27
itself.** Implemented Candidate 1 (gate): `arms.py`'s new
`_guard_margin_degeneracy` withdraws a metric's verdict to `INCONCLUSIVE`
whenever `margin_degeneracy.degenerate` is `True`, mirroring
`_guard_vacuous_comparison`'s existing precedent exactly (interval/p-value/
margin retained, a `test` string added, `boundary_clearance_ratio` cleared
per that field's own documented invariant).

**One design detail not anticipated in the issue's Experimental design:
ordering relative to `_holm_correct` matters here in a way it didn't for
`_guard_vacuous_comparison`.** `_guard_vacuous_comparison` withdraws every
metric at once or none (it's a whole-arms condition), so running it after
Holm correction is harmless. `_guard_margin_degeneracy` is per-metric --
had it run after Holm correction instead, a margin-degenerate metric's own
meaningless `CONTRIBUTES`/`HARMFUL` claim would still have counted toward
the Holm family size for any other metric audited alongside it in the same
`audit()` call, weakening that metric's correction for no real reason.
Placed *before* `_holm_correct` instead, so a withdrawn metric is already
`INCONCLUSIVE` (and excluded from `_holm_correct`'s own
`v.verdict in {CONTRIBUTES, HARMFUL}` filter) by the time the correction
family is built. No row in this project's current history has more than
one metric per `audit()` call, so this had no effect on any already-
published result -- but the ordering is still the correct general design,
not an untested convenience.

Retrospectively validated (`tests/test_audit_margin_degeneracy_gating.py`,
same 14 rows as issue #27, no new searcher/scaffold runs): reconstructed
each row's real `MetricVerdict` via `equivalence_verdict` +
`assess_margin_degeneracy` from its own committed raw arm data, ran it
through `_guard_margin_degeneracy`, and compared the result against
`DECISION_LOG.md`'s established trust labels --

| Row | Gated? | Resulting verdict | Matches DECISION_LOG.md? |
|---|---|---|---|
| `run_audit.py` Griewank, PR #16 | yes | `INCONCLUSIVE` | yes -- already `INCONCLUSIVE` |
| `run_audit.py` Griewank, issue #25 | yes | `INCONCLUSIVE` (was `HARMFUL`) | yes -- entry explicitly says don't trust it |
| every other row (12 total) | no | unchanged | yes -- none are margin-degenerate |

Zero false positives, zero false negatives, against the full 14-row set --
not just the two Griewank rows. Also confirmed live, not only
retrospectively: `test_audit_withdraws_a_genuinely_margin_degenerate_live_run`
runs a synthetic scaffold with a genuinely frozen control arm through
`arms.audit()` end-to-end and checks the withdrawal fires through the real
code path, not just the standalone helper.

## Interpretation

Closes the reporting half of the gap issue #27 closed the detection half
of: a margin-degenerate row no longer ships a verdict a reader has to know
to distrust. Concretely, this means a *future*, single, unaccompanied
`run_audit.py` run on Griewank (no side-by-side comparison to prompt
suspicion, the exact situation that let issue #25's incident go unnoticed
at first) will report `INCONCLUSIVE` mechanically, not `HARMFUL` or
`INCONCLUSIVE` decided by which seeds happened to be drawn.

**What this does NOT establish.** That `1e-4` (`DEFAULT_RATIO_THRESHOLD`,
issue #27) is the right threshold in general -- this branch validates
gating's *logic* given that threshold's existing output, not the threshold
itself, which was already validated on the same 14 rows in issue #27 and is
unchanged here. Also does not retroactively rewrite any committed
`audit.json` artifact: `results/basinhopping_audit_seed_cap_fix/run_audit/
audit.json` still shows Griewank's row as `HARMFUL` on disk, exactly as PR
#26 committed it, because regenerating it would require an actual re-run
(out of scope, per this branch's own "no new searcher/scaffold runs"
design, same as issue #27's). The gate applies to `audit()` going forward,
not to historical artifacts already on record -- readers of that specific
file should cross-reference `DECISION_LOG.md`, same as before this branch,
until a future run regenerates it.

Does not change the project's research thesis ("Does the Scaffold Earn Its
Keep") -- pure audit-infrastructure hardening, no new scaffold-contribution
claim, matching issue #27's own conclusion.

## Decision (implementer's self-assessment)

- [x] MERGE
- [ ] ARCHIVE
- [ ] REVISE
- [ ] ABANDON
- [ ] REPRODUCE

Why: all nine `GIT_WORKFLOW.md` criteria are met. Scientific relevance:
closes the one deliberately-deferred question from issue #27/PR #28.
Correctness: retrospectively validated against the full 14-row set with
zero false positives/negatives, plus a live end-to-end confirmation, not
only the standalone helper. Experimental validity: both candidate designs
from the issue were genuinely considered; gating was chosen because it
matched retrospective judgment exactly, not by default. Reproducibility:
the ordering rationale (before Holm, not after) and the full validation
table are recorded here, and `tests/test_audit_margin_degeneracy_gating.py`
re-derives the same numbers from the same committed artifacts on every run.
Documentation: `AUDIT_METHODOLOGY.md` §4.3a updated to describe gating, not
just detection; `MetricVerdict.margin_degeneracy`'s and
`margin_degeneracy.py`'s docstrings updated to stop claiming
"diagnostic only," which this branch makes false. Interpretation: what
this branch does and does not establish (particularly, that historical
artifacts are not retroactively rewritten) is stated explicitly. Research
integrity: this is flagged as a real behavior change, not equated in risk
to issue #27's purely additive field, per the Confounds section above.
Integration: matches `_guard_vacuous_comparison`'s existing withdrawal
pattern exactly, placed in the one part of `audit()` where ordering
actually needs to differ from that precedent, with the reason stated.
Evidence: 14/14 rows correctly handled, 0 already-trusted rows affected,
296/296 tests pass (276 pre-existing + 20 new).

**Depends on PR #28 (issue #27) merging first** -- this branch is stacked
on `method/audit-margin-degeneracy`, not `main`. A reviewer should not
approve this for merge until #28 is merged and this branch is rebased onto
the resulting `main`.
