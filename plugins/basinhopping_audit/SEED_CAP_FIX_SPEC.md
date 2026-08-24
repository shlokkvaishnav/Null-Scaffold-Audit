<!--
Copied verbatim from GitHub issue #25, per research/GIT_WORKFLOW.md and
research/SPEC_TEMPLATE.md. Section headers below follow SPEC_TEMPLATE.md;
their content is the issue body's corresponding section, unedited.

Named SEED_CAP_FIX_SPEC.md, not SPEC.md: this branch shares
plugins/basinhopping_audit/ with issue #15's SPEC.md, issue #17's
STEPSIZE_SPEC.md, issue #19's ACKLEY_POWER_SPEC.md, and issue #21's
RASTRIGIN_POWER_SPEC.md, none of which may be overwritten -- this branch
DOES modify run_audit.py and run_stepsize_experiment.py's code (that is
its whole point), but not their SPEC files' text, per GIT_WORKFLOW.md's
"a completed branch's pre-registered result stands."
-->

# Spec: method/basinhopping-remove-seed-cap

**Branch:** `method/basinhopping-remove-seed-cap`
**Date opened:** 2026-08-24
**Status:** IN PROGRESS
**Issue:** https://github.com/shlokkvaishnav/Null-Scaffold-Audit/issues/25

## Branch type

method/ -- a new methodological component (a runner-script convention/
protocol, not a new scaffold or domain claim)

## Research question

`plugins/basinhopping_audit/run_audit.py` (PR #16, the entry point for this
plugin) and `plugins/basinhopping_audit/run_stepsize_experiment.py` (PR #18)
both hardcode `MAX_SEEDS = 60`, silently truncating the real sweep below
whatever `required_sample_size` recommends for 80% TOST power. This defect
was found and fixed -- but only locally, per-function, in two follow-up
scripts:

- `run_ackley_power_experiment.py` (issue #19): "no `MAX_SEEDS` here, per
  SPEC.md" -- fixed Ackley's specific run, uncapped.
- `run_rastrigin_power_experiment.py` (issue #21): same fix, applied to
  Rastrigin's specific run.

Both fixes are correct for the runs they cover. Neither touched
`run_audit.py` or `run_stepsize_experiment.py` themselves -- both still
contain `MAX_SEEDS = 60` today, on `main`. Three separate PR writeups
(#16/#18/#20's SPEC files) have now independently named "fix the general
pattern" as a follow-up for the researcher, without it being filed. This
issue is that filing.

**Question:** should `plugins/basinhopping_audit`'s original entry points
(`run_audit.py`, `run_stepsize_experiment.py`) be brought in line with the
uncapped convention the two power-specific scripts already established --
and, since this pattern (compute a feasibility-recommended `n`, then
silently cap it for convenience) is exactly the kind of thing that
re-appears whenever a script is copied as a template for a new function or
plugin, should the cap-vs-no-cap decision be pulled out of ad hoc
per-script constants into one shared, explicit convention new plugins can
follow without rediscovering this defect from scratch?

## Hypothesis

Removing `MAX_SEEDS` from `run_audit.py` and `run_stepsize_experiment.py`
(matching the two power-specific scripts) and re-running both is safe and
uneventful for two of the three affected rows, and produces the same
already-known-correct answer for the third: Rastrigin's original `run_audit.py`
row (PR #16, `CONTRIBUTES` at n=60) and `run_stepsize_experiment.py` row
(PR #18, `CONTRIBUTES` at n=60) should both resolve the same way issue #21's
already-uncapped Rastrigin re-run did (`CONTRIBUTES`, confirmed at n=200).
Griewank's `run_audit.py`/`run_stepsize_experiment.py` rows were never
actually affected (required_n was below the cap in both) and should be
unchanged. Ackley's `run_audit.py` row (PR #16, `HARMFUL` at a
stepsize=0.5 that issue #17 later showed was mismatched) is not expected to
change in a way that matters -- it's already superseded by issue #17's
domain-scaled re-run regardless of this fix -- but re-running it uncapped
should still be reported rather than assumed, for completeness of this
branch's own claim to have applied the fix everywhere it was found.

## Null / alternative hypothesis

**Null (would contradict the hypothesis):** re-running any of these rows
uncapped produces a different verdict than already established elsewhere
in this project's history for the same function/stepsize configuration --
would mean something beyond the seed cap (e.g. an unnoticed difference
between `run_audit.py`'s pilot/margin logic and the power-specific scripts'
logic) is also affecting these results, and needs its own investigation
before closing this out as "just the cap."

**Alternative (supports the hypothesis):** every re-run reproduces the
verdict already on record from the corresponding power-specific
experiment (where one exists) or remains unchanged (Griewank, never
affected) -- confirming the cap was the entire defect, with nothing else
hiding behind it.

## Motivation

This is pure integrity bookkeeping for a defect this project has already
proven is real and verdict-changing (issue #19's Ackley reversal), not a
new scientific question -- but leaving it half-fixed is its own risk: a
future reader who finds `MAX_SEEDS=60` still present in the two original
scripts and absent from the two newer ones has no way to tell, without
reading four separate SPEC files, whether that's an intentional design
choice or an unfixed instance of a known bug. Consolidating the fix (and
ideally the convention itself, so a fifth script added later doesn't
reintroduce the same cap) closes that ambiguity and matches
`AUDIT_METHODOLOGY.md` §9's commitment to every verdict's provenance being
legible on its own, not contingent on cross-referencing prior PRs.

It also directly protects this plugin's reusability: if `plugins/
basinhopping_audit`'s pattern is ever copied as a starting point for
auditing a different external scaffold (the still-deferred ScaffoldSafety/
GAIA half of README item 4, or any future `experiment/`), a new author
copying `run_audit.py` verbatim would silently inherit `MAX_SEEDS=60`
again -- exactly the failure mode this issue exists to close off.

## Experimental design

**No new domain content.** Two parts:

1. **Fix:** remove `MAX_SEEDS` from `run_audit.py` and
   `run_stepsize_experiment.py`, matching the pattern already used in
   `run_ackley_power_experiment.py`/`run_rastrigin_power_experiment.py`
   (a `MIN_SEEDS` floor only, no ceiling). Consider factoring the
   seed-count-from-feasibility-probe logic into one shared helper used by
   all four scripts, so the convention lives in one place rather than
   being copy-pasted per script (a real `method/`-shaped contribution, not
   just a constant deletion) -- but only if doing so doesn't disturb any
   already-reported result's exact seed selection; if a shared helper would
   change seed derivation for already-published rows, keep the fix local
   per-script instead and note why a shared helper was not pursued.
2. **Re-run:** every row in `results/basinhopping_audit/` (PR #16) and
   `results/basinhopping_audit_stepsize_scaling/` (PR #18) that was
   actually affected by the cap (compare each row's own
   `required_n_for_80pct_power` against its `seed_count`, the same check
   this issue's Motivation section already applied) -- uncapped, on fresh
   seeds disjoint from every prior block used anywhere in this line of
   research (PR #16, #18, #20, #22's pilot and real-sweep ranges).

**Held constant:** every other design element already fixed across this
line (functions, dimension, local minimizer, stepsize values as configured
in each original script, margin convention).

## Metrics

Verdict per re-run row, compared against the already-established verdict
for the same function/stepsize configuration elsewhere in this project's
history (issue #21's uncapped Rastrigin result for the `run_
stepsize_experiment.py` Rastrigin row; no prior uncapped precedent exists
for `run_audit.py`'s original fixed-stepsize=0.5 rows, so those are new
confirmations, not re-confirmations).

## Baselines / controls

Not a new scaffold-vs-searcher comparison -- the "baseline" here is each
row's own already-published, capped-`n` result, and this branch checks
whether removing the cap changes it.

## Expected outcomes

- **All re-runs confirm existing verdicts**: closes this out cleanly;
  update `run_audit.py`/`run_stepsize_experiment.py` permanently, note in
  the writeup that this account for every currently-known instance of the
  `MAX_SEEDS` defect in this plugin.
- **A re-run changes a verdict** not already known to be affected (i.e.
  something beyond Ackley/Rastrigin's already-documented cases): report
  prominently -- this would mean the cap's effect wasn't fully understood
  from issues #19/#21 alone, and whatever changed needs its own
  interpretation, not a one-line note.
- **Shared-helper refactor turns out to risk changing seed derivation** for
  an already-published row: keep the fix local per-script instead, as
  Experimental Design already allows for, and say so explicitly rather than
  forcing a refactor that would make an already-reported result
  irreproducible from its own SPEC file.

## Interpretation plan

- Confirms existing verdicts: this branch's own value is closing an
  already-known gap cleanly, not producing a new finding -- report it as
  exactly that (a completeness/integrity branch), not oversold as new
  science.
- Changes a previously-unflagged verdict: treat with the same seriousness
  `GIT_WORKFLOW.md` gives any reversal (issue #19's precedent) -- a
  `DECISION_LOG.md` entry, explicit correction of whatever summary
  currently describes the affected row as established.
- Either way: this closes the `MAX_SEEDS` defect across every script in
  `plugins/basinhopping_audit/` where it's currently known to exist. It
  does not itself resolve `naslib`-blocked README item 3 or the deferred
  ScaffoldSafety/GAIA comparison, though a clean, defect-free plugin
  pattern is exactly what either of those would want to build on next.

## Confounds considered

- **Re-running is not "re-run until the number I want appears."** Every
  row targeted here is targeted because it's already been shown, by the
  same objective criterion (required_n vs. actual seed_count), to be
  affected by a known defect -- not chosen because its current verdict is
  inconvenient. State the affected-row list up front, before re-running
  any of them, so the target set can't be adjusted after seeing results.
- **A shared-helper refactor could inadvertently change seed derivation**
  for rows not being re-run in this branch (e.g. if it also touches
  `run_ackley_power_experiment.py`/`run_rastrigin_power_experiment.py`)
  -- if pursued, verify by test that it reproduces those two scripts'
  already-published seed counts/results exactly before trusting anything
  new built on top of it.
- **This does not itself establish a general policy for future plugins**
  beyond fixing what's already broken in this one -- whether `engine/audit/`
  itself should provide the "don't silently cap below required_n"
  convention as shared infrastructure (rather than each plugin re-solving
  it) is a broader, separate question this branch can note but shouldn't
  try to settle unilaterally.

---

## Results

*(Filled in after the experiment runs.)*

## Interpretation

*(Filled in after the experiment runs.)*

## Decision

*(Filled in after the experiment runs.)*
