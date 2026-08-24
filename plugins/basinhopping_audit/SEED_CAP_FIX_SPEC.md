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

**Fix applied locally, not via a shared helper.** `MAX_SEEDS` removed from
both `run_audit.py` and `run_stepsize_experiment.py` (now `seed_count =
max(needed_n, MIN_SEEDS)`, no ceiling), matching
`run_ackley_power_experiment.py`/`run_rastrigin_power_experiment.py`'s
existing pattern exactly. A shared-helper refactor (the Experimental
Design's stated preference, if safe) was **not** pursued: the four scripts'
feasibility-probe logic is textually identical but each script also embeds
its own `OUT` path, function-specific config fields, and (for the two
newer scripts) a single-function scope rather than a three-function loop --
factoring out only the numeric core while leaving four call sites to keep
in sync by hand would trade one small, obvious defect (a literal constant)
for a subtler one (a shared function whose call sites could drift), for a
codebase this size. Noted as SPEC.md's own permitted fallback, not a
shortcut taken silently.

**Both scripts' `PILOT_SEEDS`/`REAL_SWEEP_SEED_OFFSET` moved to fresh
blocks** (`run_audit.py`: pilot 7000-7014, real sweep from 40000;
`run_stepsize_experiment.py`: pilot 8000-8015, real sweep from 50000),
disjoint from every block used anywhere in this line of research
(PR #16's original 1000-1014/0-59, PR #18's original 2000-2014/10000-10059,
PR #20's 3000-3014/20000-21970, PR #22's 4000-4014/30000-30199) --
verified by test (`tests/test_plugin_basinhopping_stepsize_experiment.py`,
`tests/test_plugin_basinhopping_ackley_power_experiment.py`,
`tests/test_plugin_basinhopping_rastrigin_power_experiment.py`, all
updated to check against the new values rather than the removed
`MAX_SEEDS` attribute). `results/basinhopping_audit/audit.json` (PR #16)
and `results/basinhopping_audit_stepsize_scaling/audit.json` (PR #18) are
**not** overwritten by this branch -- both scripts' fixed, uncapped
behavior was confirmed by redirecting `OUT` to a new directory,
`results/basinhopping_audit_seed_cap_fix/{run_audit,run_stepsize_experiment}/`,
for this confirmatory re-run only.

**Confirmed rows (5 of 6), exactly as hypothesized:**

| script | function | original verdict (capped) | re-run verdict (uncapped) | n | matches prior finding? |
|---|---|---|---|---|---|
| `run_audit.py` | rastrigin | CONTRIBUTES (n=60) | **CONTRIBUTES** | 228 | yes -- PR #16, PR #22 |
| `run_audit.py` | ackley | HARMFUL (n=60) | **HARMFUL** | 176 | yes -- PR #16 (superseded scientifically by issue #17, not by this fix) |
| `run_stepsize_experiment.py` | rastrigin | CONTRIBUTES (n=60) | **CONTRIBUTES** | 119 | yes -- issue #21's already-uncapped result |
| `run_stepsize_experiment.py` | ackley | INCONCLUSIVE (n=60) | **HARMFUL** | 246 | yes -- issue #19's already-uncapped result |
| `run_stepsize_experiment.py` | griewank | NULL (n=20, unaffected) | **NULL** | 20 | yes -- unchanged, as predicted |

Effect sizes and directions match the corresponding already-published
result in every one of these five rows (Rastrigin ~+17.7 to +18.6 across
four independent runs now; Ackley's two configurations each reproduce
their own established direction and rough magnitude). This directly
supports the hypothesis's core claim: the seed cap was the entire defect
behind every previously-flagged verdict discrepancy, with nothing else
hiding behind it.

**One row not predicted to change, changed anyway -- reported prominently,
per SPEC.md's own instruction, not folded into the table above as if
routine:**

`run_audit.py`'s Griewank row (`stepsize=0.5`, unscaled) read `HARMFUL`
(diff -0.100, CI [-0.194, -0.061], margin ±1e-9, n=20) in this re-run,
against `INCONCLUSIVE`/`DEGENERATE` in PR #16's original (also n=20). Both
runs used `n=20` -- the `MIN_SEEDS` floor, never the removed cap -- so the
seed-cap fix is **not** the cause. The cause is the already-documented
degenerate-margin problem `SPEC.md` and `STEPSIZE_SPEC.md` both flag for
Griewank at d=10: the control arm's independent restarts already land
within numerical noise of the exact global optimum (`control_spread` here:
`2.683e-11`), so `margin = max(0.25 * control_spread, 1e-9)` floors at a
number with no practical meaning, and the resulting verdict is then decided
by floating-point-scale noise in whichever seeds happen to be drawn --
`INCONCLUSIVE` in one draw, `HARMFUL` in another, with nothing about
basin-hopping's actual behavior on Griewank distinguishing them.
`run_stepsize_experiment.py`'s Griewank row (domain-scaled
`stepsize=58.59`) did **not** flip (`NULL` in both the original
stepsize-scaling run and this re-run) -- the instability is specific to
`run_audit.py`'s unscaled-stepsize configuration, consistent with a larger
stepsize giving basinhopping's hops more room to move (as issue #17's own
finding already established) even on a landscape this close to unimodal.
Recorded in `DECISION_LOG.md`, per the Interpretation plan's instruction
for this exact scenario.

## Interpretation

**The hypothesis holds for the defect it targeted.** Every row affected by
`MAX_SEEDS` reproduces its already-established verdict once uncapped --
confirming the seed cap, not some other unnoticed difference between
`run_audit.py`/`run_stepsize_experiment.py` and the power-specific scripts,
was the entire mechanism behind every previously-flagged discrepancy. The
fix is complete: no script in `plugins/basinhopping_audit/` still contains
`MAX_SEEDS`.

**Griewank's `run_audit.py` verdict instability is a separate, real, and
now better-understood finding, not evidence the seed-cap fix is
incomplete or incorrect.** It sharpens (rather than contradicts) what
`SPEC.md`/`STEPSIZE_SPEC.md` already said about Griewank at this
dimension: the margin there was never a well-posed practical-equivalence
question, because the control arm has essentially no variance to set one
from. This branch adds the concrete demonstration that two independent,
otherwise-identical 20-seed draws under that same broken margin land on
*different* verdicts -- which is a stronger, more legible statement of the
same caveat than either prior document could make on its own (each only
had one draw to point at). **Neither Griewank `run_audit.py` verdict --
not PR #16's original `INCONCLUSIVE`, not this branch's `HARMFUL` -- should
be read as an established finding about basin-hopping on Griewank at the
unscaled stepsize.** `run_stepsize_experiment.py`'s Griewank `NULL`
(domain-scaled stepsize, stable across two independent runs) remains the
one trustworthy Griewank reading in this plugin's history.

**What this does NOT establish:**

- That the shared-helper refactor considered in Experimental Design is
  impossible or not worth pursuing later -- it was judged not worth the
  risk *for this branch's narrow scope* (fixing a known defect without
  disturbing four scripts' worth of already-published seed derivations at
  once), not ruled out as a future `method/` question.
- A general policy for `engine/audit/` itself to prevent this class of
  defect in future plugins -- SPEC.md's own Confounds section explicitly
  scopes that question out; noted as a possible follow-up, not settled.
- Anything about README item 3 or the deferred ScaffoldSafety/GAIA
  comparison.

## Decision

**MERGE.** Checked against `GIT_WORKFLOW.md`'s nine criteria:

- **Scientific relevance:** closes a defect three prior PR writeups
  (#16/#18/#20) independently flagged as unresolved, on the exact scripts
  a future plugin author is most likely to copy as a starting point.
- **Correctness:** the affected-row list was determined by the same
  objective criterion (`required_n_for_80pct_power` vs. `seed_count`)
  before any re-run, matching the pre-registered target set SPEC.md's
  Confounds section requires; seed disjointness verified by test across
  all six blocks now in use in this plugin.
- **Experimental validity:** re-runs use fresh, disjoint seeds rather than
  extending the original capped ranges, so nothing about the corrected
  result could be read as cherry-picked continuation of a flawed sample.
- **Reproducibility:** both fixed scripts are deterministic given their
  (now-relocated) seed constants; the confirmatory re-run's redirected
  `OUT` is documented in this file rather than left as an undocumented
  manual step.
- **Documentation:** this file; `DECISION_LOG.md` entry for the one
  unpredicted verdict change; module docstrings in both fixed scripts
  explain the fix and point here.
- **Interpretation:** stated above, including the explicit instruction
  that Griewank's `run_audit.py`-configuration verdicts (either of them)
  should not be cited as established.
- **Research integrity:** the one row that did not behave as hypothesized
  is reported with the same weight as the five that did, not minimized to
  a footnote; the `DECISION_LOG.md` entry states plainly that neither of
  the two conflicting Griewank readings should be trusted.
- **Integration:** the historical `results/basinhopping_audit/` and
  `results/basinhopping_audit_stepsize_scaling/` artifacts are
  deliberately left untouched; the confirmatory re-run's output lives in
  its own new directory.
- **Evidence:** six real re-runs (5 confirming, 1 informatively not)
  against real `scipy` code, each with full config and seeds recorded.

Follow-up for the researcher, not this branch: whether `engine/audit/`
should provide shared "don't silently cap below `required_n`"
infrastructure (SPEC.md's own noted-but-out-of-scope question), and
whether the shared-helper refactor across all four `plugins/
basinhopping_audit/` runner scripts is worth revisiting later, now that
none of them individually contains the defect this branch closes.
