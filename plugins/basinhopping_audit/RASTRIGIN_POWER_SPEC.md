<!--
Copied verbatim from GitHub issue #21, per research/GIT_WORKFLOW.md and
research/SPEC_TEMPLATE.md. Section headers below follow SPEC_TEMPLATE.md;
their content is the issue body's corresponding section, unedited.

Named RASTRIGIN_POWER_SPEC.md, not SPEC.md: this experiment shares
plugins/basinhopping_audit/ with issue #15's SPEC.md, issue #17's
STEPSIZE_SPEC.md, and issue #19's ACKLEY_POWER_SPEC.md, none of which may
be overwritten -- GIT_WORKFLOW.md's "a completed branch's pre-registered
result stands" applies to all three.
-->

# Spec: experiment/basinhopping-rastrigin-power

**Branch:** `experiment/basinhopping-rastrigin-power`
**Date opened:** 2026-08-23
**Status:** IN PROGRESS
**Issue:** https://github.com/shlokkvaishnav/Null-Scaffold-Audit/issues/21

## Branch type

experiment/ -- one specific, scoped experiment

Direct parallel to issue #19 / PR #20, which resolved the same silent
seed-cap defect for Ackley. This issue applies the identical treatment to
Rastrigin, the one other function affected by it.

## Research question

`plugins/basinhopping_audit/run_stepsize_experiment.py` hardcodes
`MAX_SEEDS = 60`. Issue #19 found this had silently capped Ackley's real
sweep at 60 seeds despite that function's own feasibility probe requiring
135 for 80% power -- and a fresh, uncapped re-run (PR #20) resolved Ackley
from `INCONCLUSIVE` to a decisive `HARMFUL`, the opposite of what the
60-seed data suggested was likely.

Reading `results/basinhopping_audit_stepsize_scaling/audit.json` directly:
the same cap silently applied to Rastrigin too, and by a wider margin than
Ackley's -- `required_n_for_80pct_power = 181`, but `seed_count = 60`, a
shortfall of 121 seeds (Ackley's shortfall was 75). Rastrigin is the one
function in this whole line of research whose verdict is this project's
first-ever real, non-synthetic `CONTRIBUTES` result -- currently reported
as `CONTRIBUTES` with diff +17.54, 90% CI [+15.84, +19.22], margin ±1.84
(effect ~9.5x the margin), at n=60. Griewank's own required_n (3) was well
under its actual seed_count (20, the `MIN_SEEDS` floor), so it is not
affected by this defect and does not need re-running.

**Question:** does Rastrigin's `CONTRIBUTES` verdict survive being re-run
at (or near) its own feasibility probe's recommended seed count, uncapped,
the same way issue #19 re-tested Ackley -- or does the 60-seed result
understate the uncertainty the way Ackley's did?

## Hypothesis

Rastrigin's `CONTRIBUTES` verdict persists at adequate power. Unlike
Ackley's case, the 60-seed CI here ([+15.84, +19.22]) already sits entirely
outside the pre-registered margin (±1.84) by a wide margin (~9.5x) and does
not straddle either the margin boundary or zero -- the qualitative pattern
that made Ackley's result fragile (a CI landing close to or across a
decision boundary) is not present here. The `required_n_for_80pct_power`
figure being unmet is a real, documented defect in how the sweep was run,
but it does not by itself imply the specific verdict reached is wrong --
this branch exists to confirm that inference rather than assume it.

## Null / alternative hypothesis

**Null (would contradict the hypothesis):** at adequate power, the verdict
changes away from `CONTRIBUTES` -- to `NULL`, `HARMFUL`, or even
`INCONCLUSIVE` if the fresh pilot's variance estimate comes back
substantially larger than PR #18's (issue #19's own experience shows pilot
estimates can vary meaningfully between 15-seed draws -- PR #20's fresh
pilot found a spread ~3x larger than PR #18's on the literal same Ackley
configuration). This would mean this project's marquee first-ever
`CONTRIBUTES` finding does not hold up under its own pre-registered power
target, which would need updating everywhere that result is currently
described as established.

**Alternative (supports the hypothesis):** the verdict remains
`CONTRIBUTES` at adequate power, with the CI narrowing (as expected with
larger n) while still excluding the margin boundary. This would confirm the
first real positive scaffold finding this project has produced was not an
artifact of underpowering, closing out the last open power-integrity
question in the basinhopping-audit line of research (issues #15, #17, #19).

## Motivation

Issue #19 already established, empirically, that this specific defect
(feasibility probe computes a required `n`, the runner silently caps below
it) is not merely theoretical -- it changed a real verdict (Ackley,
`INCONCLUSIVE` at n=60 to decisive `HARMFUL` at n=1971). That means the
same defect being present on Rastrigin cannot be waved off as "probably
fine because the effect looked large" without checking, especially because
Rastrigin's result is the one this project is most likely to cite going
forward (its first documented real `CONTRIBUTES` verdict, referenced in
both PR #16's and PR #18's own writeups as the uncontested, corroborated
finding of this whole line of work). Leaving a known, already-demonstrated
power defect unchecked on the specific result most likely to be quoted
elsewhere is a bigger integrity gap than leaving it unchecked on a result
that was already flagged `INCONCLUSIVE` and clearly provisional.

## Experimental design

**Unchanged:** Rastrigin (bounds ±5.12, d=10), `L-BFGS-B` local minimizer,
`stepsize = 0.5` (Rastrigin's own domain-scaled value, unchanged from PR
#18 -- by construction equal to its original PR #16 value), the
`LocalMinimizerRestart` control and budget-matching procedure (`niter+1`
local-minimizer calls per arm, `niter=50`), the 25%-of-control-spread margin
convention.

**Changed, mirroring issue #19's design exactly:** run a fresh feasibility
pilot on this exact configuration, on seeds disjoint from every prior block
used anywhere in this line of research so far (PR #16's pilot 1000-1014 and
real sweep 0-59; PR #18's pilot 2000-2014 and real sweep 10000-10059; PR
#20's pilot 3000-3014 and real sweep -- whatever range issue #19 landed on,
confirm from `results/basinhopping_audit_ackley_power/audit.json` before
picking a new block). Compute `required_sample_size` from the fresh pilot's
spread, and run the real sweep at that recommended `n`, uncapped -- no
`MAX_SEEDS`-style ceiling. If the fresh estimate differs materially from PR
#18's `181`, use the fresh one and state the discrepancy, exactly as issue
#19's writeup did for Ackley's 3x pilot-to-pilot difference.

**Only Rastrigin:** Ackley is already resolved (issue #19) and Griewank was
never under-capped (required_n=3 < seed_count=20) -- neither needs
re-running here.

## Metrics

Identical to the rest of this line: best objective value (continuous, BCa
bootstrap TOST, lower-is-better), distance to the known global optimum
(secondary), degeneracy pre-check, selection ceiling (re-confirm at the new
seed count rather than assuming PR #16/#18's ceiling-of-0.0 carries over
unchanged).

## Baselines / controls

Same as the rest of this line -- budget-matched independent restarts of
the identical `L-BFGS-B` local minimizer, same `niter+1`-call budget, same
`stepsize=0.5` for the treatment arm's internal minimizer configuration.

## Expected outcomes

- **`CONTRIBUTES` persists at adequate power** (hypothesis): confirms this
  project's first real positive scaffold finding is power-robust, not an
  artifact of the 60-seed cap; closes out the last open integrity question
  in this line of research.
- **Verdict changes** (contradicts hypothesis): would be a significant,
  must-report finding -- this project's most citable positive result would
  need to be corrected or withdrawn, and any place it has already been
  described as established (this issue's own text, PR #16/#18's writeups,
  any future summary) would need a follow-up correction noted in
  `DECISION_LOG.md`, not just silently updated.
- **Fresh pilot variance differs substantially from PR #18's**, changing
  the target `n` a lot (as happened for Ackley): report the discrepancy
  explicitly and use the fresh estimate regardless of which direction it
  moves the required `n`, per the same rule issue #19 already established
  as this project's convention.

## Interpretation plan

- `CONTRIBUTES` confirmed at adequate power: state explicitly, in this
  branch's own writeup, that Rastrigin's result is now verified
  power-robust -- this becomes the reference point anyone should cite
  going forward rather than PR #16's original (underpowered) 60-seed run.
- Verdict changes: do not treat this as ARCHIVE-worthy or embarrassing --
  per `GIT_WORKFLOW.md`, a branch that corrects a prior result is exactly
  as valid a MERGE as one that confirms it, and the correction itself is
  the useful output. Add a `DECISION_LOG.md` entry describing the reversal
  and why the earlier result was wrong (underpowered, not necessarily
  incorrect in direction, but not decisively established).
- Either way, this closes the power-integrity thread across all three
  functions in the basinhopping-audit line (issues #15, #17, #19, and this
  one) -- after this, that line of research needs no further re-running
  under the same defect; any future work on it should start from a design
  question, not a leftover implementation gap.
- Says nothing about the deferred ScaffoldSafety/GAIA cross-methodology
  comparison, `naslib`-blocked README item 3, or the general question of
  whether `plugins/basinhopping_audit`'s runner scripts should be fixed to
  never silently cap below a feasibility probe's recommendation (a
  `method/`-shaped question, separate from re-verifying this one already-
  affected result).

## Confounds considered

- **Pilot-to-pilot variance changing the target `n` substantially**, as
  demonstrated for Ackley (3x difference between two 15-seed pilots on the
  identical configuration) -- expected here too; use the fresh estimate
  as-is per the pre-registered rule, not the smaller/more convenient of the
  two.
- **Seed disjointness across four prior blocks** (PR #16's two, PR #18's
  two, PR #20's two) must be verified by test, matching the precedent
  issue #19 itself set -- an overlapping range would reintroduce the
  shared-seed-stream confound flagged back in issue #15.
- **This is not "keep re-running until CONTRIBUTES survives."** The
  higher `n` is justified by the same pre-registered, already-verified
  defect issue #19 used to justify Ackley's re-run -- not by a general
  license to add seeds specifically because this result is one people
  would prefer to keep. If the verdict changes, that stands as the answer,
  per Expected outcomes and Interpretation plan above.
- **Stepsize, local minimizer, and function/dimension are not revisited
  here** -- only the seed count changes, exactly mirroring issue #19's
  scope discipline.

---

## Results

*(Filled in after the experiment runs.)*

## Interpretation

*(Filled in after the experiment runs.)*

## Decision

*(Filled in after the experiment runs.)*
