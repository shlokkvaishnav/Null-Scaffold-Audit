<!--
Copied verbatim from GitHub issue #19, per research/GIT_WORKFLOW.md and
research/SPEC_TEMPLATE.md. Section headers below follow SPEC_TEMPLATE.md;
their content is the issue body's corresponding section, unedited.

Named ACKLEY_POWER_SPEC.md, not SPEC.md: this experiment shares
plugins/basinhopping_audit/ with issue #15's SPEC.md and issue #17's
STEPSIZE_SPEC.md, neither of which may be overwritten -- GIT_WORKFLOW.md's
"a completed branch's pre-registered result stands" applies to both.
-->

# Spec: experiment/basinhopping-ackley-power

**Branch:** `experiment/basinhopping-ackley-power`
**Date opened:** 2026-08-23
**Status:** IN PROGRESS
**Issue:** https://github.com/shlokkvaishnav/Null-Scaffold-Audit/issues/19

## Branch type

experiment/ -- one specific, scoped experiment

This is a single re-run of one function (Ackley) from the design already
implemented in `plugins/basinhopping_audit/` (issue #17 / PR #18), at a
higher seed count, to resolve a verdict that branch itself left explicitly
open. `STEPSIZE_SPEC.md`'s own closing section names this exact follow-up:
"whether a higher-power Ackley re-run (larger `n`, informed by this
experiment's own achieved power) is worth its own `experiment/` issue to
resolve the still-open `INCONCLUSIVE`."

## Research question

PR #18 (domain-scaled stepsize re-run) reported Ackley's verdict as
`INCONCLUSIVE` (CI [-0.78, +0.14], straddling zero) at a domain-scaled
stepsize of 3.2 -- resolving the question "was PR #16's `HARMFUL` label on
Ackley reliable?" (no) but explicitly not the question "what is the correct
verdict on Ackley at a properly-scaled stepsize?" (open).

Reading `results/basinhopping_audit_stepsize_scaling/audit.json` directly:
Ackley's own feasibility probe computed `required_n_for_80pct_power = 135`,
but `plugins/basinhopping_audit/run_stepsize_experiment.py` hardcodes
`MAX_SEEDS = 60` (`seed_count = min(max(needed_n, MIN_SEEDS), MAX_SEEDS)`),
so the real sweep silently ran at 60 seeds -- well under what the
experiment's own pre-registered power target called for. This is not a
data-quality problem (the 60-seed result is valid as far as it goes) but an
avoidable power shortfall: the design's own feasibility probe correctly
identified how many seeds were needed and the runner capped it anyway,
apparently for uniformity with Rastrigin/Griewank's shared script rather
than for any real cost reason -- Ackley's function evaluations are as cheap
as the other two (no training, sub-millisecond per call).

**Question:** at the same domain-scaled stepsize (3.2) and otherwise
identical design (same function, dimension, local minimizer, budget-
matching procedure), does Ackley's verdict resolve to a determinate outcome
(`NULL` or `CONTRIBUTES`) once run at (or near) the seed count its own
feasibility probe already called for, rather than the capped 60?

## Hypothesis

Running at n ≈ 135-150 (informed by a fresh feasibility probe on this exact
configuration, not simply reusing PR #18's `required_n_for_80pct_power`
number without re-checking it) resolves Ackley to `NULL`: PR #18's point
estimate was already small relative to its margin (diff -0.52 vs margin
±0.093 -- actually larger than the margin in raw magnitude, but the CI's
own width, ~0.92, is what made it straddle zero) and the CI's failure to
land cleanly on either side of zero looks more like an underpowered read of
a genuinely small effect than a sign of a large one hiding under noise.
This is a directional guess from the existing 60-seed data, not a claim
this branch is being run to confirm -- `CONTRIBUTES` or `HARMFUL` are both
live alternatives the higher-power run could actually establish instead.

## Null / alternative hypothesis

**Null (would contradict the hypothesis):** at adequate power, the verdict
is `HARMFUL` or `CONTRIBUTES` rather than `NULL` -- meaning the 60-seed
run's near-zero-straddling CI was misleading about the eventual sign, not
merely underpowered around a true near-zero effect.

**Alternative (supports the hypothesis):** the verdict resolves to `NULL`
at adequate power, with the CI narrowing around a small effect that stays
inside the margin.

**Also possible, and explicitly not a failure of this branch:** the verdict
remains `INCONCLUSIVE` even at the higher seed count -- would mean the
effect's true variance is larger than the pilot probe estimated, and
should be reported as such (with the achieved power) rather than treated as
this experiment not having "worked."

## Motivation

This is the last unresolved thread from the basinhopping line of research
(issues #15, #17): Rastrigin's `CONTRIBUTES` and Griewank's `NULL` are both
now determinate and well-powered; Ackley alone is still an open question,
and it's open for an avoidable reason (a hardcoded seed cap that ignored
the design's own power calculation) rather than a fundamental limit --
function evaluations here cost essentially nothing, so there's no real
tradeoff being made by capping at 60 the way there might be for an
expensive-to-evaluate domain. Leaving it unresolved when resolving it is
cheap would mean the eventual writeup of this project's basinhopping
findings has to report "unknown" on one of three functions for a reason
that has nothing to do with the science.

This also flags a real, generalizable issue for whoever reuses
`plugins/basinhopping_audit`'s pattern later: a feasibility probe's
recommended `n` should not be silently capped below what it asks for
without a stated reason. That's worth noting in this branch's writeup even
though fixing the runner script's cap logic in general (so it doesn't
happen again for some future function) is better scoped as its own
`method/` issue than folded into this one, which is only about resolving
Ackley's verdict.

## Experimental design

**Unchanged from PR #18:** Ackley (bounds ±32.768, d=10), `L-BFGS-B` local
minimizer, `stepsize = 3.2` (the same domain-scaled value, unchanged --
this branch does not revisit whether that specific stepsize is the right
one, only whether it was tested with enough power), `LocalMinimizerRestart`
control and budget-matching procedure (`niter+1` local-minimizer calls per
arm, `niter=50` as before), margin convention (25% of control-arm pilot
spread, per `STEPSIZE_SPEC.md`).

**Changed:** run a fresh feasibility pilot on this exact configuration
(disjoint seeds from every prior block: PR #16's pilot 1000-1014 and real
sweep 0-59, PR #18's pilot 2000-2014 and real sweep 10000-10059), compute
`required_sample_size` from the fresh pilot's spread rather than reusing
PR #18's `135` figure verbatim, and run the real sweep at that recommended
`n` uncapped (no `MAX_SEEDS`-style ceiling) -- if the fresh pilot's estimate
differs meaningfully from PR #18's, use the fresh one and state why they
differ if it's non-trivial (pilot sampling variance is expected to move the
estimate somewhat).

**Only Ackley:** Rastrigin and Griewank already have determinate,
well-powered verdicts (`CONTRIBUTES` at power ~unreported-but-effect-9.5x-
margin, and `NULL` at power ~1.00, respectively, per PR #18) and do not
need re-running here -- re-running them anyway would not answer anything
new and would just be spending seeds without a question behind it.

## Metrics

Identical to PR #16/#18: best objective value (continuous, BCa bootstrap
TOST, lower-is-better), distance to the known global optimum (secondary),
degeneracy pre-check, selection ceiling (already measured as non-trivial in
prior runs for Ackley -- re-confirm at the new seed count rather than
assuming it carries over unchanged, since ceiling is itself seed-sampled).

## Baselines / controls

Same as PR #16/#18 -- budget-matched independent restarts of the identical
`L-BFGS-B` local minimizer, same `niter+1`-call budget, same stepsize
(3.2) for the treatment arm's internal minimizer configuration.

## Expected outcomes

- **`NULL`, well-powered** (hypothesis): completes the basinhopping line of
  research with all three functions determinately resolved (`CONTRIBUTES`
  / `NULL` / `NULL`) under a consistently domain-scaled stepsize.
- **`CONTRIBUTES` or `HARMFUL`, well-powered**: contradicts the directional
  hypothesis but is a complete, equally valuable resolution -- report
  exactly as found, and update the "revised reading of the basinhopping
  results" summary this project should eventually carry (README, once
  someone folds this line of work into it) accordingly.
- **Still `INCONCLUSIVE` even at the feasibility-recommended n**: report the
  achieved power and the observed spread; this would mean Ackley's true
  effect is smaller relative to its variance than the 60-seed pilot
  suggested, which is itself a fine (if anticlimactic) place to stop this
  particular thread, rather than a signal to keep escalating `n`
  indefinitely chasing a determinate answer.

## Interpretation plan

- Determinate verdict reached: this closes the basinhopping-audit line of
  research (issues #15, #17) with all three functions resolved; note this
  explicitly as a natural stopping point rather than continuing to look for
  more parameters to vary on the same three functions.
- Still inconclusive: report as the honest end state of this specific
  thread; do not escalate `n` further without a fresh, separate
  justification for why more seeds specifically (rather than a different
  design change) would resolve it, per `GIT_WORKFLOW.md`'s caution against
  re-running until a hoped-for verdict appears.
- Does not touch, retroactively edit, or overwrite PR #16's or PR #18's
  `SPEC.md`/`STEPSIZE_SPEC.md`, code, or results artifacts -- those
  pre-registered results stand as reported; this is a new, dated,
  standalone experiment building on them.
- Says nothing about the deferred ScaffoldSafety/GAIA cross-methodology
  comparison, `naslib`-blocked README item 3, or whether
  `plugins/basinhopping_audit`'s general seed-capping pattern needs fixing
  (that's the separate possible `method/` issue `STEPSIZE_SPEC.md` also
  named, out of scope here).

## Confounds considered

- **The fresh pilot's spread estimate could differ non-trivially from PR
  #18's**, since pilot sampling itself has variance -- if the fresh
  `required_n` comes out very different (e.g. much larger), use it anyway
  rather than reverting to PR #18's `135` for convenience, and state the
  discrepancy rather than silently picking whichever number is smaller.
- **Seed disjointness across three prior blocks** (PR #16's two, PR #18's
  two) must be verified by test, not just asserted, matching the precedent
  `STEPSIZE_SPEC.md` itself set (`test_plugin_basinhopping_stepsize_
  experiment.py`) -- a fifth overlapping seed range would silently
  reintroduce a version of the shared-seed-stream confound already flagged
  in issue #15.
- **This is not "keep re-running until INCONCLUSIVE goes away."** The
  higher `n` is justified by the pilot's own pre-registered power
  calculation having been computed and then ignored (a documented,
  specific defect), not by a general license to add seeds whenever a
  result is inconvenient. If this run also comes back `INCONCLUSIVE` at
  adequate power, that stands as the answer -- see Expected outcomes.
- **Stepsize itself is not revisited here** -- 3.2 stays fixed by
  construction; whether 3.2 (Rastrigin's ratio, transplanted) is really the
  best stepsize for Ackley specifically is a different, not-yet-asked
  question, out of scope for this branch.

---

## Results

Full artifacts:
`results/basinhopping_audit_ackley_power/{audit.json,audit.csv}`. Seeds
verified disjoint from every prior block (`tests/test_plugin_
basinhopping_ackley_power_experiment.py`): this branch's pilot (3000-3014)
and real sweep (20000+) share nothing with PR #16's pilot (1000-1014) /
real sweep (0-59) or PR #18's pilot (2000-2014) / real sweep (10000-10059).

**The fresh pilot's spread estimate differed substantially from PR #18's,
exactly the discrepancy `ACKLEY_POWER_SPEC.md`'s "Confounds considered"
anticipated as possible:** PR #18's 15-seed pilot (seeds 2000-2014)
measured a paired-difference spread of 0.366, implying `n=135`; this
branch's 15-seed pilot (seeds 3000-3014), on the *identical* configuration
(same function, stepsize, niter), measured a spread of **1.090** -- roughly
3x larger -- implying `n=1971`, not 135. Per the pre-registered rule, the
fresh estimate was used as-is, uncapped: the real sweep ran at **n=1971**,
not PR #18's number and not any hardcoded ceiling.

**Result: `HARMFUL`, decisively.**

| | value |
|---|---|
| observed difference (ctrl - treat, oriented) | -0.876 |
| 90% CI | [-0.917, -0.830] |
| margin | ±0.072 |
| n | 1971 |
| p-value | 5.76e-165 |
| power (TOST-for-NULL, see note below) | 0.708 |
| degeneracy | not degenerate (0/1971, mean distinct ratio 0.988) |
| selection ceiling | 0.0 |
| identical-representation rate | 0.0 (not vacuous) |

The effect (-0.876) is ~12x the margin (0.072), and the CI's nearer bound
(-0.830) still clears the margin by more than 11x. The `p-value` of
5.76e-165 leaves essentially no room for this being a sampling artifact.
The reported `power=0.708` is, as with PR #16's Rastrigin row, the
probability of establishing *equivalence* if the true effect were zero --
not the strength of this `HARMFUL` claim, which the CI position and
p-value already establish overwhelmingly. `n_for_target_power=2320` (only
~18% more than the 1971 actually run) confirms this was already close to
fully powered for the NULL question too, for what that number is worth
here.

## Interpretation

**Ackley's verdict is now determinate: `HARMFUL`, not `NULL` as this
branch's own directional hypothesis predicted.** SPEC.md explicitly
anticipated this as a live, equally valuable alternative outcome ("Null
(would contradict the hypothesis): at adequate power, the verdict is
`HARMFUL` or `CONTRIBUTES`"), and that is what happened -- the hypothesis
being wrong does not make this branch's result less complete or less
useful. PR #18's `INCONCLUSIVE` (60 seeds, underpowered) is now resolved:
it was not a coin-flip between `NULL`, `HARMFUL`, and `CONTRIBUTES` that
happened to land on the fence -- the true effect was a real, substantial,
harmful one that 60 seeds simply could not resolve from noise.

**This closes the basinhopping-audit line of research (issues #15, #17,
#19) with all three functions now determinate and well-evidenced:**
Rastrigin `CONTRIBUTES` (PR #16/#18, unchanged), Griewank `NULL` (PR #18,
power ~1.00), Ackley `HARMFUL` (this branch, p=5.76e-165). Read together
with issue #17's finding: the domain-scaled stepsize did genuinely resolve
the *reliability* question for Ackley (PR #16's `HARMFUL` under a
mismatched stepsize was not trustworthy evidence about the mechanism), but
it did **not** resolve to "no harm" -- it resolved to a *different*,
well-evidenced `HARMFUL` verdict under the corrected, domain-scaled
stepsize. Basin-hopping's step-taking mechanism, even scaled to Ackley's
domain width in the same proportion that produces a strong `CONTRIBUTES`
on Rastrigin, measurably underperforms blind restarts on Ackley at this
budget. Whatever is different about Ackley's landscape (its many, uniformly
small-amplitude local minima packed across a much larger domain, versus
Rastrigin's more localized bowl-and-ripple structure) apparently makes this
scaffold's specific mechanism a poor fit, not merely under-tuned.

**Also notable and worth flagging explicitly, per the issue's own framing:**
the pilot-based feasibility approach used throughout this line of work is
fragile for Ackley specifically -- two independent 15-seed pilots on the
literal same configuration produced spread estimates (0.366 vs 1.090) that
implied required sample sizes differing by ~15x (135 vs 1971). This is not
a bug in `required_sample_size` or in either pilot's execution; it reflects
genuine sampling variance in estimating a standard deviation from only 15
observations, made worse if the underlying paired-difference distribution
is heavy-tailed or occasionally produces an outlier run (plausible for a
stochastic global-search method on a landscape with many local minima).
This is exactly the generalizable observation `STEPSIZE_SPEC.md`/this
issue flagged as worth a possible future `method/` issue (a 15-seed pilot
may simply be too small to reliably estimate `n` for some designs) -- noted
here, not chased further, per this branch's own scope.

**What this does NOT establish:**

- Why Ackley specifically resists this scaffold's mechanism even under a
  correctly-scaled stepsize -- this branch resolves *that* it is `HARMFUL`
  at adequate power, not the underlying reason.
- Whether a different stepsize (not derived from Rastrigin's ratio),
  different `niter`, or different local minimizer would change Ackley's
  verdict -- all held fixed by design, per `STEPSIZE_SPEC.md`'s original
  scope, unrevisited here.
- Whether 15-seed pilots are generally too small for this project's
  feasibility-probe convention across other problems/domains -- one
  documented instance of high pilot-to-pilot variance on one function is
  not a general claim about the convention.
- Anything about ScaffoldSafety/GAIA or `naslib`-blocked README item 3.

## Decision

**MERGE.** Checked against `GIT_WORKFLOW.md`'s nine criteria:

- **Scientific relevance:** resolves the last open thread in the
  basinhopping-audit line of research (issues #15, #17), completing all
  three functions with determinate, well-evidenced verdicts.
- **Correctness:** seed disjointness from all four prior blocks verified
  by test, not just claimed; the uncapped seed-count logic
  (`MIN_SEEDS` as a floor, no ceiling) verified by test to have no
  `MAX_SEEDS`-style attribute.
- **Experimental validity:** fresh feasibility probe on this exact
  configuration used as-is, not reverted to PR #18's smaller estimate for
  convenience, per the pre-registered rule -- even though the resulting
  seed count (1971) was far larger than expected.
- **Reproducibility:** deterministic given fixed seeds; full config,
  including the fresh-vs-prior probe comparison, travels with the result.
- **Documentation:** this file; the module docstring states the exact
  defect being corrected (PR #18's `MAX_SEEDS` cap silently overriding its
  own feasibility probe) and why it doesn't apply here.
- **Interpretation:** stated above, including the explicit acknowledgment
  that the directional hypothesis was wrong and what the pilot-variance
  finding does and does not establish.
- **Research integrity:** the hypothesis (`NULL`) was not confirmed, and
  this is reported as the actual, more informative result rather than
  reframed or downplayed; the substantial pilot-to-pilot spread
  discrepancy is surfaced rather than smoothed over.
- **Integration:** additive; new script and results directory only, does
  not touch or overwrite PR #16's or PR #18's `SPEC.md`/`STEPSIZE_SPEC.md`,
  code, or results artifacts.
- **Evidence:** a real, very well-powered sweep (1971 seeds, p=5.76e-165)
  against real `scipy` code -- among the most decisive single-metric
  results this project has produced.

Follow-up for the researcher, not this branch: whether the pilot-spread
instability observed here (15 seeds insufficient to reliably estimate `n`
for this configuration) warrants a `method/` issue about
`plugins/basinhopping_audit`'s (or the project's general) feasibility-probe
convention; and whether investigating *why* Ackley resists basin-hopping's
mechanism (distinct from confirming *that* it does) is worth a further
`research/`-scale question.
