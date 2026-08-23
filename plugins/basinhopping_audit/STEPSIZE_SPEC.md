<!--
Copied verbatim from GitHub issue #17, per research/GIT_WORKFLOW.md and
research/SPEC_TEMPLATE.md. Section headers below follow SPEC_TEMPLATE.md;
their content is the issue body's corresponding section, unedited.

Named STEPSIZE_SPEC.md, not SPEC.md, deliberately: this experiment shares
plugins/basinhopping_audit/ with issue #15 / PR #16, whose own SPEC.md
already lives at the root of this directory and must not be overwritten --
GIT_WORKFLOW.md's "a completed branch's pre-registered result stands"
applies to that file as much as to its results artifacts.
-->

# Spec: experiment/basinhopping-stepsize-scaling

**Branch:** `experiment/basinhopping-stepsize-scaling`
**Date opened:** 2026-08-23
**Status:** IN PROGRESS
**Issue:** https://github.com/shlokkvaishnav/Null-Scaffold-Audit/issues/17

## Branch type

experiment/ -- one specific, scoped experiment

This is a single-parameter re-run of the design already implemented in
`plugins/basinhopping_audit/` (issue #15 / PR #16), not a new plugin or
investigation. `plugins/basinhopping_audit/SPEC.md`'s own closing section
names this exact follow-up: "whether an Ackley re-run with a domain-scaled
`stepsize` is worth a separate `experiment/` issue."

## Research question

PR #16 held `basinhopping`'s `stepsize` fixed at scipy's documented default
(0.5) across all three test functions, and flagged -- without confirming --
that this could confound the mixed result, since the three functions' bound
widths differ enormously (Rastrigin ±5.12, Ackley ±32.768, Griewank ±600 --
a ~117x range) while `stepsize` did not scale with them.

Computing the ratio of the fixed `stepsize` to each function's domain width
makes the pattern sharper than PR #16's writeup stated: 0.5 is ~4.9% of
Rastrigin's width, ~0.76% of Ackley's, and only ~0.042% of Griewank's -- a
monotonically shrinking ratio that lines up exactly with the observed
verdict ordering (`CONTRIBUTES` on Rastrigin -> `HARMFUL` on Ackley ->
`INCONCLUSIVE`/`DEGENERATE` on Griewank). That ordering is consistent with
two different explanations, and PR #16 could not distinguish them:

1. The three functions genuinely differ in how much basin-hopping's
   step-taking mechanism helps at this budget, independent of stepsize.
2. A single confound -- `stepsize` becoming vanishingly small relative to
   the search domain as domain width grows -- is what's actually driving
   the pattern, and a properly domain-scaled `stepsize` would flatten or
   change the ordering.

**Question:** holding everything else in the existing design fixed (same
three functions, dimension, local minimizer, budget-matching procedure,
statistical procedure) and varying only `stepsize` -- set proportionally to
each function's domain width instead of one shared absolute value -- does
the verdict pattern from PR #16 change? Specifically: does Ackley's
`HARMFUL` verdict change (toward `NULL` or `CONTRIBUTES`), does Griewank's
`DEGENERATE` flag clear, and does Rastrigin's `CONTRIBUTES` persist,
strengthen, or weaken?

## Hypothesis

A `stepsize` set to the same *fraction* of domain width across all three
functions (using Rastrigin's original effective ratio, ~4.9% of width, as
the shared proportional standard, since that is the one configuration PR #16
already showed produces a working, non-degenerate, `CONTRIBUTES` case) will:

- Move Ackley's verdict away from `HARMFUL` -- plausibly to `NULL` or
  `CONTRIBUTES` -- since its previous effective stepsize (~0.76% of width)
  was over 6x smaller than the proportional standard.
- Clear Griewank's `DEGENERATE` flag and produce a non-degenerate verdict,
  since its previous effective stepsize (~0.042% of width) was more than
  100x smaller than the proportional standard, plausibly too small to
  escape the basin the local minimizer already converges to on its own.
- Leave Rastrigin's `CONTRIBUTES` intact, since its stepsize is
  (approximately) unchanged by this rescaling.

This is a mechanism hypothesis, not a foregone conclusion -- it is entirely
possible the per-function differences are real and a domain-scaled stepsize
changes nothing, which is itself the informative negative result this
experiment is designed to distinguish from the confound story above.

## Null / alternative hypothesis

**Null (would contradict the hypothesis):** the verdict pattern is
unchanged, or does not change in the predicted direction, under a
domain-scaled stepsize -- Ackley stays `HARMFUL` or gets worse, Griewank
stays `DEGENERATE`/`INCONCLUSIVE`. This would mean the stepsize-ratio
confound proposed above does not explain PR #16's pattern, and the
per-function verdict differences are more likely a genuine property of
each function's landscape (or some other unexamined factor) rather than an
artifact of a fixed absolute stepsize.

**Alternative (supports the hypothesis):** the verdicts shift in the
predicted direction on at least Ackley and Griewank, which would mean PR
#16's specific verdicts (not the general fact that mixed results occurred,
but the specific `HARMFUL`/`DEGENERATE` labels on those two functions)
were substantially an artifact of the stepsize/domain-scale mismatch rather
than evidence about basin-hopping's mechanism on those landscapes per se.

## Motivation

This bears directly on how PR #16's result should be read going forward,
including by whoever eventually attempts README item 4's deferred
cross-methodology-comparison half (ScaffoldSafety/GAIA): a result explained
by an un-scaled implementation parameter is a different kind of finding
than a result reflecting the scaffold's actual mechanism, and conflating
them would misrepresent what this audit has actually shown. It also matters
for any future `experiment/`-scale audit run against a new function or
domain-search library using this same plugin's pattern (fixed absolute
hyperparameters translated across widely different problem scales) --
this experiment settles whether that's a mistake worth avoiding by default
in future work, not just a one-off curiosity about these three specific
functions.

## Experimental design

**Unchanged from PR #16:** the three functions (Rastrigin, Ackley,
Griewank) and their bounds, dimension (d=10), local minimizer (`L-BFGS-B`),
the `LocalMinimizerRestart` control and budget-matching procedure
(`niter+1` local-minimizer calls per arm), the statistical procedure
(per-function feasibility probe -> pre-registered margin and seed count ->
independent real-sweep seed block), and the degeneracy pre-check /
selection-ceiling steps.

**Changed:** `stepsize`, set per function as (domain width) x (Rastrigin's
original effective ratio, `0.5 / 10.24 ≈ 0.04883`) instead of the shared
absolute value 0.5. Concretely: Rastrigin ≈0.5 (unchanged, by construction),
Ackley ≈3.20, Griewank ≈58.59. State this derivation explicitly in
`SPEC.md` before running anything, so the specific ratio used is fixed by a
pre-registered rule (Rastrigin's own PR #16 configuration) rather than
picked after seeing which value happens to flip Ackley's verdict.

**Feasibility and margins:** re-run the feasibility probe fresh for each
function under the new `stepsize` (do not reuse PR #16's margins/seed
counts) -- a different `stepsize` changes both arms' variance, and reusing
old power calculations here would repeat exactly the mistake issue #13's
margin-sensitivity analysis exists to guard against (a margin chosen for
one configuration silently applied to a different one).

**What is explicitly not varied:** function set, dimension, local
minimizer, or the shared-ratio rule itself -- if the shared-ratio choice
looks wrong after seeing results (e.g. a different function's ratio would
have been a more defensible "standard" to scale from), that is a new,
separate follow-up question, not a retroactive change to this branch's
design.

## Metrics

Identical to PR #16, applied per function: best objective value (continuous,
BCa bootstrap TOST, lower-is-better), distance to the known global optimum
(continuous, secondary/interpretive), degeneracy pre-check result, and
selection ceiling.

## Baselines / controls

Same as PR #16 -- budget-matched independent restarts of the identical
`L-BFGS-B` local minimizer basin-hopping calls internally, at the same
`niter+1`-call budget. No new baseline; the comparison of interest here is
this branch's verdicts against PR #16's already-published verdicts on the
same functions, not a new control arm.

## Expected outcomes

- **Verdicts shift as hypothesized on Ackley and/or Griewank**: supports the
  stepsize/domain-scale confound explanation; report explicitly that PR
  #16's `HARMFUL`/`DEGENERATE` labels should be read as partly an artifact
  of an unscaled parameter, not purely a property of those landscapes.
- **Verdicts unchanged**: the confound hypothesis is refuted for this
  design; the per-function differences in PR #16 stand as a more likely
  genuine finding, and this branch's negative result is exactly as
  reportable as a positive one.
- **Partial shift** (e.g. Ackley improves but Griewank stays degenerate, or
  vice versa): report per function; do not average into one headline
  conclusion, per `AUDIT_METHODOLOGY.md` §4.1's "nullity is indexed by
  problem."
- **Rastrigin's result changes materially despite an (approximately)
  unchanged stepsize**: would indicate seed-block or implementation
  differences between the two runs beyond the intended one-parameter
  change, and should be investigated as an implementation discrepancy
  before trusting either run's comparison.

## Interpretation plan

- Confound confirmed (Ackley/Griewank shift as predicted): update the
  reading of PR #16's result in any future summary of this project's
  findings to note the stepsize dependency; consider, as a possible later
  `method/` issue (not this branch's job to decide), whether
  `plugins/basinhopping_audit` should default to a domain-scaled stepsize
  rather than requiring each caller to reason about it.
- Confound refuted: PR #16's mixed pattern stands as reported: real,
  function-dependent differences in whether this scaffold clears a
  budget-matched bar, not an artifact.
- Either way: this branch does not retroactively change PR #16's own
  verdicts or `results/basinhopping_audit/`'s artifacts -- per
  `GIT_WORKFLOW.md`, a completed branch's pre-registered result stands;
  this is a new, separate, clearly-dated experiment building on it.
- This says nothing about the deferred ScaffoldSafety/GAIA
  cross-methodology comparison, and nothing about `naslib`-blocked README
  item 3 -- both remain out of scope here.

## Confounds considered

- **Choice of which function's ratio to use as "the" proportional
  standard.** Rastrigin's is used because it's the one configuration
  already shown to work (non-degenerate, clear verdict) in PR #16 -- not
  because it's provably the "correct" stepsize-to-width ratio in general.
  State this explicitly; a different choice of standard is a legitimate
  alternative design this branch does not claim to have ruled out.
- **Confusing a stepsize effect with a seed-block effect.** PR #16 and this
  branch necessarily draw different random seed blocks for their real
  sweeps (fresh feasibility probes -> fresh margins -> fresh seed counts).
  A verdict change could in principle reflect ordinary sampling variation
  rather than the stepsize change. Mitigation: the feasibility-probe-driven
  margins and achieved power reported per function make this checkable --
  if power was adequate in both runs and the verdict still differs, sampling
  variation alone is an unlikely sole explanation, but this should be
  stated rather than assumed.
- **This experiment cannot fully separate "stepsize was the whole story" from
  "stepsize was part of the story."** A partial shift (see Expected
  outcomes) is a real possible outcome and must be reported as partial, not
  rounded to a clean confirm/refute.
- **Re-deriving `stepsize` is not the same as tuning it to get a hoped-for
  result** -- the derivation rule (scale by Rastrigin's original ratio) is
  fixed in `SPEC.md` before the sweep runs, specifically so this doesn't
  become "try stepsizes until Ackley reads NULL," which `GIT_WORKFLOW.md`'s
  "when not to merge" list would treat as cherry-picking.

---

## Results

Full artifacts:
`results/basinhopping_audit_stepsize_scaling/{audit.json,audit.csv}`. Ratio
derived from Rastrigin's PR #16 configuration: `0.5 / 10.24 = 0.048828125`.
Derived stepsizes matched the issue's stated values exactly: rastrigin 0.5,
ackley 3.2, griewank 58.59375. All seeds (pilot 2000-2014, real sweep
10000+) confirmed disjoint from PR #16's blocks (pilot 1000-1014, real
sweep 0-59) both by construction and by
`tests/test_plugin_basinhopping_stepsize_experiment.py`.

| function | stepsize | PR #16 verdict | this experiment's verdict | diff | 90% CI | margin | degenerate |
|---|---|---|---|---|---|---|---|
| rastrigin | 0.5 (unchanged) | CONTRIBUTES | **CONTRIBUTES** | +17.54 | [+15.84, +19.22] | ±1.84 | No |
| ackley | 3.2 | HARMFUL | **INCONCLUSIVE** | -0.52 | [-0.78, +0.14] | ±0.093 | No |
| griewank | 58.59 | INCONCLUSIVE, DEGENERATE | **NULL** | -1.1e-10 | [-2.0e-10, -4.6e-11] | ±1e-9 | **No** (was Yes) |

**Rastrigin: `CONTRIBUTES` persisted**, as predicted -- effect size
essentially unchanged (+17.54 vs. PR #16's +17.73), still ~9.5x the
(re-derived, tighter) margin.

**Ackley: shifted away from `HARMFUL`**, exactly the predicted direction,
though not all the way to `NULL` or a clean `CONTRIBUTES` -- it landed at
`INCONCLUSIVE`, with a CI ([-0.78, +0.14]) that now straddles zero rather
than sitting entirely below it. This is a **partial** shift: the harm
signal disappeared, but no positive contribution was established either.
Power for this row is 0.00 for the same reason as PR #16's rastrigin row --
this is TOST-for-NULL power, and it isn't the number that describes an
`INCONCLUSIVE` verdict's own evidence; what matters is that the CI now
spans zero.

**Griewank: `DEGENERATE` flag cleared**, exactly as predicted -- 0/20 runs
degenerate (mean distinct ratio 0.82, vs. PR #16's 20/20 degenerate, ratio
0.02) -- and the verdict moved from `INCONCLUSIVE` to a clean `NULL`, with
very high power (~1.00) at a margin still floored at `1e-9` (control spread
remained near machine-epsilon: `1.356e-10`). The larger stepsize (58.59, vs
0.5 before) let hops actually leave the immediate neighborhood of the
optimum on some iterations (hence non-degenerate proposals) while both arms
still converge to the same global optimum on this near-unimodal-at-d=10
landscape (hence `NULL`, not `CONTRIBUTES` -- there was nothing to gain
even once the mechanism was no longer artificially stuck).

## Interpretation

**The stepsize/domain-scale confound is confirmed, but only partially --
and differently on each function**, which is itself the finding: this
supports the "single confound" explanation over "three functions
genuinely differ," but not in an all-or-nothing way that would let PR
#16's original labels be dismissed as pure artifact:

- Rastrigin's result was never in question (stepsize essentially
  unchanged); its persistence is the negative control confirming this
  experiment measured a stepsize effect and not, say, a general
  seed-block or implementation drift between the two runs (STEPSIZE_SPEC.md's
  own confound check -- see "What confounds remain").
- Griewank's confound story is now well supported: the `DEGENERATE` label
  in PR #16 was a genuine artifact of a too-small stepsize relative to that
  function's ×117 wider domain, not a property of the landscape's
  multimodality. The corrected verdict (`NULL`, well-powered, non-
  degenerate) is arguably the more trustworthy read of Griewank at this
  dimension -- consistent with the mechanism proposed in PR #16's own
  writeup (a near-unimodal effective landscape at d=10), now demonstrated
  under a stepsize that actually lets the scaffold explore rather than
  sit still.
- Ackley's confound story is **only partially supported**. `HARMFUL`
  disappeared, which is consistent with the stepsize explanation -- but the
  result did not resolve to `NULL` or `CONTRIBUTES` either, so this
  experiment cannot say whether a correctly-scaled `basinhopping` helps,
  hurts, or does nothing on Ackley at this budget; it can only say that PR
  #16's specific `HARMFUL` label was not a reliable read of that question,
  because it was reached under a mismatched stepsize.

**Revised reading of PR #16's result, per STEPSIZE_SPEC.md's own
instruction (this branch does not retroactively edit that PR's verdicts or
artifacts):** PR #16's `HARMFUL` on Ackley and `DEGENERATE` on Griewank
should now be read as *substantially confounded by an unscaled stepsize*,
not as reliable evidence about basin-hopping's mechanism on those two
landscapes specifically. PR #16's `CONTRIBUTES` on Rastrigin stands
uncontested -- this experiment's own Rastrigin row is further evidence for
it, not against it.

**What this does NOT establish:**

- Whether `basinhopping` `CONTRIBUTES`, is `NULL`, or is `HARMFUL` on
  Ackley under a properly-scaled stepsize -- the `INCONCLUSIVE` result
  here answers "was PR #16's `HARMFUL` reliable" (no) but not "what is the
  correct verdict" (unresolved; would need a higher-power re-run, a
  separate follow-up).
- That Rastrigin's original stepsize ratio (4.9% of width) is *the*
  correct proportional standard in general -- it was chosen because it is
  the one configuration already shown to work, per STEPSIZE_SPEC.md's own
  "Confounds considered," not validated as optimal.
- Anything about the deferred ScaffoldSafety/GAIA cross-methodology
  comparison, or `naslib`-blocked README item 3.

## Decision

**MERGE.** Checked against `GIT_WORKFLOW.md`'s nine criteria:

- **Scientific relevance:** directly answers the follow-up PR #16's own
  SPEC.md named, and materially changes how that PR's mixed result should
  be read going forward.
- **Correctness:** the derivation rule reproduces the issue's stated
  values exactly (`tests/test_plugin_basinhopping_stepsize_experiment.py`);
  seed disjointness from PR #16's two seed blocks is verified by test, not
  merely asserted in a docstring.
- **Experimental validity:** fresh feasibility probe per function under
  the new stepsize (not reusing PR #16's margins, which STEPSIZE_SPEC.md
  explicitly required); only `stepsize` varied, everything else held fixed.
- **Reproducibility:** deterministic given fixed seeds; full config
  (including the derivation ratio) travels with every result row.
- **Documentation:** this file; the module docstring states the exact
  derivation formula and its provenance (Rastrigin's own PR #16 value).
- **Interpretation:** stated above per function, explicitly as a *partial*
  confirmation -- not rounded up to "confound fully explains everything"
  when Ackley's own result does not support that strong a claim.
- **Research integrity:** the mixed, only-partial result is reported as
  such; Ackley's unresolved verdict is not glossed over or quietly folded
  into "confirmed."
- **Integration:** additive; new script and results directory, does not
  touch or overwrite PR #16's `SPEC.md`, code, or `results/
  basinhopping_audit/` artifacts.
- **Evidence:** a real second sweep (140 seeds across three functions)
  against real `scipy` code, on seeds independently verified disjoint from
  the first.

Follow-up for the researcher, not this branch: whether a higher-power
Ackley re-run (larger `n`, informed by this experiment's own achieved
power) is worth its own `experiment/` issue to resolve the still-open
`INCONCLUSIVE`; and whether `plugins/basinhopping_audit` should default new
functions to a domain-scaled `stepsize` given this evidence, as a possible
`method/` issue.
