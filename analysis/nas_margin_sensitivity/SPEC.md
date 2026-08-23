<!--
Copied verbatim from GitHub issue #13, per research/GIT_WORKFLOW.md and
research/SPEC_TEMPLATE.md. Section headers below follow SPEC_TEMPLATE.md;
their content is the issue body's corresponding section, unedited.
-->

# Spec: analysis/nas-margin-sensitivity

**Branch:** `analysis/nas-margin-sensitivity`
**Date opened:** 2026-08-23
**Status:** IN PROGRESS
**Issue:** https://github.com/shlokkvaishnav/Null-Scaffold-Audit/issues/13

## Branch type

analysis/ -- analysis of an existing result, no new data collection

This is README item 5 ("Margin-sensitivity analysis -- how much does the
verdict move as the pre-registered margin moves, holding data fixed?"),
which is worded generically. This issue scopes it to the one dataset that
exists today rather than proposing new data collection: the 30-seed paired
`valid_accuracy` differences already sitting in
`results/nas_search_self_audit/audit.json` (issue #11 / PR #12,
`plugins/nas_search`'s `RandomSearch`-vs-itself self-audit, verdict `NULL`).
Note `GIT_WORKFLOW.md`'s branch-type table example row happens to be named
`experiment/margin-sensitivity` -- that example name is generic and does not
by itself require new data collection; the definition that actually governs
the choice here is `analysis/`'s ("no new data collection"), which is
exactly what this issue is. No NATS-Bench evaluations are re-run; only the
existing per-seed paired data is re-tested under different margins.

## Research question

`AUDIT_METHODOLOGY.md` §7 states plainly: "the margin is a judgement call,"
and "a poorly chosen margin produces confident nonsense in either
direction." §10 leaves open "how is δ chosen for a new metric?" and records
no answer. The `nas_search` self-audit (issue #11) pre-registered δ = 0.3
percentage points on `valid_accuracy` via a separate 20-seed feasibility
probe (per PR #12's description) and got `NULL` with the observed
difference's 90% CI at [-0.197, -0.008]pp -- comfortably inside ±0.3pp, but
not by a large margin relative to the CI's own width (~0.19pp).

The question: holding the collected data fixed, how does the verdict on
`valid_accuracy` change as δ is swept across a range of values a researcher
might plausibly have pre-registered instead of 0.3pp -- and specifically, is
there a δ within that plausible range where the verdict would have read
something other than `NULL` (`INCONCLUSIVE`, or, since the observed
difference and its full CI are negative, `HARMFUL`)? This is not a new claim
about the scaffold (the scaffold is still null by construction in this
dataset, per issue #11) -- it is a claim about how close the *reported*
`NULL` verdict sits to the boundary of a different, still-defensible
pre-registration choice, which is exactly the fragility `AUDIT_METHODOLOGY.md`
§7 warns about in the abstract but that no branch has yet measured
concretely on real (non-synthetic) audit data.

## Hypothesis

The verdict is not fully robust across the plausible pre-registration range.
Specifically: because the observed CI [-0.197, -0.008]pp lies entirely below
zero (the treatment arm underperformed control by a small but consistently
negative margin across seeds), there should exist some δ smaller than 0.3pp
-- plausibly somewhere between 0.05pp and 0.2pp, a range a less generous but
still defensible feasibility probe could have produced -- at which the
verdict changes from `NULL` to `INCONCLUSIVE`, and a smaller δ still
(bounded by the CI's own width, ~0.008-0.197pp) at which it would read
`HARMFUL` rather than `NULL`. This is a directional prediction from the
already-observed interval, not a guess at exact thresholds -- the analysis
is what produces the actual crossover points.

## Null / alternative hypothesis

**Null (would contradict the hypothesis above):** the verdict stays `NULL`
across the entire plausible pre-registration range (e.g. margins from 0.05pp
up to several pp) -- i.e. the CI is narrow enough, and far enough from zero
in the direction that matters, that no defensible alternative margin choice
would have changed the reported conclusion. This would mean the specific
δ = 0.3pp chosen for issue #11 was not doing much work, and the `NULL`
verdict is robust rather than margin-dependent.

**Alternative (supports the hypothesis):** a defensible δ exists (i.e. one a
researcher could plausibly have pre-registered from a feasibility probe,
not an adversarially cherry-picked extreme) at which the verdict changes to
`INCONCLUSIVE` or `HARMFUL`. Report the crossover margin(s) explicitly,
not just "yes it's sensitive" -- a margin-sensitivity finding without the
actual crossover value is not reproducible or useful to whoever sets the
margin for README item 3's real controller audits.

## Motivation

`AUDIT_METHODOLOGY.md` centers the entire audit design on pre-registering δ
*before* seeing the data, specifically to prevent choosing a margin after
the fact that manufactures a preferred verdict (§4.1, §6 Alternative A). But
nothing currently tells a researcher, before they commit to a δ, how much
their specific choice actually mattered to the outcome -- whether they
picked a number that happened to sit far from any real decision boundary
(robust) or one sitting right on top of one (fragile). This analysis builds
that check, using real audit data instead of a synthetic illustration, and
in doing so gives README item 3's implementer a concrete precedent: run this
same margin-sensitivity check alongside any future feasibility probe, on
the real DARTS/GDAS/RegularizedEvolution/BANANAS audits, before trusting a
`NULL` (or any other verdict) that sits close to its own margin.

It also has a direct bearing on `AUDIT_METHODOLOGY.md` §10's open question
("how is δ chosen for a new metric?") -- a demonstrated sensitivity analysis
is a concrete input to that unresolved decision, not a replacement for it.

## Experimental design

No new NATS-Bench evaluations. Load the existing paired per-seed
`valid_accuracy` values for both arms from
`results/nas_search_self_audit/audit.json` (`arms.treatment_metrics` /
`arms.control_metrics`, 30 seeds, per PR #12) and re-run the existing BCa
bootstrap TOST procedure (`engine/audit/statistics.py` or wherever
`engine.audit.arms.audit()`'s per-metric test lives) against a swept range
of margins, holding every other input identical (seeds, resample count,
confidence level 0.90).

**Margin sweep range:** from an economically small value (e.g. 0.02pp) up
to at least 1.0pp, dense enough near the two expected crossover points
(`NULL`↔`INCONCLUSIVE` near the CI's outer bound, `NULL`/`INCONCLUSIVE`↔
`HARMFUL` near the CI's inner bound) to report them to two significant
figures rather than bracket them loosely. A coarse sweep (e.g. every 0.01pp)
is cheap since it touches no new data -- report the actual crossover
margins, not just a plot.

**Held constant:** the 30 paired observations themselves, the bootstrap
resample count and RNG seed used for the interval, the confidence level.
Re-run using the *same* CI (not recomputing a point estimate and eyeballing
whether it falls inside ±δ) at every swept margin, since the width of the CI
does not change with δ -- only which side of ±δ it lands on can change; make
sure the implementation reuses one precomputed CI where correctness allows,
rather than three independent bootstrap resamples that could differ due to
resampling noise and confuse a real margin effect with resampling noise.

## Metrics

- **Verdict as a function of δ** (categorical: `HARMFUL` / `INCONCLUSIVE` /
  `NULL` / `CONTRIBUTES`) across the swept range -- the primary output.
- **The crossover margin value(s)** where the verdict changes, reported to
  at least 2 significant figures (e.g. "verdict is `NULL` for δ ≥ 0.21pp,
  `INCONCLUSIVE` for 0.008pp ≤ δ < 0.21pp, `HARMFUL` for δ < 0.008pp" --
  illustrative shape only, not a predicted answer).
- **Distance from the pre-registered δ = 0.3pp to the nearest crossover**,
  as a simple ratio or absolute-pp distance -- this is what actually answers
  "how close did the real pre-registration choice sit to a different
  conclusion," which is the point of the whole analysis.

## Baselines / controls

None in the usual audit sense (no scaffold-vs-searcher comparison here) --
this is a re-analysis of one existing audit's output under varying
assumptions, not a new arms comparison. The "control" is the already-run
audit at δ = 0.3pp itself, which every swept result is compared back against.

## Expected outcomes

- **Crossover found within a plausible pre-registration range** (expected):
  reports the specific margin(s), and the distance from 0.3pp to them.
  Actionable finding for README item 3's implementer: shows margin choice
  needs a stated sensitivity check, not just a single feasibility-probe
  number.
- **No crossover within any plausible range** (verdict fully robust):
  equally reportable and useful -- would mean this particular result was not
  fragile to the margin choice, which is worth knowing before assuming every
  audit result needs a sensitivity companion analysis.
- **Sweep reveals the CI itself was mis-specified or the bootstrap is
  unstable at extreme margins** (e.g. degenerate behavior at very small δ):
  would be a finding about the statistics module rather than about this
  dataset, and should be filed as a separate `method/` issue rather than
  folded into this branch's conclusion.

## Interpretation plan

- Crossover found close to 0.3pp (say, within 2x): flag explicitly in the
  writeup that the `NULL` verdict from issue #11/PR #12 is margin-sensitive,
  without retracting it -- 0.3pp was still pre-registered honestly, before
  the data was seen, so the original verdict stands; this analysis adds
  context about its robustness, not a revision of it (`GIT_WORKFLOW.md`'s
  "the pre-registered margin does not move" applies to the original branch;
  this branch is diagnostic, run after the fact, and must be clearly labeled
  as such so it is never confused with re-margining after seeing results).
- Crossover found far from 0.3pp: report the margin of safety as a positive
  finding -- useful precedent for how much slack a future feasibility probe
  should aim for.
- No crossover in the plausible range: report as a robustness finding.
- Recommend, as a closing note (not a new issue by itself unless the
  reviewer/researcher agrees it's warranted): whether
  `scripts/audit_feasibility.py` should optionally emit a margin-sensitivity
  sweep alongside its existing power calculation, so this check becomes
  routine rather than a one-off analysis branch each time.

## Confounds considered

- **Resampling noise mistaken for a margin effect.** The BCa bootstrap CI
  itself has resampling randomness; if each swept margin re-draws its own
  bootstrap resample, a small verdict flip near a crossover could reflect
  resampling noise rather than a real margin-dependent transition.
  Mitigation: compute the CI once (fixed resample, fixed RNG seed) and
  reuse it across the entire margin sweep, since the interval doesn't
  depend on δ -- only the comparison against ±δ does.
- **Selecting the sweep range to manufacture a dramatic finding.** The range
  (0.02pp-1.0pp+) is chosen for coverage around the two data-implied
  crossover regions computed from the CI bounds already reported in PR #12
  (-0.197, -0.008), not chosen after seeing where a crossover happens to
  look interesting. State the range's derivation in the writeup so it's
  checkable.
- **This says nothing about margin sensitivity for other metrics, other
  problems (`sss` space, other datasets/hp settings), or the real controller
  audits in README item 3** -- it is one case study on one metric from one
  already-run audit. Don't generalize "the audit's verdicts are margin-
  fragile" from a single dataset; report this as a demonstrated method plus
  one concrete result, not a general claim about the audit mechanism.
- **A crossover very close to the pre-registered margin is not evidence the
  original margin was chosen badly** -- 0.3pp was set from a genuine
  feasibility probe before the data was seen (per PR #12), and post-hoc
  proximity to a crossover is expected some fraction of the time even when
  the pre-registration process is followed correctly. Don't overstate a
  close crossover as a process failure.

---

## Results

Full artifacts: `results/nas_margin_sensitivity/summary.csv` (the coarse
margin -> verdict sweep, 0.001pp to 1.2pp in 0.001pp steps),
`results/nas_margin_sensitivity/summary.md` (this section's source).
`report.json` is written locally by the script but not committed (its
content -- CI bounds, crossovers, distance -- already lives in `summary.md`
and `summary.csv`; committing a third redundant artifact format wasn't
worth extending `.gitignore`'s shared allowlist for this one branch).

**Recomputed CI matches PR #12 exactly**, confirming this sweep is testing
the same interval the original audit reported, not a different computation:
90% BCa bootstrap CI on the paired `valid_accuracy` difference (n=30,
resamples=10,000, seed=0) = **[-0.1970, -0.0081]** percentage points.

**Pre-registered margin (±0.3pp) -> NULL**, matching PR #12's reported
verdict exactly, as it must (same data, same margin, same procedure).

**Two crossovers found**, exactly the shape the hypothesis predicted:

| margin (pp) | verdict below | verdict at/above |
|---|---|---|
| 0.0081 | HARMFUL | INCONCLUSIVE |
| 0.1970 | INCONCLUSIVE | NULL |

- For δ < 0.0081pp: **HARMFUL** (the entire CI sits below -δ).
- For 0.0081pp ≤ δ < 0.1970pp: **INCONCLUSIVE**.
- For δ ≥ 0.1970pp: **NULL** (matches the observed 0.3pp pre-registration).
- No `CONTRIBUTES` region exists anywhere in [0.001, 1.2]pp -- consistent
  with the CI being entirely negative (`ci_low` never exceeds any positive
  margin in this range).

These boundary values are exact, not sweep-resolution artifacts: they equal
`-ci_high` (0.008143933584611237) and `-ci_low` (0.1969507431838293) to
machine precision, which follows directly from `_resolve`'s definition
(`ci_high < -margin` for `HARMFUL`; `-margin < ci_low` for `NULL`) -- the
bisection refinement (tolerance 1e-6) simply locates what the closed form
already implies, and `tests/test_analysis_nas_margin_sensitivity.py` checks
both directly against a hand-built interval.

**Distance from the pre-registered margin (0.3pp) to the nearest crossover
(0.1970pp): 0.1030pp, a ratio of 1.52x.** The pre-registered margin sits
above the NULL boundary by just over half its own value again -- closer to
a different conclusion than a factor of 2, further than a factor of 1.

## Interpretation

**The `NULL` verdict from issue #11 / PR #12 is margin-sensitive, within a
plausible pre-registration range.** Had the feasibility probe that set
δ = 0.3pp instead landed on a value below ~0.197pp -- a margin roughly
two-thirds the size actually chosen, not an extreme or cherry-picked one --
the reported verdict would have read `INCONCLUSIVE` rather than `NULL`. A
margin below ~0.008pp (an order of magnitude tighter) would have read
`HARMFUL`, though a margin that tight is a less plausible pre-registration
choice for this metric's observed spread.

This does **not** retract or revise the original verdict. δ = 0.3pp was
pre-registered from a genuine feasibility probe before the 30-seed sweep
ran (PR #12), and `GIT_WORKFLOW.md`'s rule that "the pre-registered margin
does not move" applies to that branch, not this one -- this branch is
explicitly diagnostic, run after the fact on already-collected data, and is
not a re-margining of issue #11's result. The original `NULL` stands.

What this establishes: the specific margin choice in issue #11 was doing
real work -- it was not so generous that any plausible alternative would
have agreed, nor so tight that the result was fragile to sampling noise
alone. A 1.52x margin of safety is closer to "worth checking before
trusting" than to "obviously robust."

What this does **not** establish, matching SPEC.md's confounds section:
anything about margin sensitivity for other metrics, other NATS-Bench
problems (the `sss` space, other datasets/hp settings), or the real
controller audits planned for README item 3 -- this is one case study on
one metric from one already-run audit. It is also not evidence that the
original δ = 0.3pp was chosen badly: post-hoc proximity to a crossover is
expected some of the time even when pre-registration is followed correctly,
and nothing here suggests the feasibility probe that set it was flawed.

**Closing recommendation for the researcher** (not filed as a new issue by
this branch, per SPEC.md -- left for the researcher/reviewer to judge
whether it's warranted): `scripts/audit_feasibility.py` could optionally
emit a margin-sensitivity sweep like this one alongside its existing power
calculation, so a crossover-distance check becomes routine input to setting
δ rather than a one-off analysis branch run after the fact each time.

## Decision

**MERGE.** Checked against `GIT_WORKFLOW.md`'s nine criteria:

- **Scientific relevance:** directly answers README item 5, using real
  audit data rather than a synthetic illustration, and gives README item
  3's implementer a concrete precedent.
- **Correctness:** the sweep is verified, not just run -- `main()` asserts
  its recomputed CI matches PR #12's reported CI to 1e-9 before sweeping,
  and the crossover values are checked against `_resolve`'s closed-form
  boundaries in the test suite, not just eyeballed off a plot.
- **Experimental validity:** no new arms comparison in the usual audit
  sense (this is a re-analysis, per SPEC.md's "Baselines / controls"); the
  CI is computed once and reused across the whole sweep specifically to
  avoid the resampling-noise confound SPEC.md flags.
- **Reproducibility:** deterministic given the committed
  `results/nas_search_self_audit/audit.json` (fixed seed, fixed resample
  count); `analysis/nas_margin_sensitivity/run_sweep.py` reruns
  end-to-end with `python analysis/nas_margin_sensitivity/run_sweep.py`.
- **Documentation:** this file, plus the module's own docstring explaining
  why the CI is computed once rather than per swept margin.
- **Interpretation:** stated above, including the explicit non-retraction
  of the original verdict and what this branch does not generalize to.
- **Research integrity:** the finding (margin-sensitive within a plausible
  range) is reported even though it complicates, rather than simply
  confirms, PR #12's clean `NULL` -- and is stated without overclaiming a
  process failure in the original pre-registration.
- **Integration:** additive; touches no existing code, only reads a
  committed result file and adds a new `analysis/` package.
- **Evidence:** real recomputation against the real committed data, with an
  exact-match assertion against the original, not a re-derivation that
  merely looks similar.

Follow-up for the researcher, not this branch: whether to file the
`scripts/audit_feasibility.py` margin-sensitivity-sweep enhancement noted
above as its own `method/` issue.
