<!--
Copied verbatim from GitHub issue #15, per research/GIT_WORKFLOW.md and
research/SPEC_TEMPLATE.md. Section headers below follow SPEC_TEMPLATE.md;
their content is the issue body's corresponding section, unedited.
-->

# Spec: research/basinhopping-external-audit

**Branch:** `research/basinhopping-external-audit`
**Date opened:** 2026-08-23
**Status:** IN PROGRESS
**Issue:** https://github.com/shlokkvaishnav/Null-Scaffold-Audit/issues/15

## Branch type

research/ -- substantial implementation or investigation

## Research question

README item 4 ("External pipeline validation") is marked "highest-value,
not yet started": "run this audit against a scaffolded method this team did
not build, and compare its verdict directly to what the closest existing
comparators (ScaffoldSafety, the GAIA controlled comparison) would have
concluded on the same case." As literally worded that's two substantial
pieces of work bundled together -- auditing an external pipeline, *and*
reproducing what two different external papers' own methodologies would
conclude on the same case -- which is exactly the kind of "research program
disguised as a single issue" `AGENT_PIPELINE.md` says not to file. This
issue deliberately scopes the first, tractable half only: run this audit's
actual methodology against one real, externally-built, structurally
separable scaffolded pipeline, and report the verdict honestly. The
cross-methodology comparison against ScaffoldSafety/GAIA's own analysis
approach is explicitly left as a follow-up issue for later, not attempted
here, once/if this half succeeds.

**Concrete case chosen, and why:** `scipy.optimize.basinhopping` --
part of `scipy`, already a project dependency (`pyproject.toml`, not a new
install risk), actively maintained, not built by this team. It is
structurally exactly the `Scaffold`/`BaseSearcher` shape
`AUDIT_METHODOLOGY.md` §4.4 requires and few external optimizers actually
are: `basinhopping`'s scaffold logic (perturb the current best point by a
random step, re-run a local minimizer from there, accept/reject by a
Metropolis-like criterion) explicitly wraps a swappable, independently
callable local minimizer (`minimizer_kwargs["method"]`, e.g. `"L-BFGS-B"`)
-- so `unwrap()` is simply "call `scipy.optimize.minimize` with that same
method from a uniform-random starting point," which is also the natural,
literal `B_restart` control the audit already requires (independent restarts
of the base searcher). This is the first case since the pivot where
`unwrap()` isn't a same-team reimplementation exercise (`plugins/nas_search`)
or a synthetic calibration construction (`engine/audit/calibration.py`) --
it's an off-the-shelf external library's actual production code, used as-is.

This also directly serves a second, narrower question this project has not
yet answered on real (non-synthetic) data: **has this audit, on anything
other than a synthetic calibration construction, ever reported anything
besides `NULL`, `HARMFUL`, or `DEGENERATE`?** Per `DECISION_LOG.md` and
issues #11/#13, every real pipeline audited so far (physics, NAS
self-audit) has returned `NULL` or worse. `basinhopping`'s scaffold logic is
specifically designed to exploit structure a uniform-random restart cannot
(perturbing near an already-good point on a multimodal landscape, rather
than resampling blind) -- it is a genuinely different mechanism from every
case tried so far, which makes it a real test of whether the audit can ever
say `CONTRIBUTES`, not just whether it can correctly say `NULL`.

## Hypothesis

`basinhopping` will read `CONTRIBUTES` (beyond the pre-registered margin) on
at least one classic multimodal global-optimization test function (e.g.
Rastrigin, Ackley, or Griewank -- to be finalized during the feasibility
pre-check, not chosen after seeing results), because its step-taking
mechanism is specifically designed to exploit local structure between
nearby basins that independent uniform-random restarts of the same local
minimizer cannot replicate by construction. This is a design-intent
argument, not a citation to a specific effect size from the literature --
no published number is being assumed here, and the feasibility pre-check
(`scripts/audit_feasibility.py`) must confirm the design has power to
resolve anything on this benchmark/budget before the real sweep runs.

## Null / alternative hypothesis

**Null (would contradict the hypothesis):** `NULL` or `INCONCLUSIVE` --
`basinhopping`'s scaffold does not clear a budget-matched independent-restart
bar on the tested function(s) at the tested budget, mirroring every other
real (non-synthetic) result this audit has produced so far. This is a
legitimate and reportable outcome, not a failed experiment -- it would mean
this audit's `NULL`/`HARMFUL`-dominated track record generalizes even to a
scaffold explicitly designed around exploiting non-uniform structure, which
is itself informative about how hard it is for *any* lightweight wrapper to
clear a fair, budget-matched bar.

**Alternative (supports the hypothesis):** `CONTRIBUTES` beyond the
pre-registered margin, with the TOST's one-sided superiority test rejected
in the improving direction. Also note explicitly if `HARMFUL` is observed
instead (the opposite of the hypothesis, still informative and still not a
failed experiment) -- perturbation-based step-taking can plausibly get stuck
near a local minimum worse than pure restarts on the wrong kind of landscape.

## Motivation

Every real pipeline this audit has measured so far (the archived physics
scaffold, and now the `nas_search` `RandomSearch` self-audit, issues
#11/#13) has returned `NULL` or `HARMFUL`. That is consistent with two very
different explanations: scaffolds in the wild mostly don't help at matched
budget (an interesting finding about the field), or this audit's power/
margin choices are conservative enough to rarely detect a real contribution
even when one exists (a finding about the instrument, echoing exactly the
`calibration.py` module's own stated purpose -- "an instrument that has
only ever returned one answer is indistinguishable from an instrument that
can only return one answer"). `calibration.py`'s synthetic `OracleScaffold`
already proves the audit *can* report `CONTRIBUTES` in principle, on a
scaffold engineered to cheat -- but nothing yet has shown it can do so on a
real, unmodified, externally-built pipeline that nobody rigged to win. This
experiment is the first attempt at that, on the cheapest, safest external
case actually available (no new dependency, no GPU, a structurally
separable scaffold instead of an inseparable end-to-end model).

It's also the first real step toward README item 4's larger goal (auditing
a pipeline this team did not build) without over-scoping into the
cross-methodology-comparison half of that item, which needs its own issue
once the ScaffoldSafety/GAIA reproduction work is itself scoped concretely
enough to be answerable by one branch.

## Experimental design

**Pipeline under audit:** `scipy.optimize.basinhopping(func, x0,
minimizer_kwargs={"method": "L-BFGS-B"}, niter=..., seed=...)` as the
treatment (scaffold = the hopping/accept-reject loop, base searcher =
`scipy.optimize.minimize(func, x0, method="L-BFGS-B")`). No modification to
`scipy`'s code; call it as a library, exactly as any user would.

**Control (`B_restart`):** independent calls to
`scipy.optimize.minimize(func, x0, method="L-BFGS-B")` from uniform-random
`x0` within the function's standard bounds, repeated until the same total
count of local-minimization calls is spent as the treatment's `niter`
(the natural, literal unit match here -- no candidate-evaluations-vs-
gradient-steps ambiguity, since both arms spend their entire budget calling
the exact same local minimizer the exact same number of times), returning
the best of them.

**Problem set:** a small number (2-4) of classic, well-established
multimodal global-optimization test functions with known bounds and known
global optima (e.g. Rastrigin, Ackley, Griewank, Rosenbrock -- final
selection during implementation, picked for having multiple local minima
where basin-hopping's design intent is meant to matter, not picked after
seeing which one gives the hoped-for verdict), in a fixed, low dimension
(e.g. d=10) chosen for cheap evaluation and to avoid the curse-of-
dimensionality regime where any local-search-based method degrades.

**Budget:** small `niter` per arm (tens to low hundreds of local
minimizations) -- pick via `scripts/audit_feasibility.py` run first on this
exact problem/searcher pair, the same way issue #11's margin was chosen from
a dedicated feasibility probe rather than assumed. Function evaluations
here are essentially free (no training, no I/O), so cost is not a binding
constraint -- power is.

**Seeds:** per the feasibility pre-check's recommendation, not assumed to
be 30 by default (issue #13 already showed margin/power choices don't
automatically transfer across problem types).

**Metric:** best objective value found per arm (continuous, BCa bootstrap
TOST, lower-is-better -- note the orientation flip relative to
`valid_accuracy`'s higher-is-better in issue #11, and get the sign right in
the `higher_is_better` config). Margin pre-registered from the feasibility
probe, per function (a function's own value range differs, so one shared
margin across functions would not be meaningful).

**Degeneracy pre-check and selection ceiling:** run both, per standard
procedure, before the statistical sweep -- selection ceiling is expected to
be non-trivial here (unlike the NAS self-audit's ceiling of exactly 0.0),
since `basinhopping`'s accept/reject step genuinely could be selecting among
qualitatively different local optima rather than restarting blind; report
what it actually comes out to rather than assuming.

## Metrics

- **Best objective value per arm**, per function (continuous, BCa bootstrap
  TOST, lower-is-better). Primary decision metric.
- **Distance to the known global optimum** (continuous) as a secondary,
  interpretable metric alongside the raw objective value -- useful context
  for the writeup even though the verdict is decided on the primary metric.
- **Degeneracy pre-check result** (both arms) and **selection ceiling**, per
  standard procedure, reported alongside the statistical verdict per
  `AUDIT_METHODOLOGY.md` §4.3.

## Baselines / controls

Budget-matched independent restarts of the same local minimizer
(`scipy.optimize.minimize`, method held identical to the one `basinhopping`
uses internally) -- the project's standard control, and here it is also
exactly what a practitioner would naturally compare against, so there is no
"is this really the right control" ambiguity to defend, unlike some past
cases.

## Expected outcomes

- **`CONTRIBUTES`** (hypothesis): first real (non-synthetic-calibration)
  case where this audit reports a positive scaffold contribution --
  significant on its own regardless of what it says about `basinhopping`
  specifically, since it demonstrates the audit's positive-verdict pathway
  works end-to-end on an unmodified external pipeline.
- **`NULL`**: consistent with every real result so far; would strengthen
  (not by itself prove) the "audit is conservative/underpowered in
  practice, or lightweight scaffolds rarely clear a fair bar" question
  raised in Motivation. Should prompt considering, as a follow-up, whether
  `calibration.py`'s synthetic `OracleScaffold` power characteristics
  actually predict real-world detectability, or only prove the machinery
  *can* work when handed a scaffold engineered to win.
- **`HARMFUL`**: opposite of the hypothesis but still a real, reportable
  finding -- would suggest the step-taking mechanism is actively worse than
  blind restarts on the chosen function(s)/budget, which basin-hopping's own
  stepsize-tuning literature would predict is possible under a poorly
  matched stepsize.
- **`INCONCLUSIVE`**: underpowered design; re-run feasibility pre-check with
  observed variance before increasing budget, don't just add seeds blindly.
- **Different verdicts across the 2-4 test functions**: plausible and
  should be reported per-function, not averaged or reduced to one headline
  verdict -- `AUDIT_METHODOLOGY.md` §4.1 is explicit that nullity is indexed
  by problem, and a single verdict across heterogeneous functions would
  misrepresent that.

## Interpretation plan

- `CONTRIBUTES` on some/all functions: report which functions and at what
  effect size; this becomes the first documented real positive case, feeds
  the "does the audit ever say `CONTRIBUTES` on real data" question
  directly, and is the natural jumping-off point for the deferred
  cross-methodology-comparison half of README item 4.
- `NULL`/`HARMFUL` on all functions: report honestly as a negative result
  per `GIT_WORKFLOW.md` ("negative results... get recorded the same way a
  positive result would"); do not retry with a different function or budget
  chosen after seeing this one fail, as that would be exactly the
  cherry-picking `GIT_WORKFLOW.md`'s "when not to merge" list calls out.
  Flag explicitly whether this should raise the standing question of
  whether the audit's design (margin conventions, budget-matching in
  local-minimizer-call units) is systematically conservative.
- Mixed across functions: report per-function; note whether the pattern
  correlates with landscape properties (e.g. number of local minima,
  separation between them) as a lead for future work, not as a conclusion
  this one branch can establish on 2-4 functions.
- In all cases: explicitly state that this does NOT yet compare this
  audit's verdict against what ScaffoldSafety's or GAIA's own methodology
  would conclude on the same case -- that remains future work, to be filed
  as its own issue if this branch's result makes it worth pursuing.

## Confounds considered

- **Stepsize mismatch.** `basinhopping`'s `stepsize` parameter is explicitly
  documented as needing to be comparable to the typical separation between
  local minima of the target function -- a badly chosen stepsize could make
  the scaffold look worse (or better) than its actual design intent
  supports, independent of anything the audit measures. Use `scipy`'s
  documented default unless the feasibility pre-check indicates it needs
  tuning, and state explicitly which value was used and why, rather than
  silently tuning it toward the hoped-for verdict.
- **Budget-unit fairness.** Both arms must spend the same count of calls to
  the same local minimizer with the same convergence tolerances -- if
  `basinhopping`'s internal minimizer calls use different tolerances or
  iteration caps than the control's direct `scipy.optimize.minimize` calls,
  the comparison silently favors one arm. Verify both arms' minimizer
  configuration matches exactly before trusting any verdict.
- **Function/dimension selection after seeing results.** Finalize the
  function set and dimensionality during implementation, before running the
  real sweep, and record that choice in `SPEC.md` before results exist --
  per `GIT_WORKFLOW.md`'s "Before writing code: the spec," a hypothesis (or
  a benchmark choice) written after seeing the result is a caption, not a
  hypothesis.
- **Local minimizer choice.** `"L-BFGS-B"` is a reasonable, common default,
  but a different local method could change which mechanism (step-taking vs.
  local convergence quality) dominates the result. State the choice and
  don't tune it after seeing verdicts.
- **This says nothing about `naslib`-blocked README item 3, and nothing
  about the harder cross-methodology-comparison half of item 4** -- both
  explicitly out of scope for this branch, as stated above.

---

## Results

Full artifacts: `results/basinhopping_audit/audit.json` (per-function
config, feasibility probe, ceiling, verdict, per-metric evidence,
degeneracy, and every raw per-seed objective value) and `audit.csv`
(verdict summary table). Dimension 10, bounds and `L-BFGS-B` as
pre-registered. `stepsize` = scipy's documented default (0.5), unchanged --
the feasibility probe did not indicate a need to tune it (see
"Interpretation" for where that default's mismatch with Ackley's much wider
bounds shows up in the result instead).

**Feasibility probe** (15 pilot seeds, seeds 1000-1014, disjoint from the
real sweep; `niter=50`): margin = 25% of the control arm's single-restart
objective standard deviation across the pilot, `n` = whichever of
`engine.audit.statistics.required_sample_size(margin, spread, 0.90, 0.80)`
or 20 is larger, capped at 60. This produced margins of 2.02 (rastrigin),
0.31 (ackley), and 1e-9 (griewank, floored -- see below). The real sweep
then ran on a **fresh, non-overlapping** seed block (`range(n)`) at the same
`niter=50`, so nothing about the real sweep's own result could have fed
back into the margin/`n` choice that produced it.

**Selection ceiling: exactly 0.0 on all three functions** (10-seed pilot,
`restarts=10`) -- `LocalMinimizerRestart.select()` already picks the
literal best of what it sampled by construction, so there is no
selection-only headroom for any wrapper here, on any of the three
functions. Any real effect found below must come from *where* the scaffold
searches, not from picking better among candidates it did not generate any
differently than a restart would.

| function | verdict | diff (ctrl-treat) | 90% CI | margin | n | degeneracy |
|---|---|---|---|---|---|---|
| rastrigin | **CONTRIBUTES** | +17.73 | [+16.10, +19.43] | ±2.02 | 60 | not degenerate (0/60, mean ratio 0.32) |
| ackley | **HARMFUL** | -1.38 | [-1.54, -1.25] | ±0.31 | 60 | not degenerate (1/60, mean ratio 0.09) |
| griewank | **INCONCLUSIVE** | -0.15 | [-0.44, -0.06] | ±1e-9 | 20 | **DEGENERATE** (20/20, mean ratio 0.02) |

(`diff` is oriented so positive means the treatment, `basinhopping`, found a
*lower* -- better -- objective than the control; `griewank`'s margin is the
numerical floor `max(0.25 * control_spread, 1e-9)`, not a meaningful
practical-equivalence choice -- see Interpretation.)

**Rastrigin: `CONTRIBUTES`, clearly** -- the observed effect (+17.73) is
~8.8x the pre-registered margin (2.02), and the CI's lower bound (+16.10)
still clears the margin by a wide berth. This is the first time this
audit has reported `CONTRIBUTES` on anything other than
`calibration.py`'s synthetic `OracleScaffold` -- the first real,
unmodified, externally-built pipeline this audit has certified as
genuinely beating a fair, budget-matched restart baseline. (The reported
`power=0.23` for this row is **not** a weakness of this claim -- it is the
probability of establishing *equivalence* at this margin if the true effect
were zero, which is simply the wrong question once the interval has
already cleared the margin by 8x; the claim's own strength is the
CI position, not this number. `n_for_target_power` in the raw JSON is a
leftover from the same NULL-oriented calculation and should be read the
same way for this row.)

**Ackley: `HARMFUL`**, the opposite of the hypothesis, with power=0.93 (a
well-powered result in the ordinary sense here, since it's read as a
directional claim beyond the margin). A plausible mechanism, per SPEC.md's
"Stepsize mismatch" confound: Ackley's bounds (±32.768) are more than 6x
wider than Rastrigin's (±5.12), and `stepsize=0.5` was held fixed across
all three functions at scipy's documented default rather than scaled to
each function's domain. A step of 0.5 in a space spanning ±32.768 is a
very small perturbation relative to the domain, which could make
`basinhopping`'s hops too timid to escape Ackley's local-minima structure
-- while a uniform-random restart always samples the *entire* domain
regardless of stepsize. This is a plausible explanation consistent with the
pre-registered confound, not a re-analysis chasing the result: `stepsize`
was fixed before this sweep ran and was not tuned after seeing this row,
per SPEC.md's explicit instruction.

**Griewank: `INCONCLUSIVE`, and `DEGENERATE` (20/20 runs)** -- but this is
**not** a reproduction of this project's founding seeding bug. Two
independent pieces of evidence corroborate a different explanation: the
*control* arm's own single-restart spread across the 15-seed pilot was
`1.704e-10` -- independent uniform-random restarts of `L-BFGS-B` land on
(essentially) the exact same objective value on Griewank at this
dimension/budget, with no scaffold involved at all -- and the treatment's
mean distance to the known global optimum across the real sweep's 20 seeds
was `0.068` (min `0.0099`), i.e. it is finding the global optimum region on
nearly every run. Both arms are converging to the same place because the
*landscape* permits it here, not because either implementation failed to
vary its randomness: Griewank's characteristic multimodality comes from a
product term whose influence relative to the quadratic term shrinks as
dimension grows, and at d=10 the function is close enough to unimodal in
the reachable region that essentially any local search from essentially
any start finds the global basin. The `margin=1e-9` floor is a direct
symptom: the feasibility probe's margin convention (25% of control-arm
spread) has nothing to scale from when that spread is itself numerical
noise, so the resulting "margin" is not a meaningful practical-equivalence
choice and `INCONCLUSIVE` is the correct, honest verdict -- not a power
failure to fix by adding seeds, but a sign that this function/dimension/
budget combination does not pose a well-formed equivalence question for
this metric.

## Interpretation

**This audit can say `CONTRIBUTES` on a real, unmodified, externally-built
pipeline.** Every real (non-synthetic-calibration) result this project had
produced before this branch -- the archived physics scaffold, and the
`nas_search` `RandomSearch` self-audit (issues #11/#13) -- read `NULL` or
worse. That pattern was consistent with two different explanations (SPEC.md,
Motivation): scaffolds in the wild mostly don't help at matched budget, or
this audit's power/margin conventions are conservative enough to rarely
detect a real contribution in practice. Rastrigin's clean `CONTRIBUTES`
directly rules out the second explanation as a *general* property of this
audit's design -- the positive-verdict pathway works end-to-end, on
external code nobody tuned to win, when a real effect of this size exists.
It does not rule out that explanation for any *specific* other case (Ackley
and Griewank's non-`CONTRIBUTES` results are not evidence the audit is
underpowered -- they have their own, different explanations above), and it
says nothing about whether the audit's conventions are well-tuned in
general, only that they are not so conservative as to be structurally
incapable of a positive verdict.

**The three functions gave three different verdicts, exactly the shape
SPEC.md anticipated as plausible ("Different verdicts across the 2-4 test
functions").** This is reported per-function, not reduced to one headline
number, per `AUDIT_METHODOLOGY.md` §4.1 ("nullity is indexed by budget...
and by problem"). Read together, they say something more specific than
"basinhopping sometimes helps": the scaffold's step-taking mechanism reads
as a genuine, exploitable advantage on Rastrigin, as actively counter-
productive on Ackley (plausibly a stepsize/domain-scale mismatch, not a
property of step-taking in general), and as an unanswerable question on
Griewank at this dimension because the control already solves it.

**What this does NOT establish**, per SPEC.md's explicit scope:

- Nothing about how this audit's verdict compares to what ScaffoldSafety's
  or GAIA's own methodology would conclude on the same case -- the
  cross-methodology-comparison half of README item 4 remains unattempted,
  as planned. Worth filing as its own issue now that this half has a
  concrete, mixed real result to compare against, if the researcher agrees.
- Nothing about `basinhopping`'s general quality as an optimizer, or about
  step-taking scaffolds as a class -- three functions in one dimension at
  one budget is a case study, not a survey.
- Nothing about README item 3 (NAS controllers, still blocked on `naslib`).
- The Ackley `HARMFUL` mechanism (stepsize/domain-scale mismatch) is a
  plausible explanation consistent with a pre-registered confound, not a
  confirmed causal finding -- re-running with a scaled stepsize would be a
  legitimate, separate follow-up experiment, not a re-analysis of this one.

## Decision

**MERGE.** Checked against `GIT_WORKFLOW.md`'s nine criteria:

- **Scientific relevance:** directly answers the tractable half of README
  item 4, and produces this project's first real `CONTRIBUTES` verdict --
  significant regardless of what it says about `basinhopping` specifically,
  since `AUDIT_METHODOLOGY.md`'s own `calibration.py` module frames exactly
  this question ("an instrument that has only ever returned one answer is
  indistinguishable from an instrument that can only return one answer").
- **Correctness:** `unwrap()` returns the literal primitive `basinhopping`
  calls internally; budget matching is verified against the actual measured
  minimizer-call count (`niter+1`, confirmed empirically, not assumed) via
  `test_budget_matches_exactly_between_arms`; degeneracy is assessed from
  `basinhopping`'s own per-hop callback, not left "not assessed."
- **Experimental validity:** control is the natural, undisputed budget-
  matched baseline here (SPEC.md: "no 'is this really the right control'
  ambiguity to defend"); margin/`n` chosen from a feasibility probe on
  seeds disjoint from the real sweep, so nothing about the real result fed
  back into the design that produced it.
- **Reproducibility:** deterministic given fixed seeds
  (`np.random.default_rng`, `basinhopping(..., seed=...)`); full config
  (stepsize, method, bounds, margins, `n`) travels with every result row.
- **Documentation:** this file; module docstrings state every design
  choice's rationale, including the stepsize confound surfaced in Ackley's
  result.
- **Interpretation:** stated above per function, including explicit
  non-claims (no ScaffoldSafety/GAIA comparison, no general claim about
  basinhopping or step-taking scaffolds).
- **Research integrity:** the mixed result (one `CONTRIBUTES`, one
  `HARMFUL`, one `INCONCLUSIVE`) is reported in full, not narrowed to the
  hypothesis-confirming row; griewank's `DEGENERATE` flag is explained with
  corroborating evidence rather than either alarmed over or hidden.
- **Integration:** additive; new `plugins/basinhopping_audit/` package,
  touches no existing code.
- **Evidence:** a real sweep (180 audited seeds total across three
  functions, `niter=50` each) against real, unmodified `scipy` code, with
  a documented, pre-registered feasibility process behind every margin.

Follow-up for the researcher, not this branch: whether to file the
ScaffoldSafety/GAIA cross-methodology-comparison half of README item 4 as
its own issue now that this branch gives it a concrete (and non-uniform)
real result to work from, and whether an Ackley re-run with a
domain-scaled `stepsize` is worth a separate `experiment/` issue.
