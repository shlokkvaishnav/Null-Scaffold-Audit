<!--
Copied verbatim from GitHub issue #11, per research/GIT_WORKFLOW.md and
research/SPEC_TEMPLATE.md. Section headers below follow SPEC_TEMPLATE.md;
their content is the issue body's corresponding section, unedited.
-->

# Spec: research/nas-random-search-self-audit

**Branch:** `research/nas-random-search-self-audit`
**Date opened:** 2026-08-23
**Status:** IN PROGRESS
**Issue:** https://github.com/shlokkvaishnav/Null-Scaffold-Audit/issues/11

## Branch type

research/ -- substantial implementation or investigation

## Research question

README item 2 ("Open research questions / next experiments") calls for
implementing `plugins/nas_search/` "starting with the cheapest possible
sanity check -- audit `RandomSearch` against itself, which should return
`NULL` -- before adding a real controller." This has not been started.
`plugins/nas_search/` does not exist yet (confirmed: only `plugins/__init__.py`
and `plugins/README.md` exist in the tree).

The question this issue scopes is narrower than "build the NAS plugin": when
the audit's `BaseSearcher`/`AuditProblem` protocols (`engine/audit/arms.py`)
are implemented for the first time against a real external benchmark
(NATS-Bench, tabular, CIFAR-10 topology search space) instead of a synthetic
or in-house problem, and that concrete `RandomSearch` base searcher is
audited against a scaffold that does nothing but call it with independently
seeded restarts -- does the audit correctly return `NULL` at a pre-registered
margin, matching what `engine/audit/calibration.py`'s domain-independent
`NullScaffold` already predicts abstractly?

This is explicitly a validation question about the *implementation*, not a
scientific claim about NAS controllers. `engine/audit/calibration.py`
already proves the audit's statistical machinery produces `NULL` for a
scaffold that is null by construction -- but that proof runs on synthetic
scaffolds and searchers that never touch a real tabular benchmark's
sampling, seeding, or discrete architecture-space quirks. This experiment is
the first time those protocols are wired to something real, and is
therefore the first point where an implementation bug (e.g. a seed not
actually threaded into the sampler, silently returning the same architecture
on every "independent" restart) could hide. That exact failure mode --
a scaffold that looks like it's doing something but is silently returning
identical output across iterations -- is this project's own origin story
(README "Why this matters" / `AUDIT_METHODOLOGY.md` §2), so this is the
cheapest possible check that the NAS plugin doesn't reproduce it before any
real controller (DARTS/GDAS/RegularizedEvolution/BANANAS, README item 3) is
audited on top of it.

Critically: this does **not** require `naslib`, which is the currently
unresolved install blocker (README item 1, `DECISION_LOG.md`). It only needs
`nats-bench`, which is already a declared, cleanly-resolving dependency
(`pyproject.toml` line 49, `nas = ["nats-bench"]`) and requires no GPU
training -- NATS-Bench is a precomputed tabular lookup. This work can
therefore proceed in parallel with item 1 rather than waiting on it, and
unblocks item 3 the moment item 1 is resolved.

## Hypothesis

The audit will return `NULL` (within the pre-registered margin, on all
declared metrics) when `RandomSearch` on NATS-Bench-CIFAR10 is audited
against a scaffold that wraps it with nothing but independent-seed restarts
at matched budget.

This is expected on statistical grounds alone -- there is no scaffold logic
to detect, by construction -- and `calibration.py`'s `NullScaffold` already
demonstrates the audit machinery can produce this verdict in the abstract.
The reason to run it anyway, rather than treat it as a foregone conclusion,
is that the hypothesis is about the *new* code (the NATS-Bench `AuditProblem`
and `RandomSearch` `BaseSearcher` implementations), not about the
already-validated statistical core -- and this project's founding incident
is precisely a case where the obviously-expected behavior didn't hold
because of an implementation bug nobody had checked for.

## Null / alternative hypothesis

**Null (expected):** TOST establishes equivalence within the pre-registered
margin `delta` on every declared metric -- verdict `NULL`, and the
degeneracy pre-check does not flag either arm.

**Alternative outcomes that would contradict the hypothesis, stated
concretely so this can actually fail:**

- Verdict `CONTRIBUTES` or `HARMFUL` on any metric -- since there is no
  scaffold behavior by construction, this would mean the two arms are not
  actually matched (e.g. asymmetric budget accounting, a metric computed
  differently per arm, or a selection rule applied inconsistently), not a
  real effect.
- Degeneracy pre-check fires on either arm (`DEGENERATE`) -- intra-run
  proposals are identical to one another, meaning `RandomSearch`'s seeding
  is not actually varying its samples across restarts. This is the
  archived-pipeline failure mode reproduced in the new domain, and would be
  the single most useful thing this experiment could find.
- `INCONCLUSIVE` at the planned budget/seed count -- would mean the design
  (budget, seeds, or margin) needs revision before it's trustworthy for the
  real controller study in README item 3, not that the scaffold is null.
- `NOT_SEPARABLE` -- would mean the `unwrap()` contract was implemented
  incorrectly for this trivial case, which would make every downstream
  controller audit on this plugin unusable regardless of the controller's
  own behavior.

## Motivation

This project's thesis is "does the scaffold earn its keep," and the
platform's answer only means something if the platform's own measurement
apparatus is trustworthy on the domain it's being pointed at. Right now
that apparatus (`engine/audit/`) has been validated on exactly two in-house
domains (physics, one synthetic problem, per `DECISION_LOG.md`'s pivot
entry) and zero external benchmarks. NATS-Bench is the first external
benchmark this project touches, and nothing has yet confirmed the
`BaseSearcher`/`AuditProblem` protocols behave correctly against a real
tabular lookup's seeding and discrete search space, as opposed to the
synthetic constructions in `calibration.py`.

If this comes back `NULL` cleanly: it validates the NAS plugin's plumbing
before any compute is spent auditing a real controller, and directly
unblocks README item 3 the moment the `naslib` blocker (item 1) is
resolved -- this branch's `RandomSearch` `BaseSearcher` is exactly the
control arm item 3 needs.

If it doesn't come back `NULL` cleanly: it means a real controller audit run
on top of this plugin today would be uninterpretable, and finds that out for
the cost of one cheap tabular-lookup sweep instead of after a compute-heavy
DARTS/GDAS run whose result would otherwise be trusted incorrectly.

## Experimental design

**Domain / pipeline:** New `plugins/nas_search/` package, minimal scope for
this issue only:

1. `AuditProblem` wrapping a NATS-Bench dataset split (recommend CIFAR-10,
   the topology search space -- 15,625 architectures, matching the
   benchmark choice already recorded in `DECISION_LOG.md`). Opaque to the
   audit per `AUDIT_METHODOLOGY.md` §4.4 -- it exposes candidates and
   accepts evaluation calls, nothing about "architecture" leaks into
   `engine/`.
2. `RandomSearch` `BaseSearcher`: on each `search(problem, budget, seed)`
   call, uses `seed` to seed an independent RNG stream, samples uniformly
   from the NATS-Bench architecture index without replacement within a
   single call, spends the full `budget` in candidate evaluations (one
   evaluation = one tabular lookup), and returns the best candidate found
   under a fixed, stated selection rule (e.g. highest reported validation
   accuracy at the benchmark's standard training epoch count).
3. A restart scaffold whose `run()` does nothing beyond what `B_restart`
   already means in `AUDIT_METHODOLOGY.md` §4.1 -- call it
   `IdentityRestartScaffold` -- that calls `base.search()` with an
   independent seed derived from (but not identical to) the seed the
   control arm receives for that trial index, and whose `unwrap()` returns
   the same `RandomSearch` instance. This is the domain-specific analogue of
   `calibration.py`'s `NullScaffold`, not a new scaffold design -- it must
   not share the exact seed stream with its own control arm, or the
   comparison degenerates into "compare a run against a byte-identical copy
   of itself," which is a different (and uninteresting) test than the one
   this issue asks.

**Held constant:** benchmark (NATS-Bench-CIFAR10 topology space only),
selection rule, budget-per-arm, evaluation cost model (tabular lookup, no
real training).

**What varies:** nothing intentionally -- this is a calibration run, and any
observed difference between arms is either noise or a bug (see Null /
alternative hypothesis above).

**Budget:** run `scripts/audit_feasibility.py` first against this new
`AuditProblem`/`BaseSearcher` pair to pick a budget and seed count with
adequate power at the chosen margin, rather than assuming the existing
30-seed convention transfers to a discrete tabular domain with different
variance properties. Report whatever the feasibility gate recommends; do
not shrink it after seeing early results.

**Selection ceiling:** run `scripts/measure_selection_ceiling.py` and expect
it to report a ceiling of (near) zero, since there is no selection
advantage available to a scaffold that is not doing anything -- record this
as an expected, not anomalous, result and explain it in the writeup so it
isn't misread as a design flaw later.

## Metrics

- **Best validation accuracy per arm** (continuous) -- primary metric,
  tested with the BCa bootstrap TOST per `AUDIT_METHODOLOGY.md` §4.2.
- **Exact-architecture-match rate** (paired-binary: did the two arms'
  best-found architecture ID match on a given seed pair) -- declared in
  advance as a Tango-score-interval metric per §4.2, not inferred after
  seeing whether it collapses to all-agree or all-disagree. This metric is
  informative here specifically because ties in NATS-Bench's tabular
  records make exact-match plausible even under independent sampling, which
  is exactly the collapsed-sample scenario §4.2 warns produces false
  certainty under the bootstrap.
- **Degeneracy pre-check result** for both arms (not a statistical metric,
  reported separately per §4.3, but a required pass/fail gate before the
  statistical procedure is trusted at all).

## Baselines / controls

Control is budget-matched independent restarts of the same `RandomSearch`
`BaseSearcher` -- the project's standard control, not "no scaffold" (there
is no scaffold to remove here in the usual sense; both arms run the same
base searcher). This is a deliberate degenerate case of the standard design
rather than a deviation from it: treatment = base searcher wrapped in a
scaffold that adds nothing, control = base searcher restarted directly. No
other baseline is used.

## Expected outcomes

- **`NULL`, cleanly, on all metrics, no degeneracy flag** (expected/hoped):
  validates the plugin's protocol implementation; README item 3 (real
  controller audit) can proceed on this base the moment `naslib` is
  resolved.
- **`DEGENERATE`**: `RandomSearch`'s seeding does not vary samples across
  restarts -- the archived-pipeline bug, reproduced in the new domain. Fix
  before anything else in this plugin is trusted.
- **`CONTRIBUTES` or `HARMFUL`**: the two arms are not actually matched
  (budget, metric computation, or selection rule differs between them) --
  an implementation bug, not a scaffold effect. Root-cause before proceeding.
- **`INCONCLUSIVE` (underpowered)**: the budget/seed count chosen from the
  feasibility pre-check was insufficient in practice; increase and re-run
  rather than reporting `NULL` by default (`AUDIT_METHODOLOGY.md` §3 is
  explicit that `INCONCLUSIVE` must never be reported as `NULL`).
- **`NOT_SEPARABLE`**: `unwrap()` was implemented incorrectly for this
  trivial case -- must be fixed before this plugin is usable for anything.

## Interpretation plan

- `NULL` cleanly: plugin plumbing validated on this domain; close this issue
  as MERGE-track and file the README-item-3 issue (real controller vs.
  budget-matched random search) as a separate, properly scoped issue per
  `AGENT_PIPELINE.md` -- do not fold controller work into this branch.
- `DEGENERATE`: this branch's outcome is the finding itself (a caught bug,
  not a shipped feature) -- fix the seeding, re-run this same experiment
  before merging anything, and record the specific defect in
  `DECISION_LOG.md` regardless of the final verdict, the way the original
  physics-scaffold degeneracy was recorded.
- `CONTRIBUTES`/`HARMFUL`: means nothing about NAS controllers -- it means
  the harness is broken. Do not report it as a scaffold finding anywhere;
  root-cause the asymmetry (most likely candidates: budget accounted in
  different units per arm, or the selection rule applied only on one side)
  before re-running.
- `INCONCLUSIVE`: re-run the feasibility pre-check with the observed
  variance, raise budget/seeds accordingly, and only then re-attempt a
  verdict.
- `NOT_SEPARABLE`: fix `unwrap()`; this result blocks every other outcome in
  this list from being reachable at all, so treat it as a hard stop rather
  than one branch among several.

## Confounds considered

- **Shared seed streams collapsing the comparison.** If the scaffold's
  restart and the control's restart draw from the same seed for a given
  trial index, the two arms are literally the same run duplicated, which
  would produce a spurious perfect match on every metric -- not evidence of
  a correctly functioning audit, but evidence the test was never really run.
  Mitigation: derive the scaffold's internal seed from, but distinct from,
  the control's seed for that trial (matching how `AUDIT_METHODOLOGY.md`
  §4.1 defines `B_restart` as independent restarts), and confirm via a
  quick check that the two arms' raw candidate sequences are *not*
  byte-identical before trusting the statistical result.
- **NATS-Bench's tabular records include measurement noise from multiple
  training seeds per architecture** (the benchmark itself reports several
  trained instances per architecture in some splits). Which statistic is
  read (mean over training seeds vs. a single recorded seed) must be fixed
  in the `AuditProblem` implementation and held identical for both arms --
  if one arm's evaluation reads a different statistic than the other, that
  alone could manufacture an apparent effect with no real cause.
- **Ties in the discrete architecture space.** Because validation accuracies
  in a tabular table repeat across many architectures, "exact architecture
  match" and "exact accuracy match" are not the same event, and conflating
  them could make the paired-binary metric look artificially concordant or
  discordant depending on which is actually measured. Pick one, state which,
  and don't switch mid-analysis.
- **Selection-ceiling gate reporting near-zero is expected here, not a
  failure of the design** -- since neither arm has any information the
  other lacks, there's no selection advantage available in principle. This
  must be stated in the writeup so it isn't misread later as "the audit
  couldn't detect anything" when the correct reading is "there was nothing
  to detect, by construction."
- **This experiment cannot say anything about whether real NAS controllers
  beat random search** -- that's README item 3, deliberately out of scope
  here per `AGENT_PIPELINE.md`'s scoping rule ("don't inflate a small
  question... don't compress a substantial investigation"). This issue is
  about whether the measurement apparatus is trustworthy, not about NAS.

---

## Results

Full artifacts: `results/nas_search_self_audit/audit.json`,
`results/nas_search_self_audit/audit.csv`. Config: NATS-Bench-tss "simple"
file (v1.0-3ffb9), dataset `cifar10-valid`, hp `200` (full 200-epoch
training record), `is_random=False` (mean over recorded training seeds).
`RandomSearch(budget=50)`, `IdentityRestartScaffold(restarts=3)`, 30 seeds,
90% confidence.

**Pre-registered margin** (fixed before the 30-seed sweep below ran, from a
separate 20-seed feasibility probe on this exact problem/searcher pair):
`valid_accuracy` margin = 0.3 percentage points. That probe measured a
paired-difference spread of 0.315pp, giving a minimum detectable effect of
0.143pp at n=30 (alpha=0.05 one-sided, power=0.80) -- comfortably resolvable
at the chosen margin. See `plugins/nas_search/run_self_audit.py`'s module
docstring for the full derivation.

**Selection ceiling** (`scripts/measure_selection_ceiling.py`'s underlying
`engine.audit.calibration.selection_ceiling`, 10 seeds, 3 restarts): ceiling
= 0.0 exactly (`ceiling_upper` = 0.0), against a metric spread of 0.36pp
within a seed's sampled candidates -- the "faithful" case
(`measure_selection_ceiling.py`'s own classification), not "tied": restarts
land on genuinely different architectures, and `RandomSearch.select()`
already picks the best of what it sampled every time, so there is no
selection-only headroom available for any wrapper at this budget. This
matches the expected outcome stated above exactly.

**Audit** (30 seeds, budget=50/restart, restarts=3 -> 4500 evaluations per
arm, matched exactly):

| metric | verdict | observed diff | 90% CI | margin | power | p |
|---|---|---|---|---|---|---|
| valid_accuracy | **NULL** | -0.100pp | [-0.197, -0.008] | ±0.3pp | 0.998 | 0.00092 |

- **Overall verdict: NULL.**
- **Degeneracy pre-check: not degenerate.** 0/30 runs produced a single
  distinct proposal; mean distinct ratio 1.00 (every one of the 3 restarts
  within every one of the 30 scaffold invocations landed on a different
  architecture). This directly rules out the founding-incident failure mode
  (a fixed/unthreaded seed collapsing "independent" restarts to the same
  output) for this plugin's `RandomSearch` implementation.
- **Identical-representation rate: 0.000** (0/30 seeds). The comparison is
  not vacuous (`AuditReport`'s vacuous-comparison guard did not fire, and
  did not need to: `audit.json`'s `arms.treatment_representations` and
  `arms.control_representations` show no seed where the two arms picked the
  same architecture index).
- **Exact-architecture-match rate** (SPEC.md's declared paired-binary
  metric): reported via `arms.identical_representation_rate` rather than
  a second Holm-corrected margin verdict -- see
  `plugins/nas_search/run_self_audit.py`'s module docstring for why this
  metric's shape (a comparison *between* the two arms' outcomes) does not
  fit `engine.audit.arms.audit()`'s `margins`/`paired_binary` machinery,
  which tests two independent *per-arm* success rates. This is an
  implementation-level finding from writing this branch, not a silent
  substitution: the rate is 0.000, which answers what SPEC.md asked
  ("did the two arms' best-found architecture ID match") directly, just not
  through the TOST/Tango pathway the issue anticipated.

## Interpretation

This establishes what the issue asked for and nothing more. `RandomSearch`
and `IdentityRestartScaffold` -- the first `BaseSearcher`/`Scaffold`
implementations wired to a real external benchmark rather than a synthetic
or in-house problem -- produce the verdict the audit's statistical
machinery is supposed to produce for a scaffold that is null by
construction, at a budget the feasibility probe confirmed could resolve an
effect at the pre-registered margin if one existed. The `RandomSearch`
seeding bug this issue was specifically checking for (a fixed/unthreaded
random state collapsing independent restarts to identical output --
this project's own founding incident) does not reproduce here: the
degeneracy pre-check found genuine variety on every single run.

This does **not** establish that NAS controllers can or cannot beat
budget-matched random search (README item 3) -- no controller was audited
here, only `RandomSearch` against itself. It also does not establish
anything about the `sss` (size) search space, other datasets, other hp
settings, or other budgets; nullity is indexed by budget and by problem
per `AUDIT_METHODOLOGY.md` §4.1, and this run covers exactly one point in
that space. The paired-binary "exact architecture match" metric SPEC.md
declared was not run through the audit's Holm-corrected margin pathway, for
the structural reason given above -- this is a genuine gap between what the
issue anticipated and what `engine.audit.arms.audit()` currently supports
for a between-arm comparison metric, worth a researcher-filed issue of its
own if that shape is wanted generally, but out of this branch's scope to
add unilaterally.

## Decision

**MERGE.** All nine `GIT_WORKFLOW.md` merge criteria are met:

- **Scientific relevance:** directly answers README item 2's explicit
  next step and unblocks item 3.
- **Correctness:** implementation matches SPEC.md's design; no shortcuts
  taken (real 1.1GB NATS-Bench data, not a mock, for the actual sweep --
  `tests/test_plugin_nas_search.py` uses a fake only for the unit suite,
  documented as such).
- **Experimental validity:** budget-matched control (not "no scaffold"),
  feasibility checked before the sweep, margin pre-registered before the
  sweep ran.
- **Reproducibility:** `results/nas_search_self_audit/audit.json` carries
  full config (dataset, hp, budget, restarts, seeds, margin); re-running
  `plugins.nas_search.run_self_audit` with the same NATS-Bench file and the
  fixed seeds 0-29 is deterministic (`np.random.default_rng(seed)`,
  `is_random=False` lookups).
- **Documentation:** this file plus module-level docstrings throughout
  `plugins/nas_search/`.
- **Interpretation:** stated above, including what this run does not
  establish.
- **Research integrity:** the paired-binary metric gap is recorded here
  rather than glossed over or silently dropped.
- **Integration:** `plugins/nas_search/` is additive; `register()` is a
  no-op without `NATS_BENCH_TSS_SIMPLE_PATH` set, so it changes nothing for
  anyone who has not opted in (CI included -- the `nas` extra and this env
  var are both absent there by design).
- **Evidence:** a real 30-seed sweep against real benchmark data, not a
  simulated or mocked result.

Follow-up, for the researcher session, not this branch: file the
README-item-3 issue (real NAS controller vs. budget-matched random search)
once the `naslib` blocker is resolved -- this branch's `RandomSearch` is
its control arm. Also worth a researcher's judgment: whether
`engine.audit.arms.audit()` should grow a metric shape for between-arm
paired-binary comparisons (the gap noted above), as a `method/` issue.
