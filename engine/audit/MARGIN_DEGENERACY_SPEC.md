<!--
Copied verbatim from GitHub issue #27, per research/GIT_WORKFLOW.md and
research/SPEC_TEMPLATE.md. Section headers below follow SPEC_TEMPLATE.md;
their content is the issue body's corresponding section, unedited.

Named MARGIN_DEGENERACY_SPEC.md, not SPEC.md: this branch shares
engine/audit/ with issue #23's SPEC.md, which may not be overwritten --
per GIT_WORKFLOW.md's "a completed branch's pre-registered result stands."
-->

# Spec: method/audit-margin-degeneracy

**Branch:** `method/audit-margin-degeneracy`
**Date opened:** 2026-08-27
**Status:** IN PROGRESS
**Issue:** https://github.com/shlokkvaishnav/Null-Scaffold-Audit/issues/27

## Branch type

method/ -- a new methodological component (a diagnostic check in
`engine/audit/`, not a new scaffold or domain claim)

## Research question

When a metric's pre-registered margin (`MARGIN_FRACTION * control_spread`)
is computed from a control arm whose spread is itself near-zero -- e.g.
basin-hopping restarts on Griewank converging to (numerically) the same
near-optimal point on every seed -- the margin floors at an arbitrary
numerical constant (currently `1e-9` in `plugins/basinhopping_audit`'s own
scripts) with no domain meaning. This is not hypothetical: it already
produced two different, mutually untrustworthy verdicts (`INCONCLUSIVE` in
PR #16, `HARMFUL` in issue #25's re-run) for the same function, same `n`,
same design, differing only in which random seeds were drawn -- caught
only by a human noticing the discrepancy and writing it up in
`DECISION_LOG.md` after the fact.

**Question:** should `engine/audit/` detect this condition directly --
margin sitting at or near its numerical floor relative to the metric's own
observed scale -- and report it as an explicit diagnostic flag (parallel
to `DegeneracyReport`'s existing intra-run-redundancy check, per
`AUDIT_METHODOLOGY.md` §4.3: "it identifies a *mechanism*, not merely an
outcome") rather than leaving a degenerate-margin verdict indistinguishable
from a trustworthy one in `AuditReport`?

## Hypothesis

A margin-degeneracy check -- comparing the pre-registered margin against a
small multiple of the control arm's own observed spread, or against the
numerical floor constant itself -- can be computed purely from data already
present in `AuditReport` (per-metric margin, control-arm spread), without
any domain knowledge, so it belongs in `engine/audit/` per Article 5
(domain independence), the same way `DegeneracyReport` already does for
intra-run redundancy. Retrofitted against this project's own history, it
should fire on exactly the row already known to be numerically meaningless
(Griewank, `run_audit.py`'s unscaled `stepsize=0.5` configuration, both
readings) and not on rows already trusted (Rastrigin and Ackley across all
basinhopping scripts; Griewank's domain-scaled-`stepsize` `NULL` result in
`run_stepsize_experiment.py`).

## Null / alternative hypothesis

**Null (would contradict the hypothesis):** the margin-degeneracy condition
cannot be distinguished from a legitimately tiny, meaningful margin using
only data already in `AuditReport` -- either it needs domain knowledge the
engine is deliberately blind to (Article 5), or a threshold that separates
the known-bad Griewank row from the known-good rows also misfires on at
least one already-trusted row. This would mean margin degeneracy has to
stay a plugin-level judgment call, not an engine mechanism.

**Alternative (supports the hypothesis):** a threshold exists, computable
from `AuditReport` data alone, that fires exactly on Griewank's
unscaled-stepsize `run_audit.py` row (both the PR #16 and issue #25
readings) and not on any other already-published row in
`plugins/basinhopping_audit/`.

## Motivation

Directly motivated by issue #25 / PR #26 (just merged): the same margin
formula produced two different, mutually untrustworthy verdicts for
Griewank because the margin had floored at a practically meaningless
`1e-9`, and nothing in `AuditReport` distinguished that row from an
ordinary, trustworthy one -- it took a human running two audits
side-by-side to even notice. `AUDIT_METHODOLOGY.md` §4.3 already commits
to catching exactly this class of defect *mechanically* for intra-run
redundancy ("this costs one run to detect and would have caught our own
defect immediately"); margin degeneracy is the same shape of problem
without the same mechanical catch yet. Without it, a single, unaccompanied
future audit run (no side-by-side comparison to prompt suspicion) could
report a margin-degenerate `HARMFUL`/`INCONCLUSIVE` verdict as an ordinary
finding.

## Experimental design

No new scaffold or scientific claim -- entirely retrospective against data
this project has already collected. Add a check to `engine/audit/`
(alongside or parallel to `degeneracy.py`) that flags when a per-metric
margin sits at or near its numerical floor relative to the control arm's
own spread. Validate it against every already-committed
`plugins/basinhopping_audit/` result across PRs #16, #18, #20, #22, and
#26 (`audit.json`/`audit.csv` artifacts already in the repo -- no new
searcher/scaffold runs needed for validation) -- the check should fire on
exactly the row already known to be untrustworthy for this reason and not
on the others.

**Held constant:** the statistical procedure's actual verdict-resolution
logic (`engine/audit/statistics.py`'s `_resolve`) is unchanged by this
branch -- this adds a diagnostic flag alongside the verdict, the same
relationship `DegeneracyReport.degenerate` already has to `verdict`, not a
change to what the verdict itself asserts.

## Metrics

A new boolean-or-similar field, structurally analogous to
`DegeneracyReport.degenerate`, computed retrospectively against already-
collected `AuditReport`/per-metric data. The exact threshold (margin vs.
its numerical floor constant, or margin vs. a small multiple of control
spread) is itself part of this branch's design work, not assumed in
advance.

## Baselines / controls

Not a new scaffold-vs-searcher comparison. The "control" is retrospective:
every already-published basinhopping row is the test set, with known-good
(Rastrigin, Ackley, Griewank domain-scaled `NULL`) and known-bad (Griewank
unscaled-stepsize, both the PR #16 and issue #25 readings) labels already
established by this project's own history and `DECISION_LOG.md`.

## Expected outcomes

- **A threshold cleanly separates the known-good/known-bad rows**: ship it
  in `engine/audit/`, retrofit `plugins/basinhopping_audit`'s scripts to
  surface it, record as new shared infrastructure in `DECISION_LOG.md`.
- **No domain-independent threshold works** (false positives on a
  known-good row, or requires domain knowledge to set correctly): report
  as a negative result -- margin degeneracy stays a plugin-level manual
  judgment call, and the existing per-incident `DECISION_LOG.md` write-up
  approach (already used correctly twice) is confirmed as the right level
  for this problem, not a gap to keep trying to close.
- **A middle case**: an engine-level flag that needs occasional
  plugin-specific threshold tuning -- ship it, but say so explicitly
  rather than presenting it as fully domain-independent.

## Interpretation plan

- Clean separation: closes a real, now-twice-hit gap between what
  `engine/audit/` mechanically catches and what still requires a human
  eyeballing a `DECISION_LOG.md` entry after the fact.
- No domain-independent threshold: a genuinely valuable negative result --
  establishes the existing manual process as correct, not deficient.
- Middle case: ship with the caveat stated plainly, matching this
  project's existing standard for partial results (e.g. the shared-helper
  refactor decision in issue #25/PR #26).

## Confounds considered

The known-bad label set is small (two readings of the same underlying
Griewank/`run_audit.py`/`stepsize=0.5` row) -- any threshold validated
against it risks overfitting to this one function/configuration; state
that limitation explicitly rather than claiming general validation from
effectively n=1 row. Adding a new field to `AuditReport`/`MetricVerdict`
must not change any already-computed verdict for already-published rows --
verify by test that re-computing every existing committed row's `verdict`
field is byte-identical after this branch; only the new diagnostic field
should be additive.

## Results

**Outcome: the Alternative hypothesis, cleanly.** A first candidate design
(comparing the pre-registered margin, or its numerical floor, against an
absolute epsilon) was considered and rejected before implementation: it
cannot be domain-independent, since "tiny in absolute terms" depends on the
metric's own scale, which the engine is deliberately blind to (Article 5).

The design actually implemented instead compares **control-arm spread to
paired treatment-arm spread on the same metric, problem, and seeds** --
data `arms.audit()` already has, independent of whatever margin formula or
floor the plugin used. Retrospectively computed from every committed
`plugins/basinhopping_audit/` `audit.json`'s raw per-seed arm values
(`tests/test_audit_margin_degeneracy.py::test_margin_degeneracy_recomputed_from_committed_artifacts`,
12 rows, no new searcher/scaffold runs):

| Row | control:treatment ratio | Trust label (DECISION_LOG.md) |
|---|---|---|
| `run_audit.py` Griewank, PR #16 | ~4.86e-10 | untrusted |
| `run_audit.py` Griewank, issue #25 | ~4.10e-10 | untrusted |
| `run_stepsize_experiment.py` Griewank, PR #18 | ~0.228 | trusted |
| `run_stepsize_experiment.py` Griewank, issue #25 | ~0.881 | trusted |
| every Rastrigin/Ackley row (8 total, both scripts, both readings) | ~0.33 to ~2.09 | trusted |

A threshold of `1e-4` (`DEFAULT_RATIO_THRESHOLD`) sits inside an eight
order-of-magnitude gap between the untrusted cluster (~4-5e-10) and the
trusted cluster's minimum (~0.228) -- not tuned to an edge.

**The clean separation was not the obvious outcome going in.** Griewank's
domain-scaled-stepsize control spread is *also* near floating-point-noise
scale in absolute terms (~1e-10 to 1e-11, the same order of magnitude as
the untrusted rows' control spreads) -- this project trusted that reading
only because it was independently corroborated by issue #17's separate
finding (a larger stepsize gives basin-hopping more room to move), not
because anything about the row looked different from the untrusted ones on
an absolute scale. An absolute-floor check would have flagged all four
Griewank rows identically, unable to reproduce this project's own
trust/untrust split -- which is exactly the SPEC's stated Null hypothesis,
and is why the relative (control-vs-treatment) design was tried instead of
an absolute one, and why it, not the more obvious absolute design, is what
shipped.

Implementation: `engine/audit/margin_degeneracy.py`
(`MarginDegeneracyReport`, `assess_margin_degeneracy`), wired into
`arms.py`'s `audit()` per-metric loop via `dataclasses.replace` after
`equivalence_verdict()`, surfaced as `MetricVerdict.margin_degeneracy`.
Held constant, verified: all 148 pre-existing `engine/audit` tests pass
unchanged, `tools/check_domain_independence.py` passes, `ruff`/`mypy`
clean -- no already-published verdict changed.

## Interpretation

Closes the gap this branch's Motivation names: a margin-degenerate row
(Griewank's `run_audit.py` configuration) is now mechanically flagged the
moment it's audited, the same way `DegeneracyReport` already flags
intra-run redundancy -- not only discoverable by a human comparing two runs
side by side after the fact, as issue #25's incident required.

**What this does NOT establish.** The flag is diagnostic, not
verdict-gating: `_resolve`'s logic is untouched, so a margin-degenerate row
still reports whatever `NULL`/`HARMFUL`/`INCONCLUSIVE` the noise happened to
produce, now simply labeled. Whether it *should* gate (e.g. force
`INCONCLUSIVE`, or withdraw the verdict the way `_guard_vacuous_comparison`
already does for a structurally different failure) is a separate,
deliberately unanswered question -- this branch answers "can it be
detected," not "what should happen once it is." The known-bad label set
remains small (two readings of one function/configuration, per the
Confounds section above); the threshold's eight-order-of-magnitude margin
of safety mitigates but does not eliminate the overfitting risk that
smallness carries.

Does not change the project's research thesis ("Does the Scaffold Earn Its
Keep") -- pure audit-infrastructure hardening, no new scaffold-contribution
claim.

## Decision (implementer's self-assessment)

- [x] MERGE
- [ ] ARCHIVE
- [ ] REVISE
- [ ] ABANDON
- [ ] REPRODUCE

Why: all nine `GIT_WORKFLOW.md` criteria are met. Scientific relevance:
closes a real, twice-hit gap directly motivated by issue #25/PR #26.
Correctness: retrospectively validated against 12 real rows, not
synthetic-only. Experimental validity: the candidate design that would have
been simpler (absolute-floor) was tried in reasoning first and explicitly
rejected as not domain-independent, rather than skipped past. Reproducibility:
threshold and validation table are in this file; `tests/test_audit_margin_degeneracy.py`
re-derives the same numbers from the same committed artifacts on every run.
Documentation: `AUDIT_METHODOLOGY.md` §4.3a. Interpretation: the
"diagnostic, not gating" boundary is stated explicitly, not implied.
Research integrity: the near-miss (an absolute check would not have worked)
is reported, not hidden behind the design that did. Integration: matches
`DegeneracyReport`'s existing pattern exactly (frozen dataclass report,
`assessed`/`degenerate` shape, wired via `dataclasses.replace`). Evidence:
12/12 retrospective rows correctly classified, 0 already-published verdicts
changed.
