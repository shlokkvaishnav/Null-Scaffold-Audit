# RFC-0001: The null-scaffold audit

| | |
|---|---|
| Number | `RFC-0001` |
| Title | The null-scaffold audit |
| Author | Shlok Vaishnav |
| Status | Draft |
| Created | 2026-08-09 |
| Supersedes | — |
| Superseded by | — |
| Resulting ADR | [`ADR-0001`](../adr/ADR-0001-null-scaffold-audit-is-advisory.md) |

---

## 1. Summary

This RFC proposes that the platform be able to answer, mechanically, one
question about any discovery pipeline: **does the scaffolding around the search
contribute anything that the search alone would not have produced at the same
compute?**

It defines when a scaffold is *null*, specifies a statistical procedure that can
support that conclusion rather than merely fail to refute it, and proposes the
engine-side interfaces required to run the procedure on any method the platform
can execute. It does not propose making the audit mandatory; that is a separate
decision, and §10 explains why it is separated.

---

## 2. Motivation

A discovery method is usually published as a whole: a search procedure wrapped
in scaffolding — an iterative refine loop, a belief state, a hypothesis
archive, a language model proposing candidates. The reported result is for the
whole. Almost nothing in the current practice of the field establishes that the
wrapper contributed to the number.

This is not hypothetical, and the clearest example available to us is our own.
The agent in `physics_discovery` runs three hypothesis-proposal rounds per
problem. Each round constructs a fresh generator with the same fixed random
state, and the belief state that the rounds exist to accumulate is passed to
the generator but never consumed during generation. The recorded consequence,
in `results/feynman_benchmark/results.csv`, is that the agent returns a
candidate equation character-identical to the plain symbolic regressor on all
eight benchmark problems, having spent 266.78s against the regressor's 79.72s —
roughly 3.3× the compute for provably nothing.

Nobody was dishonest. The system had no mechanism that would have noticed, and
the README still asserts a capability that the repository's own results file
refutes.

The wider literature has begun to notice the general pattern. A 2026 study
across eight domains and roughly 25,000 agent runs attributed 41.4% of
explained variance to the base model and 1.5% to the scaffold. Controlled
comparisons on GAIA report that light scaffolds reduce variance without
materially improving outcomes. Work decomposing second-pass gains in multi-LLM
pipelines finds much of the improvement is re-solving rather than revision.

Three properties are common to all of that work, and they are the actual gap:

1. **It is retrospective.** The audit happens after publication, by people who
   were not the proposers.
2. **It is manual.** Each study builds its own comparison; none ships a
   procedure the next method can be run through.
3. **It is voluntary.** No venue, benchmark, or platform requires it, so it is
   performed by skeptics and never by proposers.

Meanwhile the symbolic regression community has asked for exactly this
capability in writing. The GECCO 2025 *Call for Action* on the next generation
of SR benchmarks names standardized execution constraints and computational
resource allocation as open problems, and SRBench++ does not close them.

**Cost of doing nothing:** the platform can already run many methods under
identical conditions, which is most of the work. Without this, it produces
comparable leaderboard numbers while remaining unable to say whether any given
entry's contribution is real — which is the question the numbers exist to
answer.

---

## 3. Guide-level explanation

A researcher who has built a scaffolded method runs:

```bash
sde audit null-scaffold --pipeline my_method --problems feynman --seeds 30
```

The platform runs two arms at **matched compute**:

- **Treatment:** the pipeline as submitted.
- **Control:** the pipeline's own base searcher, stripped of the scaffold, given
  the same compute budget — spent on independent restarts.

The control is deliberately not "the base searcher run once." A scaffold that
costs 3× and is compared against a single base run is being credited for the
compute it consumed. The honest control is the base searcher given the same
budget to use as it likes.

The output is a verdict with an interval:

```
Pipeline:  my_method  (scaffold=IterativeRefine, base=GeneticSearch)
Budget:    120,000 candidate evaluations per arm, 30 seeds

  best_loss        NULL          90% CI of difference [-0.004, +0.007], margin ±0.02
  complexity       NULL          90% CI of difference [-1.1, +0.8],     margin ±3
  exact_recovery   NULL          0/30 vs 0/30

  Identical-output rate: 30/30 runs returned an expression symbolically
                         equivalent to the control's.

VERDICT: NULL at the stated margins. The scaffold did not contribute at this
         budget. Statistical power at margin: 0.91.
```

Three verdicts are possible — `CONTRIBUTES`, `NULL`, `HARMFUL` — plus
`INCONCLUSIVE` when the design lacked power to distinguish them.

`INCONCLUSIVE` is a first-class outcome and is never reported as `NULL`.
Conflating "we could not tell" with "there was no effect" would reproduce, in
the audit itself, the exact error the audit exists to catch.

---

## 4. Reference-level explanation

### 4.1 The definition

A pipeline is a pair `(S, B)`: a scaffold `S` wrapping a base searcher `B`.

Let `P` be a problem instance, `c` a compute budget, and `Θ` a set of seeds. Let
`M` be an outcome metric with a pre-registered practical-equivalence margin `δ`.

> **S is null on P at budget c with respect to M and δ** if the distribution of
> `M(S(B, P, c, θ))` over `θ ∈ Θ` is statistically equivalent, within `±δ`, to
> the distribution of `M(B_restart(P, c, θ))`.

`B_restart` is `B` run with independent restarts until the budget `c` is
exhausted, returning its best result under the pipeline's own selection rule.

Three properties of this definition matter and are deliberate:

**Budget is measured in base-searcher work, not wall-clock.** The unit is
candidate evaluations, with language-model tokens accounted separately where a
pipeline uses them. Wall-clock matching conflates search quality with
implementation efficiency, so a slow-but-good scaffold and a fast-but-empty one
become indistinguishable — which is the confusion the audit exists to remove.

**The margin `δ` is pre-registered, per metric.** An audit whose margin is
chosen after seeing the intervals is not an audit. `δ` is part of the
configuration and is recorded in the report.

**Nullity is indexed by budget.** A scaffold may be null at a large budget and
contribute at a small one, or the reverse. Reporting a single verdict without
its budget is meaningless, so the report always carries it.

### 4.2 The statistical procedure

**Equivalence testing, not significance testing.** `p > 0.05` on a difference
test does not license concluding there is no difference; it licenses concluding
nothing. This is the single most important design decision in this RFC, and
inverting it would make the audit worse than useless — it would let any
underpowered comparison certify a scaffold as null.

The procedure is therefore **two one-sided tests (TOST)** against the
pre-registered margin `δ`, on the paired-by-seed difference between arms:

- Reject `H₀: difference ≤ −δ` and reject `H₀: difference ≥ +δ` → **NULL**.
- One-sided superiority beyond `+δ` → **CONTRIBUTES**.
- One-sided inferiority beyond `−δ` → **HARMFUL**.
- Neither established → **INCONCLUSIVE**, reported with the achieved power.

Metrics are non-normal and heavy-tailed in this setting, so intervals are
bootstrap (BCa) rather than parametric. Multiple metrics are corrected across
(Holm), and the correction is stated in the report rather than left implicit.

**Metrics that record a success rather than a measurement are tested
differently.** A recovery rate is 0 or 1 on every seed, so its paired difference
takes only the values −1, 0 and +1. When no seed disagrees between the arms
every bootstrap resample is identical, the interval collapses to a single point,
and a point lies inside any margin — so the procedure above returns `NULL`, this
audit's one positive verdict, from a sample that observed no disagreement at
all. Simulated at a true difference of 0.11 against a 0.10 margin with a sparse
discordant rate, that certifies `NULL` about 10% of the time, against the 5% a
TOST is meant to hold.

Such metrics are therefore declared in advance and tested with a score interval
for the paired difference of proportions (Tango), which reduces to McNemar's
statistic at a difference of zero. Declared, not inferred: a continuous metric
that happened to come back all-zeros on one sweep is still continuous, and a
rule that sniffed the values would silently switch tests between sweeps of the
same design. `exact_recovery` is the only metric currently so declared.

The pre-registered margin does not move — this changes the test, not the bar.
Its effect on `NULL` is one-directional: it can withdraw an equivalence claim
the collapsed interval manufactured, and cannot manufacture one itself. It is
*not* uniformly more conservative, and should not be described as such — on a
table with few discordant pairs the score interval is narrower than the
bootstrap's, so it can also sharpen an `INCONCLUSIVE` into `HARMFUL` or
`CONTRIBUTES`. That is the correct behaviour for a better-calibrated interval,
but it means the change can move a verdict in either direction and each such
move has to be read on its own.

Each verdict records which procedure produced it, so two metrics tested unalike
cannot be read as though they were tested alike.

### 4.3 The degeneracy pre-check

Before the statistical procedure runs at all, a cheap structural check:

**Intra-run redundancy.** Within a single scaffold invocation, are the `k`
proposals identical to one another? A scaffold whose iterations produce the same
candidate is not exploring — it is repeating, and it will be null by
construction.

This costs one run to detect and would have caught our own defect immediately.
It is reported separately from the statistical verdict because it identifies a
*mechanism*, not merely an outcome: `NULL` says the scaffold did not help;
`DEGENERATE` says why.

### 4.4 Interfaces

These live in `engine/` and name no scientific concept, no algorithm, and no
domain. The audit does not know what an equation is; it compares distributions
of outcomes from two arms it was handed.

```python
class BaseSearcher(Protocol):
    """A search primitive that consumes a compute budget and returns candidates."""

    def search(self, problem: Problem, budget: Budget, seed: int) -> SearchResult: ...


class Scaffold(Protocol):
    """Logic wrapped around a base searcher.

    A scaffold must be separable from its base searcher, because the audit's
    control arm is constructed by removing it. A pipeline that cannot expose
    its base searcher cannot be audited -- and that is itself a reportable
    finding, not an exemption.
    """

    def run(self, base: BaseSearcher, problem: Problem, budget: Budget, seed: int) -> SearchResult: ...

    def unwrap(self) -> BaseSearcher: ...


@dataclass(frozen=True)
class AuditReport:
    verdict: Verdict                      # CONTRIBUTES | NULL | HARMFUL | INCONCLUSIVE
    per_metric: Mapping[str, MetricVerdict]
    budget: Budget                        # matched, per arm, in base-searcher units
    seeds: int
    power: Mapping[str, float]
    degenerate: DegeneracyReport
    provenance: Provenance                # config, git SHA, environment, hardware
```

`Verdict` is a closed enumeration so that reports are machine-comparable across
methods and across time.

---

## 5. Constitutional review

| Article | Question | Assessment |
|---|---|---|
| 5 — Domain independence | Does this put scientific content in `engine/`? | No. The audit compares outcome distributions from two arms; it never inspects a hypothesis's content. `Problem` and `SearchResult` are opaque to it. Checked by `tools/check_domain_independence.py`. |
| 6 — Algorithm independence | Does this couple the engine to an algorithm? | No, but it does impose a **structural requirement**: a pipeline must expose its base searcher via `unwrap()`. This is a real constraint on plugin authors and is the main cost of this proposal — see §7. |
| 9 — Dependency direction | Which way do dependencies flow? | Engine defines `Scaffold` and `BaseSearcher`; plugins implement them. No new inbound dependency. |
| 13 — Evidence | What claims will this make? | Every verdict is a generated artifact carrying its config, seeds, budget, git SHA, and environment. The audit is itself subject to the rule it enforces. |
| 15 — Scope | Does this add a special case to the core? | No. It adds a capability behind the existing plugin contract. It does, however, *extend* that contract, which is a change requiring its own review. |

The uncomfortable answer is Article 6's. `unwrap()` is a genuine constraint, and
a pipeline whose scaffold is not cleanly separable from its searcher cannot
satisfy it. That is argued in §7 rather than minimised here.

---

## 6. Alternatives considered

### Alternative A — Retrospective manual ablation (the status quo)

**What it is:** Each paper performs its own ablation; skeptics re-audit later.

**In its favour:** No platform work. Maximum flexibility — an author can design
an ablation suited to their method, which a generic procedure cannot. The
existing literature was produced this way and it did find real problems.

**Why it loses:** It is voluntary, and the incentive structure of a voluntary
check is that it is performed by people hoping to find nothing. Retrospective
audits arrive years after the result has propagated. Nothing accumulates: each
study rebuilds the comparison from scratch, so the next method is audited only
if someone volunteers again.

### Alternative B — Wall-clock matched comparison

**What it is:** Give both arms equal wall-clock time.

**In its favour:** Trivial to implement, needs no `unwrap()`, and is arguably
what a practitioner actually cares about — if the scaffold wins in the time
available, it wins.

**Why it loses:** It measures implementation quality, not scaffold
contribution. A scaffold written in optimised C would "contribute" over the
same scaffold in Python. Worse, it is gameable in the wrong direction: slowing
the control arm's implementation improves the treatment's verdict. The
practitioner's question is legitimate and should be answered — but as a
*separate*, clearly labelled efficiency comparison, not as this one.

### Alternative C — Report base-searcher performance as one more baseline row

**What it is:** Add the unwrapped searcher to the standard results table.

**In its favour:** Nearly free, fits existing reporting, no new subsystem, and
gets most of the informational value in front of a reader.

**Why it loses:** It permits the inference the audit exists to prevent. A table
showing the scaffold marginally ahead invites "our method improves on the base
searcher" with no equivalence testing, no power analysis, and no matched
budget. Being in the table is not the same as being tested, and the gap between
them is where the error lives.

### Alternative D — Do nothing

**In its favour:** Milestone 0 is complete; the engine still has an Article 5
violation and an unmigrated legacy tree. This is not obviously the most urgent
work, and the cost of building it is real.

**Why it loses:** The platform's entire claim is that it makes results
comparable. Comparability without attribution produces a leaderboard whose
entries cannot be interpreted, which is a more sophisticated version of the
problem `VISION.md` §1 describes rather than a solution to it.

---

## 7. Drawbacks

**`unwrap()` is a real constraint on plugin authors.** Some methods genuinely do
not decompose into scaffold-plus-searcher — an end-to-end trained model has no
inner searcher to strip. Those pipelines cannot be audited this way. The
proposal's honest position is that such a pipeline reports `NOT_SEPARABLE`
rather than being silently exempted, but that is a weaker guarantee than it
first appears, and a determined author could hide a scaffold's work inside a
"base" searcher to obtain a favourable verdict.

**Compute cost.** A 30-seed two-arm audit at matched budget costs roughly twice
a single evaluation run, per problem. For expensive methods this is the
dominant cost of using the platform.

**The margin is a judgement call.** `δ` is pre-registered, which prevents
post-hoc gaming, but choosing it well requires domain knowledge the engine does
not have. A poorly chosen margin produces confident nonsense in either
direction.

**Verdicts will be socially unwelcome.** A tool whose output is "your
contribution is null" will meet resistance, including from us. The first method
it should be run against is our own, and §2 already commits to the result.

---

## 8. Prior art

The genre this belongs to is the reality-check paper: *Are we really making much
progress?* on recommender systems, *A metric learning reality check*, and *On
the state of the art of evaluation in neural language models*. Each found that
reported gains largely vanished under controlled comparison. Each is heavily
cited. **None of them shipped a mechanism** — they shipped a finding, and the
field's practice reverted.

The recent agent-scaffolding audits cited in §2 are the same shape, applied to a
newer subject.

Equivalence testing (TOST) is standard in bioequivalence and increasingly in
psychology, where the "absence of evidence is not evidence of absence" problem
was confronted earlier and more directly than in machine learning. The
statistical machinery is mature and borrowed wholesale; the contribution here is
not the test but its placement — inside the platform, on the path every method
takes, rather than in a paper about methods someone else built.

Where the analogy breaks down: bioequivalence has regulators. We have a CI job,
which is weaker, and the difference should not be overstated.

---

## 9. Impact

**Affected subsystems:** `engine/` (new protocols), `algorithms/`, `benchmarks/`,
`reports/`, `cli/`.

**Interface changes:** Additive but contract-extending. Existing plugins
continue to work and report `NOT_SEPARABLE` until they implement `unwrap()`.

**Migration required:** Each algorithm plugin that wraps a searcher must expose
it. The legacy `physics_discovery` agent is the first candidate and already
has a known verdict.

**Documentation to update:** `docs/design/` gains an audit design document;
`CONTRIBUTING.md` gains the audit to the local gate list if it becomes required.

**Governing documents to amend:** None if the audit is advisory. If it becomes
a merge gate, `BOOTSTRAP.md` §14 gains an item, and that amendment belongs in
the deciding ADR.

---

## 10. Unresolved questions

**Is the audit advisory or mandatory?** Deliberately left open. Mandatory is the
version with teeth and the version that makes the paper's argument; advisory is
the version that does not deter the external plugin authors success criterion 1
depends on. This should be settled in the ADR, on evidence about adoption cost,
not assumed here.

**How is `δ` chosen for a new metric?** Currently: by the plugin author,
pre-registered. Whether the platform should supply defaults per metric family
is unresolved.

**What is the minimum seed count?** Power depends on the metric's variance,
which is not known before running. A sequential design that stops when the
verdict is determined would be more efficient than a fixed 30 and is not
specified here.

**Does the control arm's restart policy need to be per-plugin?** "Independent
restarts, best-of" is one selection rule among several, and a scaffold could be
disadvantaged by a badly chosen one — which would make the audit unfair in the
opposite direction to the one it guards against.

---

## 11. Future possibilities

Out of scope now, enabled if this holds: auditing scaffold contribution as a
*function* of budget rather than at one point; attributing contribution across
components of a multi-part scaffold; and publishing verdicts alongside
leaderboard entries so that a method's reported score and its audited
contribution are read together.
