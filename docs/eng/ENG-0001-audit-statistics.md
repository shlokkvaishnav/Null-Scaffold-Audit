# ENG-0001: Equivalence statistics for the null-scaffold audit

| | |
|---|---|
| ID | `ENG-0001` |
| Title | Equivalence statistics for the null-scaffold audit |
| Originating RFC | [`RFC-0001`](../rfc/RFC-0001-null-scaffold-audit.md) |
| Governing ADR | [`ADR-0001`](../adr/ADR-0001-null-scaffold-audit-is-advisory.md) |
| Owner | Shlok Vaishnav |
| Status | Ready |
| Blocked by | — |

---

## 1. Objective

Given paired per-seed outcomes from a treatment arm and a control arm, return a
verdict that distinguishes "the scaffold contributed nothing" from "we could not
tell" — with the interval and the achieved power that justify it.

---

## 2. Scope

**In scope:** The pure statistical core. Two paired samples and a margin in;
a `MetricVerdict` out. No I/O, no configuration loading, no searcher execution.

**Explicitly out of scope, tracked separately:**

- Running the two arms and matching their budgets (ENG-0002).
- The `Scaffold` / `BaseSearcher` protocols and `unwrap()` (ENG-0002).
- The intra-run degeneracy pre-check of RFC-0001 §4.3 (ENG-0003).
- Report rendering and the results-schema verdict field (ENG-0004).

This split exists because the statistics are the part that must be *correct*
rather than merely working, and they are the only part testable against known
ground truth without executing a search. A defect here produces confident,
plausible, wrong verdicts — the failure mode RFC-0001 exists to prevent,
reproduced inside the tool that prevents it.

---

## 3. Files

| Path | Change |
|---|---|
| `engine/audit/__init__.py` | New. Re-exports `Verdict`, `MetricVerdict`, `equivalence_verdict`. |
| `engine/audit/verdict.py` | New. The closed `Verdict` enumeration and `MetricVerdict` record. |
| `engine/audit/statistics.py` | New. TOST, BCa bootstrap interval, achieved power, verdict resolution. |
| `tests/test_audit_statistics.py` | New. Tests named in §6. |

---

## 4. Interfaces

```python
class Verdict(StrEnum):
    CONTRIBUTES = "CONTRIBUTES"
    NULL = "NULL"
    HARMFUL = "HARMFUL"
    INCONCLUSIVE = "INCONCLUSIVE"
    NOT_SEPARABLE = "NOT_SEPARABLE"


@dataclass(frozen=True)
class MetricVerdict:
    metric: str
    verdict: Verdict
    observed_difference: float   # mean(treatment) - mean(control)
    ci_low: float                # bootstrap BCa bounds at `confidence`
    ci_high: float
    margin: float                # the pre-registered delta
    power: float                 # achieved power against `margin`, in [0, 1]
    n: int                       # number of paired seeds
    higher_is_better: bool


def equivalence_verdict(
    treatment: Sequence[float],
    control: Sequence[float],
    *,
    metric: str,
    margin: float,
    higher_is_better: bool = False,
    confidence: float = 0.90,
    resamples: int = 10_000,
    seed: int = 0,
) -> MetricVerdict: ...
```

**Error behavior, specified rather than left to discover:**

| Condition | Behavior |
|---|---|
| `len(treatment) != len(control)` | `ValueError`. Samples are paired by seed; unequal lengths mean the pairing is already wrong. |
| Fewer than 2 pairs | `ValueError`. A bootstrap over one pair is not a wide interval, it is a meaningless one. |
| `margin <= 0` | `ValueError`. A non-positive margin makes equivalence unprovable and would silently yield permanent `INCONCLUSIVE`. |
| `confidence` outside `(0, 1)` | `ValueError`. |
| Any non-finite input value | `ValueError`, naming the offending arm and index. |
| Zero variance in both arms, identical values | Valid. Difference is exactly 0; verdict is `NULL` when `margin > 0`. |
| Zero variance, differing constant values | Valid. BCa degenerates to a point interval; resolution follows §5. |

`equivalence_verdict` is a pure function. Given the same inputs and `seed` it
returns the same result, because a verdict that moves between runs cannot be
published next to a number (ADR-0001, Compliance).

---

## 5. Behavioral specification

Let `d = mean(treatment) - mean(control)`, oriented so that positive `d` always
means *treatment is better* (negate when `higher_is_better` is false).

| Case | Condition | Verdict |
|---|---|---|
| Equivalent | CI entirely within `(-margin, +margin)` | `NULL` |
| Superior | CI lower bound `> +margin` | `CONTRIBUTES` |
| Inferior | CI upper bound `< -margin` | `HARMFUL` |
| Undetermined | CI overlaps a boundary | `INCONCLUSIVE` |

`NOT_SEPARABLE` is never returned by this function. It is a property of a
pipeline, not of a sample, and is assigned by the caller in ENG-0002.

**Power** is the probability of correctly concluding equivalence at the observed
variance and `n`, computed against `margin`. It is reported on every verdict,
not only on `INCONCLUSIVE`, because a `NULL` from an underpowered design and a
`NULL` from a well-powered one are different claims and must be
distinguishable by a reader who sees only the record.

**Orientation must not be silently wrong.** For a loss metric, lower is better;
for a recovery rate, higher is. The same numbers with `higher_is_better`
flipped must produce `CONTRIBUTES` where they produced `HARMFUL`. This is
tested explicitly because it is the most plausible way for the module to be
confidently backwards.

---

## 6. Tests required

- [x] `test_identical_arms_are_null` — zero difference, positive margin → `NULL`.
- [x] `test_large_improvement_is_contributes` — treatment clearly better → `CONTRIBUTES`.
- [x] `test_large_regression_is_harmful` — treatment clearly worse → `HARMFUL`.
- [x] `test_small_sample_is_inconclusive_not_null` — a true zero effect with n=3 and high variance must not be certified `NULL`.
- [x] `test_well_powered_zero_effect_is_null` — the same zero effect at n=50 and low variance *is* `NULL`.
- [x] `test_orientation_flips_contributes_and_harmful` — same data, `higher_is_better` toggled, verdicts swap.
- [x] `test_observed_difference_is_positive_when_treatment_wins` — orientation of the reported difference.
- [x] `test_null_requires_ci_inside_margin` — CI overlapping the margin is `INCONCLUSIVE`, not `NULL`.
- [x] `test_power_reported_on_every_verdict` — power in `[0, 1]` for all four outcomes.
- [x] `test_underpowered_comparison_reports_low_power` — n=3 high-variance reports lower power than n=50 low-variance.
- [x] `test_deterministic_for_fixed_seed` — two identical calls return equal results.
- [x] `test_record_carries_the_evidence_for_its_claim` — metric, margin, n, orientation, and interval bracket the estimate.
- [x] `test_mismatched_lengths_raise` / `test_single_pair_raises` / `test_non_positive_margin_raises` / `test_non_finite_input_raises` / `test_non_finite_control_names_the_control_arm` / `test_confidence_out_of_range_raises`
- [x] `test_zero_variance_identical_arms` — degenerate but valid input does not raise.
- [x] `test_zero_variance_constant_offset_beyond_margin` — degenerate point interval outside the margin resolves.
- [x] `test_known_ground_truth_recovery` — effects of `0`, `-2*margin`, and `+2*margin` at n=50 yield `NULL`, `CONTRIBUTES`, `HARMFUL`.

**Coverage floor for `engine/audit/`:** 95% — **achieved: 100%** (29 tests).

> Names above are as implemented. Two differ from this task's first draft
> (`test_underpowered_null_has_low_power` became
> `test_underpowered_comparison_reports_low_power`; the ground-truth case was
> parameterised over all three effects rather than two), and four tests were
> added during implementation. Recorded here rather than left to drift, since
> §14 requires the governing documents to remain accurate.

---

## 7. Evidence required

| Claim | Regenerating command | Artifact |
|---|---|---|
| The audit distinguishes `NULL` from `INCONCLUSIVE` | `uv run pytest tests/test_audit_statistics.py` | test output |
| `engine/audit/` contains no scientific content | `uv run python tools/check_domain_independence.py --path engine` | exit code |

No benchmark numbers are produced by this task, and no claim about any
discovery method is made by it. Stated explicitly so a reviewer knows the
evidence table was considered rather than skipped.

---

## 8. Constraints

- `engine/audit/` acquires no scientific content, no domain name, and no plugin
  import (Article 5). The module must not know what a hypothesis or an equation
  is — it receives floats.
- `scipy` and `numpy` are already project dependencies; no new dependency is
  introduced, so no ADR is required (BOOTSTRAP.md §10).
- The function is pure and seeded. A verdict that varies between runs cannot be
  published beside a number.

---

## 9. Definition of done

- [ ] Implementation complete
- [ ] Tests pass and coverage floor is met
- [ ] Type checking passes
- [ ] Documentation written, including limitations
- [ ] Benchmarks recorded, where applicable — *not applicable; no performance claim*
- [ ] Every claim made about the feature is backed by a regenerable artifact
- [ ] Governing architecture documents remain accurate
- [ ] CI passes
- [ ] Review checklist passes

---

## 10. Notes for the implementer

BCa bootstrap is available via `scipy.stats.bootstrap(method="BCa")`. It fails
on degenerate zero-variance input, so that case is handled explicitly rather
than allowed to raise from inside SciPy — hence the two zero-variance rows in
§4.

The temptation to use a t-test and report `p > 0.05` as `NULL` will be strong
because it is one line. It is the specific error RFC-0001 §4.2 exists to
prevent, and a review that finds it will return the task.
