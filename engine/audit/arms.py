"""Budget-matched two-arm execution for the null-scaffold audit (RFC-0001 section 4).

The treatment arm is a wrapper as submitted. The control arm is that wrapper's
own inner primitive, stripped out and given the *same* compute -- spent on
independent restarts rather than on the wrapper's logic.

The control is deliberately not "the primitive run once". A wrapper that costs
3x and is compared against a single primitive run is being credited for the
compute it consumed rather than for what it did with it.

Budget is counted in the primitive's own work units, never wall-clock. Matching
on wall-clock would measure implementation efficiency: the same wrapper rewritten
in a faster language would appear to contribute more, and slowing the control's
implementation would improve the treatment's verdict. That is backwards, and it
is the failure mode RFC-0001 section 6 (Alternative B) rejects.

Nothing here inspects what a candidate *is*. Outcomes are floats and an opaque
representation string, so this module holds no subject-matter knowledge
(Article 5).
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from engine.audit.degeneracy import DegeneracyReport, assess_degeneracy
from engine.audit.statistics import equivalence_verdict
from engine.audit.verdict import MetricVerdict, Verdict

# A restart's seed must be reproducible and distinct from every other restart's
# and from the arm seed itself. Multiplying by a large prime before offsetting
# keeps successive arm seeds from overlapping into each other's restart ranges,
# which a naive `seed + r` would do the moment restarts exceed the seed stride.
_SEED_STRIDE = 1_000_003


class NotSeparableError(RuntimeError):
    """Raised when a pipeline cannot expose the primitive its control arm needs.

    This is a reportable finding, not an exemption: RFC-0001 section 7 is explicit
    that a pipeline which cannot be decomposed is recorded as ``NOT_SEPARABLE``
    rather than quietly skipped.
    """


@dataclass(frozen=True)
class Budget:
    """Compute allowance for one arm, in the primitive's own work units."""

    evaluations: int

    def __post_init__(self) -> None:
        if self.evaluations <= 0:
            raise ValueError(f"budget must be positive, got {self.evaluations}")


@dataclass(frozen=True)
class SearchOutcome:
    """What one arm produced on one seed.

    ``representation`` is opaque to the engine. It is compared only for equality
    between arms, to report the identical-output rate -- the engine never parses
    it, because doing so would require knowing what it means.
    """

    metrics: Mapping[str, float]
    evaluations_used: int
    representation: str | None = None
    # Every candidate the arm proposed internally, in order. Optional: a pipeline
    # that does not expose its internals is not penalised, it is simply reported
    # as unassessed for degeneracy (RFC-0001 section 4.3).
    intermediate_representations: tuple[str, ...] = ()


@runtime_checkable
class BaseSearcher(Protocol):
    """The primitive a wrapper wraps.

    ``restart_cost`` is what one independent restart consumes, in the same units
    as ``SearchOutcome.evaluations_used``. It is what makes the arms comparable,
    so a pipeline that cannot state it cannot be audited.
    """

    # Declared read-only so a pipeline may derive it. SymbolicRestartSearcher
    # computes it as population_size * generations rather than storing it, and a
    # settable-attribute declaration would reject that -- for no benefit, since
    # nothing here ever assigns to it.
    @property
    def restart_cost(self) -> int: ...

    def search(self, problem: Any, seed: int) -> SearchOutcome: ...

    def select(self, outcomes: Sequence[SearchOutcome]) -> SearchOutcome:
        """Pick the best of several restarts, using the *pipeline's own* rule.

        Supplied by the pipeline rather than fixed here on purpose: imposing a
        selection rule would let a badly chosen one disadvantage the control arm,
        making the audit unfair in the opposite direction to the one it guards
        against (RFC-0001 section 10).
        """
        ...


@runtime_checkable
class Scaffold(Protocol):
    """Logic wrapped around a base searcher."""

    name: str

    def run(self, problem: Any, seed: int) -> SearchOutcome: ...

    def unwrap(self) -> BaseSearcher:
        """Return the bare primitive. Raise ``NotSeparableError`` if there isn't one."""
        ...


@dataclass(frozen=True)
class ArmOutcomes:
    """Paired per-seed results from both arms, with the budget actually spent."""

    treatment: list[SearchOutcome]
    control: list[SearchOutcome]
    seeds: list[int]
    restarts_per_seed: int
    treatment_evaluations: int
    control_evaluations: int

    @property
    def identical_representation_rate(self) -> float:
        """Fraction of seeds where both arms returned the same representation.

        A blunt instrument -- it catches literal equality, not semantic
        equivalence, so it under-reports. It is included because when it *does*
        fire it identifies the mechanism, which no interval can.
        """
        if not self.seeds:
            return 0.0
        matches = sum(
            1
            for t, c in zip(self.treatment, self.control, strict=True)
            if t.representation is not None and t.representation == c.representation
        )
        return matches / len(self.seeds)


@dataclass(frozen=True)
class AuditReport:
    """The verdict, and the evidence a reader needs to disbelieve it."""

    scaffold: str
    verdict: Verdict
    per_metric: dict[str, MetricVerdict]
    arms: ArmOutcomes
    # Reported beside the verdict, never folded into it. NULL says the wrapper did
    # not help; DEGENERATE says why, and the two are different findings.
    degeneracy: DegeneracyReport = field(default_factory=lambda: DegeneracyReport(assessed=False))
    limitations: list[str] = field(default_factory=list)


def _budget_matched_restarts(treatment_evaluations: int, restart_cost: int) -> int:
    if restart_cost <= 0:
        raise ValueError(f"restart_cost must be positive, got {restart_cost}")
    # Floor, not round: the control arm may never be given more compute than the
    # treatment spent, because an over-funded control would understate the
    # wrapper's contribution -- an error in the direction this tool is least
    # entitled to make. At least one restart, or there is nothing to compare.
    return max(1, treatment_evaluations // restart_cost)


def run_arms(
    scaffold: Scaffold,
    problem: Any,
    seeds: Sequence[int],
    *,
    map_fn: Callable[[Callable[[int], Any], Iterable[int]], Iterable[Any]] = map,
) -> ArmOutcomes:
    """Run both arms over ``seeds``, matching the control's budget to the treatment's.

    ``map_fn`` is the parallelism seam. It defaults to the builtin ``map`` so a
    single-machine run needs no configuration, and accepts anything with the same
    shape -- ``concurrent.futures.Executor.map``, or a cluster submitter -- so
    distribution is a caller's choice rather than a rewrite. Seeds are
    independent by construction, so the only requirement on an alternative is
    that it preserves input order, which the pairing depends on.
    """
    if not seeds:
        raise ValueError("at least one seed is required")

    base = scaffold.unwrap()  # raises NotSeparableError if there is no inner primitive

    treatment = list(map_fn(_RunTreatment(scaffold, problem), seeds))

    # The control's budget is read from what the treatment actually spent, not
    # from what it was configured to spend. A wrapper that exits early has its
    # control arm shrink to match; crediting the control with unspent compute
    # would flatter the wrapper.
    treatment_evaluations = sum(o.evaluations_used for o in treatment)
    per_seed_evaluations = treatment_evaluations // len(seeds)
    restarts = _budget_matched_restarts(per_seed_evaluations, base.restart_cost)

    control = list(map_fn(_RunControl(base, problem, restarts), seeds))

    return ArmOutcomes(
        treatment=treatment,
        control=control,
        seeds=list(seeds),
        restarts_per_seed=restarts,
        treatment_evaluations=treatment_evaluations,
        control_evaluations=sum(o.evaluations_used for o in control),
    )


@dataclass(frozen=True)
class _RunTreatment:
    """One treatment-arm run, as a picklable callable.

    `map_fn` is this module's parallelism seam, and every obvious parallel
    implementation of it -- `multiprocessing.Pool.map`, `ProcessPoolExecutor.map`
    -- pickles the callable it is handed. A lambda or a nested function cannot be
    pickled, so passing one here made the seam decorative: it worked with the
    default `map` and failed with every process pool. These small classes are
    what actually open it.

    Threads are not an option regardless: the adapter counts real fits by
    patching a class attribute, which is process-wide.
    """

    scaffold: Any
    problem: Any

    def __call__(self, seed: int) -> SearchOutcome:
        return self.scaffold.run(self.problem, seed)


@dataclass(frozen=True)
class _RunControl:
    """One control-arm run: `restarts` independent searches, then the pipeline's own pick."""

    base: Any
    problem: Any
    restarts: int

    def __call__(self, seed: int) -> SearchOutcome:
        outcomes = [
            self.base.search(self.problem, seed * _SEED_STRIDE + r) for r in range(self.restarts)
        ]
        best = self.base.select(outcomes)
        # Report the whole arm's spend, not just the winning restart's: the
        # restarts that lost were still paid for.
        return SearchOutcome(
            metrics=best.metrics,
            evaluations_used=sum(o.evaluations_used for o in outcomes),
            representation=best.representation,
        )


def _holm_correct(
    per_metric: Mapping[str, MetricVerdict], alpha: float
) -> dict[str, MetricVerdict]:
    """Correct across the metrics audited together (RFC-0001 section 4.2).

    Three metrics each tested at 0.05 is not a 0.05 procedure. Left uncorrected,
    the chance of at least one spurious claim grows with the number of metrics,
    and this audit exists to refuse exactly that sort of quietly-inflated
    confidence -- it would be incoherent to police the control arm's budget and
    then not police our own error rate.

    Holm rather than Bonferroni because Holm is uniformly more powerful and
    controls the same family-wise rate: claims are sorted by strength and the
    k-th is tested against ``alpha / (m - k)``, stepping down until one fails.

    Only verdicts that actually claim something enter the family. An
    ``INCONCLUSIVE`` asserts nothing, so counting it would inflate ``m`` and
    penalise the real claims for company they never kept.

    A claim that does not survive becomes ``INCONCLUSIVE``: the evidence did not
    hold up once the family was accounted for, which is precisely "could not
    tell" rather than "no effect". Its uncorrected p-value is retained so the
    downgrade is auditable rather than silent.
    """
    claims = sorted(
        ((name, v) for name, v in per_metric.items() if v.p_value is not None),
        key=lambda item: item[1].p_value or 0.0,
    )

    corrected = dict(per_metric)
    running_max = 0.0
    family_size = len(claims)

    for index, (name, verdict) in enumerate(claims):
        raw = verdict.p_value or 0.0
        # Step-down: enforce monotonic non-decreasing adjusted p-values, so a
        # weak early claim cannot let a stronger later one through.
        running_max = max(running_max, min(1.0, (family_size - index) * raw))
        survives = running_max <= alpha
        corrected[name] = dataclasses.replace(
            verdict,
            verdict=verdict.verdict if survives else Verdict.INCONCLUSIVE,
            adjusted_p_value=running_max,
            correction=f"holm(family={family_size}, alpha={alpha:g})",
        )

    return corrected


def _overall(per_metric: Mapping[str, MetricVerdict]) -> Verdict:
    """Collapse per-metric verdicts into one.

    Order matters and is deliberate. ``HARMFUL`` outranks ``CONTRIBUTES``, so a
    wrapper that helps on one pre-registered metric while hurting on another is
    reported by its harm. The first real sweep produced exactly that case -- a
    scaffold significantly worse on error and significantly better on complexity
    -- and the opposite precedence reported it as a contributor, burying the
    regression under the win. A reader can find the improvement unaided; the
    regression is the part they will not find, and this tool exists to refuse
    the error that flatters the wrapper.

    ``NULL`` requires *every* metric to be established as equivalent, so a single
    ``INCONCLUSIVE`` blocks it -- absence of evidence must not aggregate into
    evidence of absence.

    A mixed result is therefore reported as ``HARMFUL`` with the per-metric table
    beside it. That is lossy, and the per-metric verdicts, not this label, are
    the finding.
    """
    verdicts = {v.verdict for v in per_metric.values()}
    if Verdict.HARMFUL in verdicts:
        return Verdict.HARMFUL
    if Verdict.CONTRIBUTES in verdicts:
        return Verdict.CONTRIBUTES
    if verdicts == {Verdict.NULL}:
        return Verdict.NULL
    return Verdict.INCONCLUSIVE


def audit(
    scaffold: Scaffold,
    problem: Any,
    seeds: Sequence[int],
    *,
    margins: Mapping[str, float],
    higher_is_better: Mapping[str, bool] | None = None,
    confidence: float = 0.90,
    map_fn: Callable[[Callable[[int], Any], Iterable[int]], Iterable[Any]] = map,
) -> AuditReport:
    """Run the audit and return a verdict per metric plus an overall verdict.

    ``margins`` is the pre-registered practical-equivalence margin per metric and
    is required, with no default. A margin chosen after seeing the intervals is
    not an audit, and a default would be exactly that choice made silently on the
    caller's behalf.
    """
    if not margins:
        raise ValueError("margins are required: an audit with no pre-registered margin is not one")

    orientation = dict(higher_is_better or {})
    arms = run_arms(scaffold, problem, seeds, map_fn=map_fn)

    per_metric: dict[str, MetricVerdict] = {}
    for metric, margin in margins.items():
        for arm_name, outcomes in (("treatment", arms.treatment), ("control", arms.control)):
            if any(metric not in o.metrics for o in outcomes):
                raise KeyError(f"metric {metric!r} missing from {arm_name} outcomes")
        per_metric[metric] = equivalence_verdict(
            [float(o.metrics[metric]) for o in arms.treatment],
            [float(o.metrics[metric]) for o in arms.control],
            metric=metric,
            margin=margin,
            higher_is_better=orientation.get(metric, False),
            confidence=confidence,
        )

    per_metric = _holm_correct(per_metric, alpha=(1.0 - confidence) / 2.0)

    limitations = [
        (
            "Per-metric verdicts are Holm-corrected across the metrics audited "
            "together, and the correction is recorded on each verdict; a claim that "
            "did not survive reads INCONCLUSIVE with its uncorrected p-value retained. "
            "The correction's p-values are parametric (TOST or one-sided t) while the "
            "intervals that set the verdicts are BCa bootstrap, and on heavy-tailed "
            "metrics the two can disagree -- so this bounds the family-wise error rate "
            "approximately rather than exactly. The mismatch can only withdraw claims, "
            "never add them, so it errs toward reporting less than was found."
        ),
        (
            "The identical-representation rate compares strings literally, not "
            "semantically, so it under-reports agreement between the arms."
        ),
        (
            f"The verdict holds at this budget only ({arms.treatment_evaluations} "
            "evaluations across all seeds) and does not transfer to another."
        ),
    ]

    return AuditReport(
        scaffold=scaffold.name,
        verdict=_overall(per_metric),
        per_metric=per_metric,
        arms=arms,
        degeneracy=assess_degeneracy([o.intermediate_representations for o in arms.treatment]),
        limitations=limitations,
    )
