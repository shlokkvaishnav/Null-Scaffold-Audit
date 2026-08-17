"""Exposes the DiscoveryAgent loop to the engine's null-scaffold audit.

The decomposition the audit needs is already present in this pipeline, which is
why it is the first thing audited: `DiscoveryAgent` is the scaffold (observe ->
retrieve -> reason -> verify -> learn), and the search primitive it wraps is
`SymbolicHypothesisGenerator`, constructed inside `GplearnReasoner`. So
`unwrap()` is real here rather than a courtesy -- the control arm runs the same
class the treatment arm runs, not a reimplementation of it.

Accounting: one gplearn fit is charged `population_size * generations` candidate
evaluations. That is the configured ceiling, not necessarily the amount spent --
gplearn stops early when `stopping_criteria` is met. Both arms are charged the
same way, so the *match* between arms is exact even though the absolute figure
is an upper bound. Stated because a reader comparing this to gplearn's internal
counters would otherwise find a discrepancy and assume a bug.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from algorithms.symbolic import SymbolicHypothesisGenerator
from engine.audit import (
    AuditProblem,
    NullScaffold,
    OracleScaffold,
    SearchOutcome,
    WastefulScaffold,
    paired_seed,
)
from engine.evaluation.equivalence import check_equivalence
from engine.evaluation.metrics import compute_fit_metrics
from engine.expressions.hypothesis import Hypothesis
from engine.expressions.refit import refit_constants
from engine.scoring import HypothesisScorer
from plugins.physics.scaffold.agent import DiscoveryAgent
from validators.equation_validity import EquationValidator

# gplearn defaults as configured by SymbolicHypothesisGenerator._build_model.
DEFAULT_POPULATION_SIZE = 500
DEFAULT_GENERATIONS = 20


@contextmanager
def count_fits() -> Iterator[dict[str, int]]:
    """Count real calls to the search primitive, rather than trusting configuration.

    The scaffold's configured spend and its actual spend diverge whenever a
    proposal is skipped (too little data in a regime, or an exception swallowed
    inside the reasoner). Charging the control arm for work the treatment never
    did would hand the control extra restarts and understate the scaffold --
    exactly the direction of error this tool must not make. So the treatment's
    budget is measured, not assumed.

    Not thread safe: it patches a class attribute. Parallel audits must use
    processes, not threads.
    """
    counter = {"fits": 0}
    original = SymbolicHypothesisGenerator.fit

    def counting_fit(self: SymbolicHypothesisGenerator, X: Any, y: Any) -> Any:
        counter["fits"] += 1
        return original(self, X, y)

    SymbolicHypothesisGenerator.fit = counting_fit  # type: ignore[method-assign]
    try:
        yield counter
    finally:
        SymbolicHypothesisGenerator.fit = original  # type: ignore[method-assign]


def score_like_the_pipeline(equation: str, observation: dict[str, Any]) -> float:
    """Score a candidate exactly as the scaffold's verification stage would.

    Both arms must be selected by the *same* rule, or the audit compares two
    things at once and attributes both to the scaffold. `HypothesisScorer`
    penalises complexity and constraint violations as well as misfit, so a
    control arm selected on raw training error would be picking longer, less
    valid candidates than the treatment -- and the resulting "the scaffold
    produces simpler equations" finding would be an artifact of the selection
    rule rather than a property of the scaffold.

    Mirrors `DiscoveryAgent.verify`: construct, verify against constraints, then
    score against the observation.
    """
    hypothesis = Hypothesis(equation=equation, regime_id=0, iteration=0)
    scorer = HypothesisScorer({}, validator=EquationValidator())
    hypothesis.verify(scorer)
    scorer.score_hypothesis(hypothesis, observation)
    score = getattr(hypothesis, "score", None)
    return float(score) if score is not None else float("-inf")


def _outcome_metrics(problem: AuditProblem, equation: str, seed: int) -> dict[str, float]:
    """Test-set metrics for a candidate, plus the ground-truth recovery flag.

    Predictions come from re-evaluating the *equation string*, for both arms.
    The control arm holds a fitted model object whose `predict` would be more
    accurate than its own printed equation -- gplearn rounds constants when it
    renders a program -- so using it would hand the control an advantage the
    treatment cannot have, since the scaffold retains only the string. What both
    pipelines actually emit is an equation, so that is what is measured.
    """
    candidate = Hypothesis(equation=equation, regime_id=0, iteration=0)
    y_pred = np.asarray(candidate.evaluate(problem.x_test), dtype=float)

    # A candidate can be non-evaluable on some test rows -- log of a negative,
    # division by zero -- and a symbolic search will produce such expressions
    # routinely. Those rows fall back to the training-set mean, which is what a
    # predictor that has learned nothing would say, so an unusable equation
    # scores like an uninformative one rather than crashing the sweep.
    #
    # Dropping the seed instead would be worse than arbitrary: failures are not
    # evenly distributed between the arms, so excluding them silently favours
    # whichever arm fails more often. The substitution is applied identically to
    # both arms, uses the TRAIN mean so no test information leaks, and the rate
    # is reported as its own metric rather than absorbed into rmse.
    nonfinite = ~np.isfinite(y_pred)
    if nonfinite.any():
        y_pred = np.where(
            nonfinite, float(np.mean(np.asarray(problem.y_train, dtype=float))), y_pred
        )

    fit = compute_fit_metrics(problem.y_test, y_pred, equation=equation)
    try:
        equivalence = check_equivalence(
            candidate_equation=equation,
            ground_truth_formula=problem.ground_truth["formula"],
            variables=problem.ground_truth["variables"],
            test_ranges=problem.ground_truth["ranges"],
            seed=seed,
        )
        recovered = float(bool(equivalence["symbolic_match"]))
        strictly_recovered = float(bool(equivalence["strict_match"]))
    # The blanket catch below is deliberate. sympy raises no single documented
    # exception type for unparseable or pathological input -- TypeError,
    # AttributeError, RecursionError and its own SympifyError all appear,
    # depending on the expression -- so enumerating them would silently miss
    # cases and crash a multi-hour sweep. An equivalence check that failed has
    # not shown the candidate to be correct, so the conservative reading is
    # "not recovered", which a crash would prevent us from recording at all.
    except Exception:  # noqa: BLE001
        recovered = 0.0
        strictly_recovered = 0.0

    return {
        "rmse": float(fit["rmse"]),
        "mae": float(fit["mae"]),
        "symbolic_complexity": float(fit["symbolic_complexity"]),
        # SRBench's definition: difference or ratio to ground truth simplifies
        # to a constant. This is the one the audit registers a margin for.
        "exact_recovery": recovered,
        # The stricter definition this project used before, recorded alongside
        # so relaxing the rule can never quietly inflate the headline figure.
        "strict_recovery": strictly_recovered,
        "nonfinite_fraction": float(nonfinite.mean()),
    }


@dataclass
class SymbolicRestartSearcher:
    """The bare search primitive, run as independent restarts.

    This is the same `SymbolicHypothesisGenerator` the agent's reasoner builds
    internally -- the control arm is the treatment arm with the loop removed,
    not a separate implementation that might differ for uninteresting reasons.
    """

    backend: str = "gplearn"
    population_size: int = DEFAULT_POPULATION_SIZE
    generations: int = DEFAULT_GENERATIONS

    @property
    def restart_cost(self) -> int:
        return self.population_size * self.generations

    def search(self, problem: AuditProblem, seed: int) -> SearchOutcome:
        model = SymbolicHypothesisGenerator(
            {
                "backend": self.backend,
                "random_state": seed,
                "population_size": self.population_size,
                "generations": self.generations,
            }
        )
        model.fit(problem.x_train, problem.y_train)
        equation = self._postprocess(str(model.equation), problem)

        metrics = _outcome_metrics(problem, equation, seed)
        # Selection uses the TRAINING set, via the pipeline's own scorer. The
        # scaffold picks its best hypothesis by a score computed on what it
        # observed; selecting on test data would give the control an advantage
        # the treatment never had and invert the audit's bias.
        metrics["selection_score"] = score_like_the_pipeline(
            equation, {"features": problem.x_train, "targets": problem.y_train}
        )

        return SearchOutcome(
            metrics=metrics,
            evaluations_used=self.restart_cost,
            representation=equation,
        )

    def _postprocess(self, equation: str, problem: AuditProblem) -> str:
        """What the searcher emits, before anything is measured about it.

        Identity here. It exists so a variant can change the candidate itself without
        copying the fifteen lines above it -- and, more importantly, so that when one
        does, both the metrics and the selection score are computed from the same
        post-processed string. Refitting a candidate and then scoring the original
        would be a silent inconsistency of exactly the kind this audit exists to catch.
        """
        return equation

    def select(self, outcomes: Sequence[SearchOutcome]) -> SearchOutcome:
        return max(outcomes, key=lambda o: o.metrics["selection_score"])


@dataclass
class RefittingRestartSearcher(SymbolicRestartSearcher):
    """The same searcher, with the inner optimiser it does not have.

    gplearn is a one-level searcher: it evolves structures, and the constants inside
    them are random terminals shuffled by crossover like any other leaf. There is no
    parameter fit anywhere in it -- `const_range` appears twenty-three times in
    `gplearn.genetic`, and not one optimiser, least-squares or curve fit does.

    That matters for what this project measures. The selection ceiling came out near
    zero on twenty-six of thirty-six problems, which was read as "the training signal
    ranks candidates faithfully". But with no constants fitted, training error and
    held-out error are both dominated by the same quantity -- structural misfit plus
    whatever the constants happened to land on -- so the two ranks agreeing is close to
    mechanical. The ceiling might be measuring the absence of an optimiser rather than
    a property of the problems.

    This subclass is how that gets tested rather than argued: the ceiling measured
    through it, against the ceiling measured through its parent, on the same problems
    and seeds. Refitting uses the TRAIN split only, because the ceiling is a held-out
    quantity and fitting against the split it is scored on would manufacture the very
    advantage being measured.
    """

    def _postprocess(self, equation: str, problem: AuditProblem) -> str:
        return refit_constants(equation, problem.x_train, problem.y_train)


@dataclass
class ConcentratedSearchScaffold:
    """Spend the whole budget on one long search instead of several short ones.

    Every scaffold audited here so far can only win by *selecting* better among
    the candidates its searcher produced, and `engine.audit.calibration
    .selection_ceiling` puts a hard upper bound on what that is worth -- a bound
    measured below a twentieth of the margin on most of this domain's problems.
    A wrapper that cannot exceed the ceiling is null however clever it is.

    This one is not bounded by it, because it changes which candidates get
    produced: the same total evaluations, spent as a single run of
    `generations * factor` rather than as `factor` independent restarts. That
    makes it the first scaffold here whose verdict is genuinely open. It is an
    experiment, not a calibration, and either answer is worth having --
    concentration winning would say the searcher is still improving when a
    normal run stops, and losing would be direct evidence for the restart
    baseline that this whole audit leans on.

    Budget matching is exact rather than approximate: one fit of
    `population_size * generations * factor` against `unwrap()`'s restart cost
    of `population_size * generations` gives the control exactly `factor`
    restarts, with no remainder for the flooring rule to discard.
    """

    name: str = "ConcentratedSearch"
    factor: int = 3
    population_size: int = DEFAULT_POPULATION_SIZE
    generations: int = DEFAULT_GENERATIONS

    def unwrap(self) -> SymbolicRestartSearcher:
        return SymbolicRestartSearcher(
            population_size=self.population_size, generations=self.generations
        )

    def run(self, problem: AuditProblem, seed: int) -> SearchOutcome:
        # Derived through `paired_seed` so that under common random numbers this
        # run shares its stream with the control's first restart. The two are not
        # the same search -- one runs `factor` times as long -- but they start
        # from the same draw, which is exactly the pairing that makes the
        # difference between them attributable to the extra generations.
        concentrated = SymbolicRestartSearcher(
            population_size=self.population_size,
            generations=self.generations * self.factor,
        )
        return concentrated.search(problem, paired_seed(seed, 0))


def concentrated_search(
    *,
    max_iters: int = 3,
    population_size: int = DEFAULT_POPULATION_SIZE,
    generations: int = DEFAULT_GENERATIONS,
) -> ConcentratedSearchScaffold:
    """Runner seam for `ConcentratedSearchScaffold`.

    The runner constructs every scaffold with the same three keyword arguments,
    and `max_iters` is the one that means "how many searches would this have
    been". Here that number becomes the concentration factor, so the same
    command line describes the same budget whichever scaffold it names.
    """
    return ConcentratedSearchScaffold(
        factor=max_iters, population_size=population_size, generations=generations
    )


@dataclass
class RefitScaffold:
    """Restarts, then fits the constants of what they found, then selects.

    The one scaffold here whose verdict is not capped by the selection ceiling, because
    it changes the candidates rather than the choice among them -- and the one whose
    sign the literature actually predicts, since fitting parameters is reported to move
    correct-but-badly-scored structures up the ranking.

    **Its budget is not matched, and that has to be said rather than buried.** Refitting
    spends optimiser evaluations, and this audit counts budget in the searcher's own
    candidate evaluations, which do not include them. The control arm gets neither the
    refit nor compensating compute. So a CONTRIBUTES here answers "what does adding an
    inner optimiser buy?" and not "does this wrapper earn its budget?" -- it is an upper
    bound on the second, reported as such.

    The fair version of the same question is the ceiling measured through
    `RefittingRestartSearcher`, where both sides of the comparison are refitted.
    """

    name: str = "RefitScaffold"
    max_iters: int = 3
    population_size: int = DEFAULT_POPULATION_SIZE
    generations: int = DEFAULT_GENERATIONS

    def unwrap(self) -> SymbolicRestartSearcher:
        # The plain searcher, so the control arm is identical to every other
        # scaffold's here and the verdicts stay comparable across scaffolds.
        return SymbolicRestartSearcher(
            population_size=self.population_size, generations=self.generations
        )

    def run(self, problem: AuditProblem, seed: int) -> SearchOutcome:
        refitting = RefittingRestartSearcher(
            population_size=self.population_size, generations=self.generations
        )
        outcomes = [
            refitting.search(problem, paired_seed(seed, restart))
            for restart in range(self.max_iters)
        ]
        best = refitting.select(outcomes)
        return SearchOutcome(
            metrics=best.metrics,
            evaluations_used=sum(outcome.evaluations_used for outcome in outcomes),
            representation=best.representation,
            intermediate_representations=tuple(o.representation or "" for o in outcomes),
        )


def refit_scaffold(
    *,
    max_iters: int = 3,
    population_size: int = DEFAULT_POPULATION_SIZE,
    generations: int = DEFAULT_GENERATIONS,
) -> RefitScaffold:
    """Runner seam for `RefitScaffold`."""
    return RefitScaffold(
        max_iters=max_iters, population_size=population_size, generations=generations
    )


def _calibration_searcher(population_size: int, generations: int) -> SymbolicRestartSearcher:
    return SymbolicRestartSearcher(population_size=population_size, generations=generations)


def null_calibration(
    *,
    max_iters: int = 3,
    population_size: int = DEFAULT_POPULATION_SIZE,
    generations: int = DEFAULT_GENERATIONS,
) -> NullScaffold:
    """This domain's primitive, wrapped in the engine's null-by-construction scaffold.

    The three factories here exist because the audit runner constructs a
    scaffold from a `module:attribute` path with a fixed set of keyword
    arguments. They are the seam and nothing more: the calibration logic is the
    engine's and is shared by every domain, while the searcher it wraps is this
    plugin's. A second domain calibrates its own audit by writing three
    functions of this shape and no new statistics.

    Expected verdict: NULL. See `engine.audit.calibration` for what each of the
    three possible failures would mean.
    """
    return NullScaffold(
        base=_calibration_searcher(population_size, generations), restarts=max_iters
    )


def wasteful_calibration(
    *,
    max_iters: int = 3,
    population_size: int = DEFAULT_POPULATION_SIZE,
    generations: int = DEFAULT_GENERATIONS,
) -> WastefulScaffold:
    """Expected verdict: HARMFUL. The same budget, then keeps the worst of it."""
    return WastefulScaffold(
        base=_calibration_searcher(population_size, generations), restarts=max_iters
    )


def oracle_calibration(
    *,
    max_iters: int = 3,
    population_size: int = DEFAULT_POPULATION_SIZE,
    generations: int = DEFAULT_GENERATIONS,
) -> OracleScaffold:
    """Expected verdict: CONTRIBUTES. Selects on held-out error, which is cheating.

    Deliberately so: it is a ruler of known length, not a method. It answers the
    one question no run of a real pipeline can -- whether this audit, at this
    budget and this seed count, would notice a contribution if there were one.
    """
    return OracleScaffold(
        base=_calibration_searcher(population_size, generations),
        restarts=max_iters,
        metric="rmse",
        higher_is_better=False,
    )


@dataclass
class DiscoveryAgentScaffold:
    """The full agent loop, as submitted."""

    name: str = "DiscoveryAgent"
    backend: str = "gplearn"
    max_iters: int = 3
    num_regimes: int = 1
    population_size: int = DEFAULT_POPULATION_SIZE
    generations: int = DEFAULT_GENERATIONS
    failures: list[str] = field(default_factory=list)

    def unwrap(self) -> SymbolicRestartSearcher:
        return SymbolicRestartSearcher(
            backend=self.backend,
            population_size=self.population_size,
            generations=self.generations,
        )

    def run(self, problem: AuditProblem, seed: int) -> SearchOutcome:
        # Mirrors benchmark_runner._run_discovery_agent so the audit measures the
        # pipeline as benchmarked, not a variant configured for the occasion.
        config = {
            "agent": {
                "num_regimes": self.num_regimes,
                "use_verification": True,
                "use_memory": True,
                "use_belief": True,
                "use_reasoning": True,
                "reasoning_mode": self.backend,
                "random_state": seed,
                "population_size": self.population_size,
                "generations": self.generations,
            }
        }
        agent = DiscoveryAgent(config)
        observation = {"features": problem.x_train, "targets": problem.y_train}

        # Collected per step rather than read from memory at the end, because the
        # archive stores only *verified* hypotheses and deduplicates them. Reading
        # it would hide the repetition this check exists to find: three identical
        # proposals become one archive entry, which looks like one proposal rather
        # than like a loop that proposed the same thing three times.
        proposals: list[str] = []
        with count_fits() as counter:
            for _ in range(self.max_iters):
                agent.step(observation)
                proposals.extend(str(h.equation) for h in agent.proposed_hypotheses)

        if not agent.memory or not agent.memory.hypotheses:
            raise RuntimeError(f"scaffold produced no candidate on seed {seed}")

        best = max(agent.memory.hypotheses, key=lambda h: getattr(h, "score", float("-inf")))
        equation = str(best.equation)
        metrics = _outcome_metrics(problem, equation, seed)
        # Rescored through the same helper the control uses rather than read off
        # the agent's own field, so the two arms' selection scores are produced
        # by identical code and remain comparable.
        metrics["selection_score"] = score_like_the_pipeline(
            equation, {"features": problem.x_train, "targets": problem.y_train}
        )

        return SearchOutcome(
            metrics=metrics,
            evaluations_used=counter["fits"] * self.population_size * self.generations,
            representation=equation,
            intermediate_representations=tuple(proposals),
        )
