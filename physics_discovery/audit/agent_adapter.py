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

from engine.audit import SearchOutcome
from physics_discovery.core.agent import DiscoveryAgent
from physics_discovery.core.hypothesis import Hypothesis
from physics_discovery.core.scorer import HypothesisScorer
from physics_discovery.evaluation.metrics import compute_fit_metrics
from physics_discovery.evaluation.rediscovery import check_equivalence
from physics_discovery.generators.symbolic import SymbolicHypothesisGenerator

# gplearn defaults as configured by SymbolicHypothesisGenerator._build_model.
DEFAULT_POPULATION_SIZE = 500
DEFAULT_GENERATIONS = 20


@dataclass(frozen=True)
class RediscoveryProblem:
    """One benchmark instance, already split. Opaque to the engine."""

    equation_id: str
    x_train: np.ndarray
    y_train: np.ndarray
    x_test: np.ndarray
    y_test: np.ndarray
    ground_truth: dict[str, Any]


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
    scorer = HypothesisScorer({})
    hypothesis.verify(scorer)
    scorer.score_hypothesis(hypothesis, observation)
    score = getattr(hypothesis, "score", None)
    return float(score) if score is not None else float("-inf")


def _outcome_metrics(problem: RediscoveryProblem, equation: str, seed: int) -> dict[str, float]:
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
    fit = compute_fit_metrics(problem.y_test, y_pred, equation=equation)
    equivalence = check_equivalence(
        candidate_equation=equation,
        ground_truth_formula=problem.ground_truth["formula"],
        variables=problem.ground_truth["variables"],
        test_ranges=problem.ground_truth["ranges"],
        seed=seed,
    )
    return {
        "rmse": float(fit["rmse"]),
        "mae": float(fit["mae"]),
        "symbolic_complexity": float(fit["symbolic_complexity"]),
        "exact_recovery": float(bool(equivalence["symbolic_match"])),
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

    def search(self, problem: RediscoveryProblem, seed: int) -> SearchOutcome:
        model = SymbolicHypothesisGenerator(
            {
                "backend": self.backend,
                "random_state": seed,
                "population_size": self.population_size,
                "generations": self.generations,
            }
        )
        model.fit(problem.x_train, problem.y_train)
        equation = str(model.equation)

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

    def select(self, outcomes: Sequence[SearchOutcome]) -> SearchOutcome:
        return max(outcomes, key=lambda o: o.metrics["selection_score"])


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

    def run(self, problem: RediscoveryProblem, seed: int) -> SearchOutcome:
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
