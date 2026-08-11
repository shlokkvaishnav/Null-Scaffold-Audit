"""The second SDE domain plugin: synthetic tabular regression, no ground truth.

This exists specifically to stress-test the DomainPlugin contract designed in
Step 1 against something meaningfully different from the Feynman/physics
domain: no `equation_id` kwarg, no known ground-truth formula in metadata,
and a domain that doesn't assume every dataset comes with an answer to check
against. Deliberately reuses the existing algorithm plugins registered by
`plugins.physics.plugin` rather than adding new ones -- the point
of this plugin is to validate DomainPlugin generality, not add more
algorithms.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from engine.audit.problem import AuditProblem
from engine.evaluation.metrics import compute_fit_metrics
from engine.plugin import Dataset
from engine.registry import PluginRegistry
from plugins.synthetic.data import generate_synthetic_regression
from validators.equation_validity import EquationValidator


class SyntheticRegressionDomainPlugin:
    """DomainPlugin adapter around generate_synthetic_regression.

    Unlike FeynmanDomainPlugin, this domain has no known ground-truth
    equation to compare candidates against -- `metadata` describes the
    generating formula for reference only, it is not used for scoring.
    """

    name = "synthetic_regression"

    def __init__(self) -> None:
        self._validator = EquationValidator()

    def load_dataset(self, **kwargs: Any) -> Dataset:
        """Generate a synthetic regression dataset.

        Takes **kwargs to match the DomainPlugin contract, which the
        orchestrator satisfies by splatting an untyped config dict. Every key
        here is optional, so an empty call is valid and yields the defaults.

        Keys: seed (0), n_samples (200), n_features (6), noise_std (0.1).
        """
        n_features = kwargs.get("n_features", 6)
        noise_std = kwargs.get("noise_std", 0.1)
        # train_fraction=1.0 puts every sample in x_train/y_train (x_test/y_test
        # empty) -- the orchestrator does its own train/test split downstream,
        # so this just reuses generate_synthetic_regression as an unsplit source.
        split_data = generate_synthetic_regression(
            seed=kwargs.get("seed", 0),
            n_samples=kwargs.get("n_samples", 200),
            n_features=n_features,
            train_fraction=1.0,
            noise_std=noise_std,
        )
        return Dataset(
            X=split_data.x_train,
            y=split_data.y_train,
            feature_names=[f"x{i}" for i in range(n_features)],
            metadata={
                "generating_formula": "y = 1.5*x0 - 0.8*x1 + 0.5*x2*x3 + sin(x4) + noise",
                "noise_std": noise_std,
            },
        )

    def validate(self, equation: str | None) -> dict[str, Any]:
        if not equation:
            return {}
        return self._validator.check_constraints(equation)

    def score(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        equation: str | None = None,
    ) -> dict[str, float]:
        return compute_fit_metrics(y_true, y_pred, equation=equation)


class SyntheticProblemSource:
    """Makes this domain auditable, which is the point of it existing.

    The audit was written against one domain. Until it runs against a second
    one sharing no loader, no constraint set and no ground truth with the
    first, "the audit is domain independent" is a claim about intent rather
    than a checked property. This is that second domain.

    There is exactly one problem here: the generator's own target. Its ground
    truth is the formula from `data.py`, transcribed -- if that generator
    changes, this must change with it, and the recovery metric will say so by
    reporting nothing recovered.
    """

    name = "synthetic"

    TRAIN_FRACTION = 0.8
    N_FEATURES = 6
    # Features are standard normal, so +/-4 sigma covers the sampled support
    # with room to spare. The equivalence check samples inside these ranges.
    _RANGE = (-4.0, 4.0)

    def list_problems(self) -> list[str]:
        return ["synthetic_regression"]

    def build_problem(self, problem_id: str, *, n_samples: int, seed: int) -> AuditProblem:
        if problem_id not in self.list_problems():
            raise KeyError(f"{self.name} has no problem {problem_id!r}; try {self.list_problems()}")

        # train_fraction=1.0 so the split below is the only one applied, keeping
        # this identical in shape to every other problem source.
        data = generate_synthetic_regression(
            seed=seed,
            n_samples=n_samples,
            n_features=self.N_FEATURES,
            train_fraction=1.0,
        )
        split = int(self.TRAIN_FRACTION * len(data.y_train))
        variables = [f"x{i}" for i in range(self.N_FEATURES)]
        return AuditProblem(
            equation_id=problem_id,
            x_train=data.x_train[:split],
            y_train=data.y_train[:split],
            x_test=data.x_train[split:],
            y_test=data.y_train[split:],
            ground_truth={
                # The noise-free target. Additive noise is not part of the law
                # being recovered, and including it would make exact recovery
                # unachievable by construction.
                "formula": "1.5*x0 - 0.8*x1 + 0.5*x2*x3 + sin(x4)",
                "variables": variables,
                "ranges": {name: self._RANGE for name in variables},
            },
        )


def register(registry: PluginRegistry) -> None:
    """Register this module's domain plugin. Reuses algorithms already
    registered elsewhere (e.g. by plugins.physics.plugin.register)
    rather than re-registering them here."""
    registry.register_domain(SyntheticRegressionDomainPlugin.name, SyntheticRegressionDomainPlugin)
    registry.register_problem_source(SyntheticProblemSource.name, SyntheticProblemSource)
