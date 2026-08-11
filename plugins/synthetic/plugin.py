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


def register(registry: PluginRegistry) -> None:
    """Register this module's domain plugin. Reuses algorithms already
    registered elsewhere (e.g. by plugins.physics.plugin.register)
    rather than re-registering them here."""
    registry.register_domain(SyntheticRegressionDomainPlugin.name, SyntheticRegressionDomainPlugin)
