"""
Hypothesis Generator Module: symbolic hypothesis generation.

Supports two modes:
1. placeholder: Deterministic templates for infrastructure validation
2. pysr: Real symbolic regression (for final results)
"""

import itertools


class DeterministicReasoner:
    """
    Simple symbolic generator to validate agent + ablation behavior.
    NOT used for final scientific results.
    """
    def __init__(self):
        self.templates = [
            "x0",
            "x1",
            "x0 + x1",
            "x0 * x1",
            "x0 + 0.1",
            "x0 - x1",
        ]
        self.counter = itertools.cycle(self.templates)

    def propose(self):
        """Generate next equation from template cycle."""
        return next(self.counter)


class GplearnReasoner:
    """
    Real symbolic regression using gplearn (default backend -- pure Python,
    no Julia runtime, so this is what the agent uses out-of-the-box in
    Docker/CI, with PySR available as an opt-in upgrade).
    """
    def __init__(self, config):
        self.config = config or {}

    def propose_hypothesis(self, observation, regime_id, priors=None):
        """
        Fit a fresh gplearn symbolic regressor and wrap its equation as a
        Hypothesis.

        Returns:
            Hypothesis object or None if insufficient data
        """
        if observation is None or "features" not in observation:
            return None

        X = observation["features"]
        y = observation["targets"]

        # Regime conditioning (optional)
        if "regime_labels" in observation:
            mask = observation["regime_labels"] == regime_id
            X = X[mask]
            y = y[mask]

        if len(X) < 10:
            return None

        try:
            from physics_discovery.generators.symbolic import SymbolicHypothesisGenerator

            gplearn_config = {
                "backend": "gplearn",
                **{k: v for k, v in self.config.get("agent", {}).items() if k in {
                    "population_size", "generations", "stopping_criteria", "random_state",
                }},
            }
            model = SymbolicHypothesisGenerator(gplearn_config)
            model.fit(X, y)
            equation_str = model.equation

            from .hypothesis import Hypothesis
            iteration = observation.get("iteration", 0) if isinstance(observation, dict) else 0
            return Hypothesis(equation=equation_str, regime_id=regime_id, iteration=iteration)
        # Blanket by necessity: a gplearn fit over caller-supplied data raises
        # anything from ValueError on degenerate input to arbitrary numpy
        # errors, and a failed proposal is simply "no hypothesis this round".
        # Returning None keeps the loop running, which is what the agent's
        # ablation flags depend on.
        except Exception:  # noqa: BLE001
            return None


class PySRReasoner:
    """
    Real symbolic regression using PySR (for final results).
    """
    def __init__(self, config):
        from pysr import PySRRegressor
        self.model = PySRRegressor(
            niterations=20,
            populations=10,
            population_size=50,
            maxsize=10,
            binary_operators=["+", "-", "*"],
            unary_operators=["exp", "log"],
            elementwise_loss="loss(x, y) = (x - y)^2",  # Updated parameter name
            verbosity=0,
            progress=False
        )

    def propose_hypothesis(self, observation, regime_id, priors=None):
        """
        Generate symbolic hypothesis using PySR.

        Returns:
            Hypothesis object or None if insufficient data
        """
        if observation is None or "features" not in observation:
            return None

        X = observation["features"]
        y = observation["targets"]

        # Regime conditioning (optional)
        if "regime_labels" in observation:
            mask = observation["regime_labels"] == regime_id
            X = X[mask]
            y = y[mask]

        # Insufficient data check
        if len(X) < 10:
            return None

        try:
            # Fit symbolic regression
            self.model.fit(X, y)

            # Get equations from PySR
            if hasattr(self.model, 'equations_'):
                eqs = self.model.equations_
                if len(eqs) > 0:
                    # Get best equation (last row typically best)
                    best = eqs.iloc[-1]
                    equation_str = best["equation"]
                else:
                    return None
            else:
                return None

            # Create hypothesis
            from .hypothesis import Hypothesis
            iteration = observation.get("iteration", 0) if isinstance(observation, dict) else 0
            h = Hypothesis(equation=equation_str, regime_id=regime_id, iteration=iteration)

            return h

        # Same reasoning as the gplearn path above, plus PySR's Julia bridge,
        # which surfaces backend failures as its own exception types.
        except Exception:  # noqa: BLE001
            return None


class HypothesisGenerator:
    """
    Core symbolic hypothesis-generation engine with switchable backends.
    """
    def __init__(self, config=None):
        self.config = config or {}
        # "gplearn" is the default: real symbolic regression, pure Python, no
        # Julia dependency, matching the Docker/CI default backend. "pysr" is
        # an opt-in upgrade. "placeholder" is test-infrastructure only (fixed
        # template strings, not real reasoning) -- kept for tests that need a
        # fast, deterministic stand-in, never the default.
        self.mode = self.config.get("agent", {}).get("reasoning_mode", "gplearn")

        if self.mode == "placeholder":
            self.engine = DeterministicReasoner()
        elif self.mode == "pysr":
            self.engine = PySRReasoner(self.config)
        elif self.mode == "gplearn":
            self.engine = GplearnReasoner(self.config)
        else:
            raise ValueError(f"Unknown reasoning_mode: {self.mode!r}")

    def propose_hypothesis(self, observation, regime_id, priors=None):
        """
        Generate a symbolic hypothesis for a specific regime.

        Args:
            observation: dict with 'features' and 'targets'
            regime_id: int, which regime to generate for
            priors: dict, prior knowledge (optional)

        Returns:
            Hypothesis object or None
        """
        from .hypothesis import Hypothesis

        if self.mode == "placeholder":
            # Deterministic template
            equation = self.engine.propose()
            iteration = observation.get("iteration", 0) if isinstance(observation, dict) else 0
            return Hypothesis(equation=equation, regime_id=regime_id, iteration=iteration)

        # "gplearn" and "pysr" reasoners share the same propose_hypothesis(...) interface.
        return self.engine.propose_hypothesis(observation, regime_id, priors)
