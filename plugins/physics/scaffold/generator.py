"""
Hypothesis Generator Module: symbolic hypothesis generation.

Supports two modes:
1. placeholder: Deterministic templates for infrastructure validation
2. pysr: Real symbolic regression (for final results)
"""

import itertools

# Large co-prime strides so that (iteration, regime) pairs cannot collide onto
# the same seed for any small configuration -- iteration 2 of regime 0 must not
# repeat the search that iteration 0 of regime 2 already ran.
_ITERATION_STRIDE = 7919
_REGIME_STRIDE = 104729
_SEED_MODULUS = 2**31 - 1


def _search_seed(base_seed: int, iteration: int, regime_id: int) -> int:
    """Derive a distinct-but-reproducible search seed for one (iteration, regime).

    The loop is supposed to explore. Handing the same seed to a deterministic
    search on every iteration means it cannot: the second and third calls
    recompute the first one's answer, at full cost. This spreads the configured
    seed across iterations so successive proposals differ, while keeping the
    whole run a pure function of `base_seed` -- a reproducible experiment, not a
    randomised one.

    Note what this does NOT do: it makes iterations *different*, not
    *informed*. Nothing here carries iteration k's outcome into iteration k+1;
    `priors` is still passed to the reasoner and ignored. Whether the loop
    should learn from its own history is a design question, and the audit will
    now measure the difference between "three varied restarts" and "a loop"
    rather than being unable to tell them apart.
    """
    return (
        int(base_seed) + iteration * _ITERATION_STRIDE + regime_id * _REGIME_STRIDE
    ) % _SEED_MODULUS


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
            from algorithms.symbolic import SymbolicHypothesisGenerator

            iteration = observation.get("iteration", 0) if isinstance(observation, dict) else 0
            agent_config = self.config.get("agent", {})
            gplearn_config = {
                "backend": "gplearn",
                **{k: v for k, v in agent_config.items() if k in {
                    "population_size", "generations", "stopping_criteria",
                }},
                # NOT the configured random_state verbatim. Passing that made
                # every iteration an identical call -- same X, same y, same
                # seed, same hyperparameters, and gplearn is deterministic. All
                # three iterations produced one distinct proposal, which is why
                # the audit's degeneracy ratio was exactly 1/3 on every problem,
                # at every budget, in both domains: arithmetic, not a tendency.
                "random_state": _search_seed(
                    agent_config.get("random_state", 0), iteration, regime_id
                ),
            }
            model = SymbolicHypothesisGenerator(gplearn_config)
            model.fit(X, y)
            equation_str = model.equation

            from engine.expressions.hypothesis import Hypothesis
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
            from engine.expressions.hypothesis import Hypothesis
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
        from engine.expressions.hypothesis import Hypothesis

        if self.mode == "placeholder":
            # Deterministic template
            equation = self.engine.propose()
            iteration = observation.get("iteration", 0) if isinstance(observation, dict) else 0
            return Hypothesis(equation=equation, regime_id=regime_id, iteration=iteration)

        # "gplearn" and "pysr" reasoners share the same propose_hypothesis(...) interface.
        return self.engine.propose_hypothesis(observation, regime_id, priors)
