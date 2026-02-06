"""
Reasoning Module: Symbolic hypothesis generation.

Supports two modes:
1. placeholder: Deterministic templates for infrastructure validation
2. pysr: Real symbolic regression (for final results)
"""

import numpy as np
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
            from ..agent.hypothesis import Hypothesis
            iteration = observation.get("iteration", 0) if isinstance(observation, dict) else 0
            h = Hypothesis(equation=equation_str, regime_id=regime_id, iteration=iteration)
            
            return h
            
        except Exception as e:
            # PySR failed
            return None


class ReasoningModule:
    """
    Core symbolic reasoning engine with switchable backends.
    """
    def __init__(self, config=None):
        self.config = config or {}
        self.mode = self.config.get("agent", {}).get("reasoning_mode", "placeholder")
        
        if self.mode == "placeholder":
            self.engine = DeterministicReasoner()
        elif self.mode == "pysr":
            self.engine = PySRReasoner(self.config)
        else:
            # Default to placeholder
            self.engine = DeterministicReasoner()
    
    def propose_hypothesis(self, observation, regime_id, priors=None):
        """
        Generate a symbolic hypothesis for a specific regime.
        
        Args:
            observation: dict with 'features' and 'targets'
            regime_id: int, which regime to generate for
            priors: dict, scientific priors (optional)
        
        Returns:
            Hypothesis object or None
        """
        from ..agent.hypothesis import Hypothesis
        
        if self.mode == "placeholder":
            # Deterministic template
            equation = self.engine.propose()
            iteration = observation.get("iteration", 0) if isinstance(observation, dict) else 0
            return Hypothesis(equation=equation, regime_id=regime_id, iteration=iteration)
        
        elif self.mode == "pysr":
            # Real PySR
            return self.engine.propose_hypothesis(observation, regime_id, priors)
        
        return None
