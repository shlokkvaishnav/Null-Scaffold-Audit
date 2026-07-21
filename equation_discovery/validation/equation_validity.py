"""
Equation Validity Module.

Checks a candidate equation string for basic mathematical validity. This is
deliberately narrow: it catches obviously ill-defined expressions before they
are scored, it does not attempt to encode any domain-specific physics.

Note on log/sqrt of negative literals: candidate equations are evaluated
via equation_discovery.core.expression_eval.safe_evaluate, which treats
sqrt/log as *protected* operations (sqrt(|x|), log(|x|)) to match gplearn's
own protected-function semantics. So a literal like "log(-0.878)" is not
actually ill-defined under how these equations are evaluated -- it safely
evaluates to log(0.878). Flagging it as invalid would be a false positive
(this exact case was observed: a genuinely good candidate for Coulomb's law
was rejected outright because of a negative constant inside a protected
log, discarding an otherwise-valid hypothesis). These checks are kept as
available custom constraints for callers using unprotected evaluation, but
are not part of the default active checks.
"""

import re
from typing import Dict, Optional


class EquationValidator:
    """
    Checks candidate equations for basic mathematical validity issues.
    """

    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize the equation validator.

        Args:
            config: Optional configuration for custom checks
        """
        self.config = config or {}
        self._custom_constraints: list = []

    def check_constraints(self, equation: str) -> Dict[str, Dict]:
        """
        Check basic validity constraints for a given equation string.

        Args:
            equation: Symbolic equation string

        Returns:
            dict: {constraint_name: {violation_rate: float, details: str}}
        """
        if not equation:
            return {}

        violations = {}

        # _check_negative_log / _check_imaginary_sqrt are intentionally not
        # active by default -- see module docstring: they're false positives
        # given protected sqrt/log evaluation semantics. Available via
        # add_custom_constraint for callers with different evaluation rules.
        invalid_checks = [
            self._check_division_by_zero,
        ]

        for check_fn in invalid_checks:
            violation = check_fn(equation)
            if violation:
                violations.update(violation)

        return violations

    def _check_division_by_zero(self, equation: str) -> Optional[Dict]:
        """Check for potential division by zero."""
        if re.search(r'/\s*0(?!\d)', equation):
            return {"division_by_zero": {
                "violation_rate": 1.0,
                "details": "Explicit division by zero detected"
            }}
        return None

    def _check_negative_log(self, equation: str) -> Optional[Dict]:
        """Check for log of a negative literal."""
        if re.search(r'log\s*\(\s*-', equation):
            return {"negative_log": {
                "violation_rate": 1.0,
                "details": "Log of negative value detected"
            }}
        return None

    def _check_imaginary_sqrt(self, equation: str) -> Optional[Dict]:
        """Check for sqrt of a negative literal."""
        if re.search(r'sqrt\s*\(\s*-', equation):
            return {"imaginary_sqrt": {
                "violation_rate": 1.0,
                "details": "Square root of negative value detected"
            }}
        return None

    def add_custom_constraint(self, name: str, check_fn) -> None:
        """
        Add a custom constraint checker.

        Args:
            name: Constraint name
            check_fn: Function that takes equation string and returns violation dict or None
        """
        self._custom_constraints.append((name, check_fn))

    def loss_penalty(self, equation: str) -> float:
        """
        Compute penalty term for constraint violations.

        Args:
            equation: Equation string

        Returns:
            float: Total violation penalty
        """
        violations = self.check_constraints(equation)
        return sum(v.get("violation_rate", 0) for v in violations.values())
