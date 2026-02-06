"""
Hypothesis Module.
First-class symbolic hypothesis objects.
"""

import numpy as np
from typing import Dict, Optional, Any


class Hypothesis:
    """
    Represents a single symbolic hypothesis for a specific regime.
    """
    
    def __init__(self, equation: str, regime_id: int, iteration: int = 0):
        """
        Initialize a hypothesis.
        
        Args:
            equation: Symbolic equation string
            regime_id: Regime this hypothesis applies to
            iteration: Agent iteration when created
        """
        self.equation = str(equation)
        self.regime_id = regime_id
        
        # Validation state
        self.valid = True
        self.violation_log: Dict = {}
        
        # Evaluation scores
        self.likelihood: Optional[float] = None
        self.complexity: Optional[int] = None
        self.score: Optional[float] = None
        
        # Lineage tracking
        self.created_at = iteration
        self.parent_id: Optional[str] = None
    
    def evaluate(self, features: np.ndarray) -> np.ndarray:
        """
        Evaluate hypothesis on input features.
        
        Args:
            features: numpy array (n_samples, n_features)
        
        Returns:
            predictions: numpy array (n_samples,)
        """
        if features is None or len(features) == 0:
            return np.array([])
        
        # Ensure 2D
        if features.ndim == 1:
            features = features.reshape(-1, 1)
        
        n_samples, n_features = features.shape
        
        # Extract features safely
        x0 = features[:, 0] if n_features > 0 else np.zeros(n_samples)
        x1 = features[:, 1] if n_features > 1 else np.zeros(n_samples)
        
        eq = self.equation.strip()
        
        # Template evaluation
        evaluators = {
            "x0": lambda: x0,
            "x1": lambda: x1,
            "x0 + x1": lambda: x0 + x1,
            "x0 * x1": lambda: x0 * x1,
            "x0 + 0.1": lambda: x0 + 0.1,
            "x0 - x1": lambda: x0 - x1,
        }
        
        if eq in evaluators:
            return evaluators[eq]()
        
        # Try safe evaluation for simple expressions
        try:
            # Only allow safe operations
            allowed_names = {
                'x0': x0, 'x1': x1,
                'np': np, 'exp': np.exp, 'log': np.log,
                'sin': np.sin, 'cos': np.cos, 'sqrt': np.sqrt,
                'abs': np.abs
            }
            result = eval(eq, {"__builtins__": {}}, allowed_names)
            if isinstance(result, np.ndarray) and len(result) == n_samples:
                return result
        except Exception:
            pass
        
        # Fallback: return x0
        return x0
    
    def verify(self, verifier) -> bool:
        """
        Verify this hypothesis against physics constraints.
        
        Args:
            verifier: VerificationModule instance
            
        Returns:
            True if valid
        """
        self.violation_log = verifier.check(self) or {}
        
        # Check if any violations exceed threshold
        self.valid = not any(
            v.get("violation_rate", 0) > 0.1 
            for v in self.violation_log.values()
        ) if self.violation_log else True
        
        return self.valid
    
    def get_score_components(self) -> Dict[str, Optional[float]]:
        """Get breakdown of score components."""
        return {
            "likelihood": self.likelihood,
            "complexity": self.complexity,
            "total_score": self.score
        }
    
    def __repr__(self) -> str:
        status = "VALID" if self.valid else "INVALID"
        score_str = f"{self.score:.3f}" if self.score is not None else "N/A"
        return f"<Hypothesis regime={self.regime_id} status={status} score={score_str} eq={self.equation}>"
    
    def __str__(self) -> str:
        return self.equation
    
    def __eq__(self, other) -> bool:
        if not isinstance(other, Hypothesis):
            return False
        return self.equation == other.equation and self.regime_id == other.regime_id
    
    def __hash__(self) -> int:
        return hash((self.equation, self.regime_id))
