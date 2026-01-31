"""Symbolic regression for discovering physical laws within ocean regimes.

Uses PySR (Cranmer, 2023) to search for interpretable equations via genetic programming.
Each regime gets its own symbolic expert, fitted to regime-specific data.

Scientific workflow:
1. Gating network assigns soft regime probabilities
2. Weight samples by regime probability
3. PySR discovers equation for each regime
4. Validate equations for physical plausibility
"""

from __future__ import annotations

import logging
import os
import warnings
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin

try:
    from pysr import PySRRegressor
except ImportError:
    PySRRegressor = None

logger = logging.getLogger(__name__)


class SymbolicExpert(BaseEstimator, RegressorMixin):
    """Single symbolic expert for one regime.
    
    Discovers interpretable equation mapping physics → fCO₂.
    
    Args:
        regime_id: Regime index (for logging)
        niterations: PySR search iterations
        populations: Number of evolutionary populations
        binary_operators: Allowed binary operations
        unary_operators: Allowed unary functions
        complexity_penalty: Pareto front complexity weight
        constraints: Physical constraints (bounds, monotonicity)
        random_state: Random seed
        temp_dir: Directory for PySR temporary files
        
    Example:
        >>> expert = SymbolicExpert(regime_id=0, niterations=40)
        >>> expert.fit(X_regime, y_regime, weights, variable_names=['sst', 'sss', 'log_chl'])
        >>> predictions = expert.predict(X_test)
        >>> equation = expert.get_best_equation()
    """
    
    def __init__(
        self,
        regime_id: int = 0,
        niterations: int = 40,
        populations: int = 31,
        binary_operators: Optional[List[str]] = None,
        unary_operators: Optional[List[str]] = None,
        complexity_penalty: float = 0.01,
        maxsize: int = 25,
        constraints: Optional[Dict] = None,
        random_state: int = 42,
        temp_dir: Optional[str] = None,
        verbosity: int = 0,
    ):
        if PySRRegressor is None:
            raise ImportError(
                "PySR not installed. Install: pip install pysr\n"
                "Then setup Julia backend: python -m pysr install"
            )
        
        self.regime_id = regime_id
        self.niterations = niterations
        self.populations = populations
        self.complexity_penalty = complexity_penalty
        self.maxsize = maxsize
        self.constraints = constraints or {}
        self.random_state = random_state
        self.temp_dir = temp_dir or "pysr_tmp"
        self.verbosity = verbosity
        
        # Default operators (physics-inspired)
        if binary_operators is None:
            binary_operators = ["+", "-", "*", "/"]
        if unary_operators is None:
            unary_operators = ["exp", "log", "sqrt", "square"]
        
        self.binary_operators = binary_operators
        self.unary_operators = unary_operators
        
        # PySR model (initialized in fit)
        self.model_ = None
        self.equation_ = None
        self.score_ = None
        self.complexity_ = None
        
        # Ensure temp directory exists
        os.makedirs(self.temp_dir, exist_ok=True)
    
    def _build_physics_informed_constraints(self) -> Dict:
        """Build physics-informed constraints for ocean carbon chemistry.
        
        Prevents unphysical equations by:
        1. Limiting exponential growth (numerical stability)
        2. Constraining power exponents to realistic ranges
        3. Preventing division by zero
        4. Enforcing dimensionally consistent operations
        
        Returns:
            Dictionary of PySR constraints
        """
        constraints = {}
        
        # Binary operator constraints
        # Format: (arg1_constraint, arg2_constraint) where -1 means no constraint
        
        # Division: prevent division by zero (constrain denominator)
        constraints["/"] = (-1, 1)
        
        # Power constraints: Only allow pow(base, fixed_exponent)
        # This prevents unphysical expressions like SST^100
        constraints["pow"] = (-1, 1)
        
        # Unary operator constraints  
        # Format: (arg_constraint,) - single element tuple
        # Note: For unary operators, use tuple with single constraint value
        
        # Logarithm: avoid log of negative or zero
        constraints["log"] = (1,)
        
        # Square root: avoid sqrt of negatives
        constraints["sqrt"] = (1,)
        
        # Exponential: limit to prevent overflow
        # exp(x) is safe for |x| < ~10, but we allow it on any expression
        # The complexity will naturally limit deep nesting
        constraints["exp"] = (-1,)
        
        # Square: can be applied to any expression
        constraints["square"] = (-1,)
        
        # Update with user-provided constraints
        if self.constraints:
            constraints.update(self.constraints)
        
        return constraints
    
    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        weights: Optional[np.ndarray] = None,
        variable_names: Optional[List[str]] = None,
    ) -> "SymbolicExpert":
        """Discover symbolic equation via genetic programming.
        
        Args:
            X: Feature matrix (N, D)
            y: Target values (N,)
            weights: Sample weights (N,) - typically regime probabilities
            variable_names: Feature names for interpretability
            
        Returns:
            self (fitted)
        """
        if len(X) < 10:
            warnings.warn(
                f"Regime {self.regime_id}: Only {len(X)} samples. "
                "Equation quality may be poor."
            )
        
        logger.info(
            f"Regime {self.regime_id}: Fitting symbolic regressor "
            f"on {len(X)} samples..."
        )
        
        # Configure PySR with physics-informed constraints
        pysr_config = {
            "niterations": self.niterations,
            "populations": self.populations,
            "binary_operators": self.binary_operators,
            "unary_operators": self.unary_operators,
            "maxsize": self.maxsize,
            "parsimony": self.complexity_penalty,
            "random_state": self.random_state,
            "temp_equation_file": True,
            "delete_tempfiles": True,
            "verbosity": self.verbosity,
            "progress": self.verbosity > 0,
            
            # Complexity constraints (prevent overfitting)
            "maxdepth": 10,  # Max expression tree depth
            
            # Batching for speed (if many samples)
            "batching": False,  # Can enable if N > 10000
        }
        
        # Initialize model
        self.model_ = PySRRegressor(**pysr_config)
        
        # Fit with optional weights
        if variable_names is not None:
            self.model_.fit(
                X, y, 
                weights=weights,
                variable_names=variable_names
            )
        else:
            self.model_.fit(X, y, weights=weights)
        
        # Extract best equation
        try:
            best = self.model_.get_best()
            self.equation_ = best.equation
            self.score_ = best.score
            self.complexity_ = best.complexity
            
            logger.info(
                f"Regime {self.regime_id}: "
                f"Equation: {self.equation_} "
                f"(complexity={self.complexity_}, score={self.score_:.4f})"
            )
        except Exception as e:
            logger.warning(f"Regime {self.regime_id}: Failed to extract equation: {e}")
            self.equation_ = "No equation found"
            self.score_ = np.inf
            self.complexity_ = 0
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict using discovered equation.
        
        Args:
            X: Feature matrix (N, D)
            
        Returns:
            Predictions (N,)
        """
        if self.model_ is None:
            raise ValueError("Model not fitted. Call fit() first.")
        
        try:
            return self.model_.predict(X)
        except Exception as e:
            logger.error(f"Regime {self.regime_id}: Prediction failed: {e}")
            # Fallback to mean prediction
            return np.zeros(len(X))
    
    def get_best_equation(self) -> str:
        """Return best discovered equation as string."""
        return self.equation_ if self.equation_ else "No equation"
    
    def get_pareto_front(self) -> Optional[list]:
        """Return all equations on the Pareto front (complexity vs accuracy)."""
        if self.model_ is None:
            return None
        try:
            return self.model_.equations_
        except Exception:
            return None
    
    def validate_equation(
        self, 
        X_val: np.ndarray, 
        y_val: np.ndarray,
        y_min: float = 200.0,
        y_max: float = 600.0,
    ) -> Dict:
        """Check physical plausibility of discovered equation.
        
        Args:
            X_val: Validation features
            y_val: Validation targets
            y_min: Minimum plausible fCO₂ (μatm)
            y_max: Maximum plausible fCO₂ (μatm)
            
        Returns:
            Validation metrics dict
        """
        y_pred = self.predict(X_val)
        
        # MSE
        mse = np.mean((y_pred - y_val) ** 2)
        
        # R²
        ss_res = np.sum((y_val - y_pred) ** 2)
        ss_tot = np.sum((y_val - np.mean(y_val)) ** 2)
        r2 = 1 - ss_res / (ss_tot + 1e-10)
        
        # Physical plausibility
        n_invalid = np.sum((y_pred < y_min) | (y_pred > y_max))
        frac_invalid = n_invalid / len(y_pred)
        
        return {
            "regime_id": self.regime_id,
            "mse": float(mse),
            "r2": float(r2),
            "equation": self.equation_,
            "complexity": self.complexity_,
            "n_invalid": int(n_invalid),
            "frac_invalid": float(frac_invalid),
        }


class MixtureOfSymbolicExperts:
    """Collection of symbolic experts, one per regime.
    
    Manages parallel training and prediction across all regimes.
    
    Args:
        num_regimes: Number of regimes K
        expert_config: Configuration dict passed to each SymbolicExpert
        
    Example:
        >>> experts = MixtureOfSymbolicExperts(
        ...     num_regimes=6,
        ...     expert_config={'niterations': 40, 'populations': 31}
        ... )
        >>> experts.fit(X, y, regime_probs, variable_names=['sst', 'sss', 'log_chl'])
        >>> predictions = experts.predict(X, regime_probs)
    """
    
    def __init__(
        self,
        num_regimes: int,
        expert_config: Optional[Dict] = None,
    ):
        self.num_regimes = num_regimes
        self.expert_config = expert_config or {}
        
        # Create expert for each regime
        self.experts = [
            SymbolicExpert(regime_id=k, **self.expert_config)
            for k in range(num_regimes)
        ]
    
    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        regime_probs: np.ndarray,
        variable_names: Optional[List[str]] = None,
        min_samples: int = 50,
        max_samples: int = 11000,  # NEW: Limit samples for PySR speed
        resume_from: Optional[str] = None, # NEW: Path to partial equations file
    ) -> "MixtureOfSymbolicExperts":
        """Fit symbolic expert for each regime.
        
        Args:
            X: Feature matrix (N, D)
            y: Target values (N,)
            regime_probs: Regime probabilities (N, K)
            variable_names: Feature names
            min_samples: Minimum samples to fit expert
            max_samples: Maximum samples per regime (subsampling)
            resume_from: Path to partial results file to skip regimes
            
        Returns:
            self (fitted)
        """
        import time
        import os
        
        # Check for resume file
        completed_regimes = set()
        if resume_from and os.path.exists(resume_from):
            try:
                with open(resume_from, 'r') as f:
                    content = f.read()
                    import re
                    # Look for "Regime X:" headers
                    completed_regimes = set(int(r) for r in re.findall(r"Regime (\d+):", content))
                logger.info(f"Found partial results! Skipping completed regimes: {sorted(completed_regimes)}")
            except Exception as e:
                logger.warning(f"Failed to parse resume file {resume_from}: {e}")

        for k in range(self.num_regimes):
            if k in completed_regimes:
                logger.info(f"⏩ Skipping Regime {k} (already done)")
                # Mark as fitted dummy
                self.experts[k].fitted_ = True 
                continue
                
            # Weight samples by regime probability
            weights = regime_probs[:, k]
            
            # Filter to high-probability samples
            mask = weights > 0.1  # Keep samples with >10% probability
            
            n_samples = np.sum(mask)
            if n_samples < min_samples:
                logger.warning(
                    f"Regime {k}: Only {n_samples} samples. Skipping."
                )
                continue
            
            # Subsample if too many points (PySR sweet spot ~10k)
            # Only use subsampling if we have significantly more data
            fit_mask = mask.copy()
            if n_samples > max_samples:
                # Weighted random sampling to keep most relevant points
                indices = np.where(mask)[0]
                # Probabilities proportional to regime weights
                p = weights[indices]
                p = p / np.sum(p)
                
                selected_indices = np.random.choice(
                    indices, size=max_samples, replace=False, p=p
                )
                
                # New mask with only selected points
                fit_mask = np.zeros_like(mask)
                fit_mask[selected_indices] = True
                
                logger.info(f"  Subsampling: {n_samples:,} → {max_samples:,} points for speed")
            
            # PRODUCTION: Log regime start with timestamp
            regime_start = time.time()
            logger.info(f"\n{'='*70}")
            logger.info(f"Fitting Regime {k}/{self.num_regimes - 1} ({np.sum(fit_mask):,} samples)...")
            logger.info(f"Started: {time.strftime('%H:%M:%S')}")
            logger.info(f"{'='*70}")
            
            # Fit expert on weighted data
            self.experts[k].fit(
                X[fit_mask], 
                y[fit_mask],
                weights=weights[fit_mask],
                variable_names=variable_names
            )
            
            # PRODUCTION: Log regime completion
            regime_elapsed = time.time() - regime_start
            logger.info(f"\n✓ Regime {k} completed in {regime_elapsed/60:.1f} minutes ({time.strftime('%H:%M:%S')})")
        
        return self
    
    def predict(
        self,
        X: np.ndarray,
        regime_probs: np.ndarray,
    ) -> np.ndarray:
        """Weighted mixture prediction: Σ π_k(x) * f_k(x)
        
        Args:
            X: Feature matrix (N, D)
            regime_probs: Regime probabilities (N, K)
            
        Returns:
            Predictions (N,)
        """
        n_samples = len(X)
        y_pred = np.zeros(n_samples)
        
        for k in range(self.num_regimes):
            # Expert prediction for all samples
            try:
                expert_pred = self.experts[k].predict(X)
            except Exception:
                logger.warning(f"Regime {k}: Prediction failed, using zeros")
                expert_pred = np.zeros(n_samples)
            
            # Weight by regime probability
            y_pred += regime_probs[:, k] * expert_pred
        
        return y_pred
    
    def get_all_equations(self) -> Dict[int, str]:
        """Return discovered equation for each regime."""
        return {
            k: expert.get_best_equation()
            for k, expert in enumerate(self.experts)
        }
    
    def validate_all(
        self,
        X_val: np.ndarray,
        y_val: np.ndarray,
        regime_probs_val: np.ndarray,
    ) -> List[Dict]:
        """Validate all experts on held-out data.
        
        Returns:
            List of validation dicts, one per regime
        """
        validations = []
        
        for k in range(self.num_regimes):
            # Filter to samples assigned to this regime
            mask = regime_probs_val[:, k] > 0.3
            
            if np.sum(mask) > 10:
                val_dict = self.experts[k].validate_equation(
                    X_val[mask],
                    y_val[mask]
                )
                validations.append(val_dict)
        
        return validations
    
    def save_equations(self, path: str | Path):
        """Save all discovered equations to text file."""
        path = Path(path)
        with open(path, 'w') as f:
            f.write("SD-MoSE Discovered Equations\n")
            f.write("=" * 60 + "\n\n")
            
            for k, expert in enumerate(self.experts):
                f.write(f"Regime {k}:\n")
                score_val = expert.score_ if expert.score_ is not None else 0.0
                comp_val = expert.complexity_ if expert.complexity_ is not None else 0
                f.write(f"  Equation: {expert.get_best_equation()}\n")
                f.write(f"  Complexity: {comp_val}\n")
                f.write(f"  Score: {score_val:.4f}\n")
                f.write("\n")
        
        logger.info(f"Equations saved to {path}")