"""Evaluation metrics for SD-MoSE.

Provides R², RMSE, and comprehensive metric calculation.
"""

import numpy as np
from typing import Dict


def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate R² (coefficient of determination).
    
    R² = 1 - SS_res / SS_tot
    
    Args:
        y_true: Ground truth values
        y_pred: Predicted values
        
    Returns:
        R² score (higher is better, max 1.0)
    """
    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()
    
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    
    if ss_tot == 0:
        return 0.0
    
    return 1 - (ss_res / ss_tot)


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate Root Mean Square Error.
    
    RMSE = sqrt(mean((y_true - y_pred)²))
    
    Args:
        y_true: Ground truth values
        y_pred: Predicted values
        
    Returns:
        RMSE (lower is better, units same as target)
    """
    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()
    
    return np.sqrt(np.mean((y_true - y_pred) ** 2))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate Mean Absolute Error."""
    return np.mean(np.abs(y_true - y_pred))


def calculate_metrics(
    y_true: np.ndarray, 
    y_pred: np.ndarray,
    regime_assignments: np.ndarray = None,
    n_regimes: int = None,
) -> Dict:
    """Calculate comprehensive metrics for SD-MoSE evaluation.
    
    Args:
        y_true: Ground truth pCO₂ values
        y_pred: Predicted pCO₂ values
        regime_assignments: Optional regime labels for per-regime metrics
        n_regimes: Number of regimes (required if regime_assignments given)
        
    Returns:
        Dictionary with:
        - overall_r2: Overall R² score
        - overall_rmse: Overall RMSE
        - overall_mae: Overall MAE
        - per_regime_r2: R² by regime (if regime_assignments given)
        - per_regime_rmse: RMSE by regime
        - per_regime_samples: Sample count by regime
    """
    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()
    
    metrics = {
        'overall_r2': r2_score(y_true, y_pred),
        'overall_rmse': rmse(y_true, y_pred),
        'overall_mae': mae(y_true, y_pred),
        'n_samples': len(y_true),
    }
    
    # Per-regime metrics
    if regime_assignments is not None and n_regimes is not None:
        regime_assignments = np.asarray(regime_assignments).flatten()
        
        per_regime_r2 = []
        per_regime_rmse = []
        per_regime_samples = []
        
        for k in range(n_regimes):
            mask = regime_assignments == k
            n_k = np.sum(mask)
            per_regime_samples.append(n_k)
            
            if n_k > 10:  # Need minimum samples for meaningful metrics
                r2_k = r2_score(y_true[mask], y_pred[mask])
                rmse_k = rmse(y_true[mask], y_pred[mask])
            else:
                r2_k = np.nan
                rmse_k = np.nan
            
            per_regime_r2.append(r2_k)
            per_regime_rmse.append(rmse_k)
        
        metrics['per_regime_r2'] = per_regime_r2
        metrics['per_regime_rmse'] = per_regime_rmse
        metrics['per_regime_samples'] = per_regime_samples
    
    return metrics


def print_metrics(metrics: Dict, regime_names: list = None):
    """Pretty-print evaluation metrics."""
    print("\n" + "=" * 50)
    print("EVALUATION METRICS")
    print("=" * 50)
    
    print(f"Overall R²:   {metrics['overall_r2']:.4f}")
    print(f"Overall RMSE: {metrics['overall_rmse']:.2f} μatm")
    print(f"Overall MAE:  {metrics['overall_mae']:.2f} μatm")
    print(f"N Samples:    {metrics['n_samples']}")
    
    if 'per_regime_r2' in metrics:
        print("\n" + "-" * 50)
        print("PER-REGIME METRICS:")
        print("-" * 50)
        print(f"{'Regime':<10} {'R²':>8} {'RMSE':>10} {'Samples':>10}")
        print("-" * 50)
        
        for k in range(len(metrics['per_regime_r2'])):
            name = regime_names[k] if regime_names else f"Regime {k}"
            r2 = metrics['per_regime_r2'][k]
            rmse_val = metrics['per_regime_rmse'][k]
            n = metrics['per_regime_samples'][k]
            
            r2_str = f"{r2:.4f}" if not np.isnan(r2) else "N/A"
            rmse_str = f"{rmse_val:.2f}" if not np.isnan(rmse_val) else "N/A"
            
            print(f"{name:<10} {r2_str:>8} {rmse_str:>10} {n:>10}")
    
    print("=" * 50)
