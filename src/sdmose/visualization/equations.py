"""Equation Sensitivity Analysis

Compute and visualize how sensitive discovered equations are to each input feature.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(root / "src"))

from sdmose.models.symbolic import MixtureOfSymbolicExperts
from sdmose.config import FEATURES_EXPERT


def compute_sensitivity(experts, X, feature_names, perturbation=0.01):
    """Compute sensitivity ∂f/∂x for each feature.
    
    Args:
        experts: Fitted MixtureOfSymbolicExperts
        X: Input data
        feature_names: List of feature names
        perturbation: Perturbation amount for finite differences
        
    Returns:
        DataFrame with sensitivity scores
    """
    n_regimes = len(experts.experts)
    sensitivities = {f: [] for f in feature_names}
    
    for regime_id in range(n_regimes):
        expert = experts.experts[regime_id]
        
        if not expert.fitted_:
            sensitivities[f].append(0)
            continue
        
        for feat_idx, feat_name in enumerate(feature_names):
            # Perturb feature
            X_perturbed = X.copy()
            X_perturbed[:, feat_idx] += perturbation
            
            try:
                # Compute finite difference
                y_orig = expert.predict(X)
                y_pert = expert.predict(X_perturbed)
                
                sensitivity = np.mean(np.abs(y_pert - y_orig)) / perturbation
                sensitivities[feat_name].append(sensitivity)
            except:
                sensitivities[feat_name].append(0)
    
    # Create DataFrame
    df = pd.DataFrame(sensitivities, index=[f'Regime {i}' for i in range(n_regimes)])
    return df


def plot_sensitivity_heatmap(sensitivity_df, save_path="figures/sensitivity_heatmap.png"):
    """Plot sensitivity heatmap."""
    plt.figure(figsize=(12, 8))
    
    sns.heatmap(
        sensitivity_df.T,
        annot=True,
        fmt='.2f',
        cmap='YlOrRd',
        cbar_kws={'label': 'Sensitivity (∂fCO₂/∂x)'},
        linewidths=0.5
    )
    
    plt.title('Equation Sensitivity to Input Features', fontsize=14, fontweight='bold', pad=15)
    plt.xlabel('Ocean Regime', fontsize=12, fontweight='bold')
    plt.ylabel('Environmental Variable', fontsize=12, fontweight='bold')
    plt.tight_layout()
    
    Path(save_path).parent.mkdir(exist_ok=True, parents=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved sensitivity heatmap: {save_path}")


if __name__ == "__main__":
    print("To run sensitivity analysis, call from your pipeline after fitting experts:")
    print("  from scripts.analysis.equation_sensitivity import compute_sensitivity, plot_sensitivity_heatmap")
    print("  sensitivity = compute_sensitivity(experts, X, FEATURES_EXPERT)")
    print("  plot_sensitivity_heatmap(sensitivity)")
