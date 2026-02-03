"""Performance visualization module - extracted from generate_publication_figures.py"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


def plot_performance_summary(perf_df, complexities=None, save_path="figures/performance_summary.png"):
    """Plot R², RMSE, coverage, and complexity for all regimes."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle('SD-MoSE Performance by Ocean Regime', fontsize=14, fontweight='bold')
    
    n_regimes = len(perf_df)
    colors = plt.cm.viridis(np.linspace(0.2, 0.9, n_regimes))
    
    # R² scores
    axes[0, 0].bar(perf_df['regime'], perf_df['r2'], color=colors, edgecolor='black', linewidth=0.5)
    axes[0, 0].axhline(y=0, color='red', linestyle='--', alpha=0.5)
    axes[0, 0].set_xlabel('Regime')
    axes[0, 0].set_ylabel('R² Score')
    axes[0, 0].set_title('(a) Prediction Quality')
    axes[0, 0].grid(axis='y', alpha=0.3)
    
    # RMSE
    axes[0, 1].bar(perf_df['regime'], perf_df['rmse'], color=colors, edgecolor='black', linewidth=0.5)
    axes[0, 1].set_xlabel('Regime')
    axes[0, 1].set_ylabel('RMSE (μatm)')
    axes[0, 1].set_title('(b) Prediction Error')
    axes[0, 1].grid(axis='y', alpha=0.3)
    
    # Coverage
    percentages = perf_df['frac_samples'] * 100
    bars = axes[1, 0].bar(perf_df['regime'], percentages, color=colors, edgecolor='black', linewidth=0.5)
    axes[1, 0].set_xlabel('Regime')
    axes[1, 0].set_ylabel('Coverage (%)')
    axes[1, 0].set_title('(c) Geographic Distribution')
    axes[1, 0].grid(axis='y', alpha=0.3)
    
    # Complexity
    if complexities:
        axes[1, 1].bar(range(n_regimes), complexities, color=colors, edgecolor='black', linewidth=0.5)
    else:
        axes[1, 1].bar(perf_df['regime'], [4]*n_regimes, color=colors, edgecolor='black', linewidth=0.5)
    axes[1, 1].set_xlabel('Regime')
    axes[1, 1].set_ylabel('Equation Complexity')
    axes[1, 1].set_title('(d) Model Interpretability')
    axes[1, 1].grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    Path(save_path).parent.mkdir(exist_ok=True, parents=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {save_path}")


def plot_tradeoff(perf_df, save_path="figures/r2_rmse_tradeoff.png"):
    """Plot R² vs RMSE trade-off with regime IDs."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    scatter = ax.scatter(
        perf_df['r2'],
        perf_df['rmse'],
        s=perf_df['n_samples'] / 100,
        c=perf_df['regime'],
        cmap='tab10',
        edgecolors='black',
        linewidth=1,
        alpha=0.7
    )
    
    # Annotate
    for idx, row in perf_df.iterrows():
        ax.annotate(
            f"R{int(row['regime'])}",
            (row['r2'], row['rmse']),
            xytext=(5, 5),
            textcoords='offset points',
            fontsize=8
        )
    
    ax.set_xlabel('R² Score')
    ax.set_ylabel('RMSE (μatm)')
    ax.set_title('Prediction Quality vs. Error Trade-off', fontweight='bold')
    ax.grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=ax, label='Regime ID')
    
    plt.tight_layout()
    Path(save_path).parent.mkdir(exist_ok=True, parents=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {save_path}")
