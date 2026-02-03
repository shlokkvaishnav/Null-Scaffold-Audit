"""Generate Publication-Quality Figures

Regenerates all pipeline figures with publication settings:
- 300 DPI
- LaTeX-style fonts  
- Colorblind-safe palettes
- PDF + PNG exports
"""

import sys
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import pandas as pd
import seaborn as sns

# Publication settings
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.serif'] = ['Times New Roman']
mpl.rcParams['font.size'] = 10
mpl.rcParams['axes.labelsize'] = 11
mpl.rcParams['axes.titlesize'] = 12
mpl.rcParams['xtick.labelsize'] = 9
mpl.rcParams['ytick.labelsize'] = 9
mpl.rcParams['legend.fontsize'] = 9
mpl.rcParams['figure.titlesize'] = 13
mpl.rcParams['pdf.fonttype'] = 42  # TrueType for better compatibility
mpl.rcParams['ps.fonttype'] = 42


def generate_all_figures(results_dir="results", figures_dir="figures", output_dir="figures/publication"):
    """Generate all publication figures."""
    results_path = Path(results_dir)
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True, parents=True)
    
    print("Generating publication-quality figures...")
    
    # Load performance data
    perf_df = pd.read_csv(results_path / "regime_performance.csv")
    
    # Colorblind-safe palette
    colors = sns.color_palette("colorblind", n_colors=len(perf_df))
    
    #=== Figure 1: Performance Summary ===
    fig, axes = plt.subplots(2, 2, figsize=(7.5, 6))
    
    # R²
    axes[0, 0].bar(perf_df['regime'], perf_df['r2'], color=colors, edgecolor='black', linewidth=0.8)
    axes[0, 0].axhline(0, color='red', linestyle='--', linewidth=1, alpha=0.7)
    axes[0, 0].set_xlabel('Regime')
    axes[0, 0].set_ylabel('R² Score')
    axes[0, 0].set_title('(a) Prediction Quality')
    axes[0, 0].grid(axis='y', alpha=0.3, linestyle='--')
    
    # RMSE
    axes[0, 1].bar(perf_df['regime'], perf_df['rmse'], color=colors, edgecolor='black', linewidth=0.8)
    axes[0, 1].set_xlabel('Regime')
    axes[0, 1].set_ylabel('RMSE (μatm)')
    axes[0, 1].set_title('(b) Prediction Error')
    axes[0, 1].grid(axis='y', alpha=0.3, linestyle='--')
    
    # Coverage
    percentages = perf_df['frac_samples'] * 100
    axes[1, 0].bar(perf_df['regime'], percentages, color=colors, edgecolor='black', linewidth=0.8)
    axes[1, 0].set_xlabel('Regime')
    axes[1, 0].set_ylabel('Coverage (%)')
    axes[1, 0].set_title('(c) Geographic Distribution')
    axes[1, 0].grid(axis='y', alpha=0.3, linestyle='--')
    
    # Sample counts
    axes[1, 1].bar(perf_df['regime'], perf_df['n_samples'], color=colors, edgecolor='black', linewidth=0.8)
    axes[1, 1].set_xlabel('Regime')
    axes[1, 1].set_ylabel('Number of Samples')
    axes[1, 1].set_title('(d) Sample Distribution')
    axes[1, 1].grid(axis='y', alpha=0.3, linestyle='--')
    
    plt.suptitle('SD-MoSE Performance Across Ocean Regimes', fontweight='bold', y=0.995)
    plt.tight_layout()
    
    # Save both formats
    plt.savefig(output_path / "figure1_performance.pdf", dpi=300, bbox_inches='tight')
    plt.savefig(output_path / "figure1_performance.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Figure 1: Performance summary")
    
    #=== Figure 2: R² vs RMSE Trade-off ===
    fig, ax = plt.subplots(figsize=(6, 5))
    
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
    
    # Annotate each point
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
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # Colorbar
    cbar = plt.colorbar(scatter, ax=ax, label='Regime ID')
    
    plt.tight_layout()
    plt.savefig(output_path / "figure2_tradeoff.pdf", dpi=300, bbox_inches='tight')
    plt.savefig(output_path / "figure2_tradeoff.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Figure 2: R² vs RMSE trade-off")
    
    print(f"\n✓ Generated {2} publication figures in {output_path}/")
    print(f"  Formats: PDF (vector) + PNG (raster)")
    print(f"  Resolution: 300 DPI")
    print(f"  Font: Times New Roman (LaTeX-compatible)")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--results', type=str, default='results')
    parser.add_argument('--output', type=str, default='figures/publication')
    args = parser.parse_args()
    
    generate_all_figures(args.results, output_dir=args.output)
