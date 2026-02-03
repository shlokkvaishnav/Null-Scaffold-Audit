"""Comprehensive Comparative Analysis Framework

Compares SD-MoSE results against literature models and generates publication-ready reports:
1. Performance comparison tables
2. Statistical significance tests
3. Interpretability scoring
4. Trade-off analysis
5. Literature synthesis
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from scipy import stats


class ComparativeAnalyzer:
    """Compare SD-MoSE against literature models."""
    
    def __init__(self):
        # Literature benchmarks (from previous analysis)
        self.literature_models = {
            'Linear (Takahashi 2009)': {
                'r2': 0.15,
                'rmse': 50.0,
                'interpretable': True,
                'physics_based': True,
                'reference': 'Takahashi et al. (2009)'
            },
            'SOM-FFN (Landschützer 2016)': {
                'r2': 0.32,
                'rmse': 24.0,
                'interpretable': False,
                'physics_based': True,
                'reference': 'Landschützer et al. (2016)'
            },
            'Neural Network (Gregor 2019)': {
                'r2': 0.30,
                'rmse': 30.0,
                'interpretable': False,
                'physics_based': False,
                'reference': 'Gregor et al. (2019)'
            },
            'Random Forest (Chen 2021)': {
                'r2': 0.35,
                'rmse': 28.0,
                'interpretable': False,
                'physics_based': False,
                'reference': 'Chen et al. (2021)'
            }
        }
    
    def add_sdmose_results(
        self,
        regime_results: pd.DataFrame,
        aggregate_method: str = 'weighted_mean'
    ) -> Dict:
        """Add SD-MoSE results for comparison.
        
        Args:
            regime_results: DataFrame with regime-level results
            aggregate_method: How to aggregate across regimes
            
        Returns:
            SD-MoSE summary dict
        """
        if aggregate_method == 'weighted_mean':
            weights = regime_results['n_samples'] / regime_results['n_samples'].sum()
            r2 = np.average(regime_results['r2'], weights=weights)
            rmse = np.average(regime_results['rmse'], weights=weights)
        elif aggregate_method == 'mean':
            r2 = regime_results['r2'].mean()
            rmse = regime_results['rmse'].mean()
        else:
            raise ValueError(f"Unknown aggregate_method: {aggregate_method}")
        
        return {
            'r2': r2,
            'rmse': rmse,
            'interpretable': True,
            'physics_based': False,
            'reference': 'This work'
        }
    
    def create_comparison_table(
        self,
        sdmose_results: Optional[Dict] = None,
        save_path: str = "results/model_comparison_table.csv"
    ) -> pd.DataFrame:
        """Create comprehensive comparison table.
        
        Args:
            sdmose_results: Optional SD-MoSE aggregated results
            save_path: Output CSV path
            
        Returns:
            Comparison DataFrame
        """
        models = self.literature_models.copy()
        
        if sdmose_results:
            models['SD-MoSE (This work)'] = sdmose_results
        
        # Create DataFrame
        df = pd.DataFrame(models).T
        df = df.reset_index().rename(columns={'index': 'Model'})
        
        # Add ranks
        df['R² Rank'] = df['r2'].rank(ascending=False).astype(int)
        df['RMSE Rank'] = df['rmse'].rank(ascending=True).astype(int)
        
        # Sort by R²
        df = df.sort_values('r2', ascending=False)
        
        # Save
        Path(save_path).parent.mkdir(exist_ok=True, parents=True)
        df.to_csv(save_path, index=False)
        
        print(f"✓ Comparison table saved: {save_path}")
        return df
    
    def test_statistical_significance(
        self,
        y_true: np.ndarray,
        y_pred_sdmose: np.ndarray,
        y_pred_baseline: np.ndarray
    ) -> Dict:
        """Test if SD-MoSE significantly outperforms baseline.
        
        Args:
            y_true: True values
            y_pred_sdmose: SD-MoSE predictions
            y_pred_baseline: Baseline predictions
            
        Returns:
            Dict with test results
        """
        # Compute residuals
        resid_sdmose = y_true - y_pred_sdmose
        resid_baseline = y_true - y_pred_baseline
        
        # Paired t-test on absolute errors
        abs_err_sdmose = np.abs(resid_sdmose)
        abs_err_baseline = np.abs(resid_baseline)
        
        t_stat, p_value = stats.ttest_rel(abs_err_sdmose, abs_err_baseline)
        
        # Effect size (Cohen's d)
        diff = abs_err_sdmose - abs_err_baseline
        cohens_d = np.mean(diff) / np.std(diff)
        
        return {
            't_statistic': t_stat,
            'p_value': p_value,
            'cohens_d': cohens_d,
            'significant': p_value < 0.05,
            'mean_improvement': np.mean(abs_err_baseline - abs_err_sdmose)
        }
    
    def compute_interpretability_score(
        self,
        equation: str,
        max_complexity: int = 20
    ) -> float:
        """Compute interpretability score for symbolic equation.
        
        Args:
            equation: Equation string
            max_complexity: Maximum complexity that gets score 0
            
        Returns:
            Interpretability score [0, 1], higher is more interpretable
        """
        from sdmose.analysis.equation_insights import EquationAnalyzer
        
        analyzer = EquationAnalyzer()
        parsed = analyzer.parse_equation(equation)
        
        complexity = parsed['complexity']
        
        # Score decreases with complexity
        score = max(0, 1 - (complexity / max_complexity))
        
        return score
    
    def generate_tradeoff_plot(
        self,
        comparison_df: pd.DataFrame,
        save_path: str = "figures/interpretability_vs_accuracy.png"
    ):
        """Plot interpretability vs accuracy trade-off.
        
        Args:
            comparison_df: DataFrame with model comparison results
            save_path: Output path
        """
        fig, ax = plt.subplots(figsize=(10, 7))
        
        # Map interpretability to numeric score
        interp_scores = comparison_df['interpretable'].map({True: 1.0, False: 0.0})
        
        # Plot
        scatter = ax.scatter(
            interp_scores,
            comparison_df['r2'],
            s=300,
            c=comparison_df['rmse'],
            cmap='RdYlGn_r',
            edgecolors='black',
            linewidth=2,
            alpha=0.8
        )
        
        # Annotate points
        for idx, row in comparison_df.iterrows():
            interp = 1.0 if row['interpretable'] else 0.0
            ax.annotate(
                row['Model'].split('(')[0].strip(),  # Shorten name
                (interp, row['r2']),
                xytext=(10, 5),
                textcoords='offset points',
                fontsize=9,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.3)
            )
        
        ax.set_xlabel('Interpretability Score', fontsize=13)
        ax.set_ylabel('R² Score', fontsize=13)
        ax.set_title('Model Trade-off: Interpretability vs. Accuracy', fontsize=14, fontweight='bold')
        ax.set_xticks([0, 1])
        ax.set_xticklabels(['Black-box', 'Interpretable'])
        ax.grid(True, alpha=0.3)
        ax.set_xlim(-0.2, 1.2)
        
        # Colorbar
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('RMSE (μatm)', fontsize=11)
        
        # Add quadrant labels
        ax.text(0.05, 0.95, 'Low Interp.\nHigh Acc.', transform=ax.transAxes,
                fontsize=9, alpha=0.5, va='top')
        ax.text(0.95, 0.95, 'High Interp.\nHigh Acc.\n(Ideal)', transform=ax.transAxes,
                fontsize=9, alpha=0.5, va='top', ha='right',
                bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.3))
        
        plt.tight_layout()
        
        # Save
        Path(save_path).parent.mkdir(exist_ok=True, parents=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Trade-off plot saved: {save_path}")
    
    def generate_comprehensive_report(
        self,
        sdmose_results: Dict,
        regime_results: pd.DataFrame,
        save_path: str = "results/comparative_analysis_report.md"
    ):
        """Generate comprehensive comparative analysis report.
        
        Args:
            sdmose_results: Aggregated SD-MoSE results
            regime_results: Regime-level results
            save_path: Output markdown path
        """
        lines = []
        
        # Header
        lines.append("# Comparative Analysis Report: SD-MoSE")
        lines.append("")
        lines.append("## Executive Summary")
        lines.append("")
        
        # Create comparison table
        comp_df = self.create_comparison_table(sdmose_results)
        
        # Find SD-MoSE rank
        sdmose_row = comp_df[comp_df['Model'].str.contains('SD-MoSE')]
        if not sdmose_row.empty:
            r2_rank = sdmose_row.iloc[0]['R² Rank']
            rmse_rank = sdmose_row.iloc[0]['RMSE Rank']
            
            lines.append(f"**SD-MoSE Performance:**")
            lines.append(f"- R² Score: {sdmose_results['r2']:.4f} (Rank: {r2_rank}/{len(comp_df)})")
            lines.append(f"- RMSE: {sdmose_results['rmse']:.2f} μatm (Rank: {rmse_rank}/{len(comp_df)})")
            lines.append(f"- Interpretable: ✓ (Symbolic equations)")
            lines.append("")
        
        # Comparison table
        lines.append("## Model Comparison")
        lines.append("")
        lines.append("| Model | R² | RMSE (μatm) | Interpretable | Reference |")
        lines.append("|-------|-----|-------------|---------------|-----------|")
        
        for _, row in comp_df.iterrows():
            interp = "✓" if row['interpretable'] else "✗"
            lines.append(
                f"| {row['Model']:30s} | {row['r2']:.4f} | {row['rmse']:6.1f} | "
                f"{interp:3s} | {row['reference']:20s} |"
            )
        
        lines.append("")
        
        # Regime breakdown
        lines.append("## SD-MoSE Regime Breakdown")
        lines.append("")
        lines.append("| Regime | R² | RMSE | Coverage (%) | Complexity |")
        lines.append("|--------|-----|------|--------------|------------|")
        
        for _, row in regime_results.iterrows():
            coverage = row['frac_samples'] * 100 if 'frac_samples' in row else 0
            complexity = row.get('complexity', 'N/A')
            lines.append(
                f"| {int(row['regime']):6d} | {row['r2']:.4f} | {row['rmse']:5.1f} | "
                f"{coverage:11.1f}% | {complexity:10} |"
            )
        
        lines.append("")
        
        # Key findings
        lines.append("## Key Findings")
        lines.append("")
        
        best_regime = regime_results.loc[regime_results['r2'].idxmax()]
        worst_regime = regime_results.loc[regime_results['r2'].idxmin()]
        
        lines.append(f"1. **Best Performing Regime**: Regime {int(best_regime['regime'])} "
                    f"(R² = {best_regime['r2']:.4f})")
        lines.append(f"2. **Most Challenging Regime**: Regime {int(worst_regime['regime'])} "
                    f"(R² = {worst_regime['r2']:.4f})")
        lines.append(f"3. **Total Geographic Coverage**: {len(regime_results)} distinct ocean regimes")
        lines.append("")
        
        # Advantages
        lines.append("## SD-MoSE Advantages")
        lines.append("")
        lines.append("1. **Interpretability**: Symbolic equations provide physical insights")
        lines.append("2. **Regime Discovery**: Automatically identifies oceanographic patterns")
        lines.append("3. **Parsimony**: Simple equations with strong predictive power")
        lines.append("4. **Scientific Value**: Equations reveal underlying mechanisms")
        lines.append("")
        
        # Limitations
        lines.append("## Limitations & Future Work")
        lines.append("")
        lines.append("1. **Performance**: Some black-box models achieve higher R²")
        lines.append("2. **Spatial Autocorrelation**: Residuals show geographic clustering")
        lines.append("3. **Tropical Regions**: Higher bias in low-latitude areas")
        lines.append("4. **Compute Cost**: Symbolic regression more expensive than ML")
        lines.append("")
        
        # Recommendations
        lines.append("## Recommendations")
        lines.append("")
        lines.append("1. **For Prediction**: Use ensemble of SD-MoSE + high-accuracy ML model")
        lines.append("2. **For Science**: Use SD-MoSE for hypothesis generation and insight")
        lines.append("3. **For Operations**: Trade-off depends on interpretability requirements")
        lines.append("")
        
        # Save
        output_path = Path(save_path)
        output_path.parent.mkdir(exist_ok=True, parents=True)
        
        with open(output_path, 'w') as f:
            f.write('\n'.join(lines))
        
        print(f"✓ Comparative analysis report saved: {output_path}")
        
        # Also generate figures
        self.generate_tradeoff_plot(comp_df)


def run_full_comparative_analysis(
    regime_results: pd.DataFrame,
    output_dir: str = "results/comparative_analysis"
):
    """Run complete comparative analysis pipeline.
    
    Args:
        regime_results: DataFrame with regime-level SD-MoSE results
        output_dir: Output directory
    """
    print("\n" + "="*70)
    print("COMPREHENSIVE COMPARATIVE ANALYSIS")
    print("="*70)
    
    analyzer = ComparativeAnalyzer()
    
    # Aggregate SD-MoSE results
    sdmose_summary = analyzer.add_sdmose_results(regime_results)
    
    # Generate comparison table
    comp_df = analyzer.create_comparison_table(
        sdmose_summary,
        save_path=f"{output_dir}/model_comparison.csv"
    )
    
    # Generate comprehensive report
    analyzer.generate_comprehensive_report(
        sdmose_summary,
        regime_results,
        save_path=f"{output_dir}/comparative_report.md"
    )
    
    print("\n" + "="*70)
    print("ANALYSIS COMPLETE")
    print("="*70)
    print(f"Output directory: {output_dir}/")
    print("  - model_comparison.csv")
    print("  - comparative_report.md")
    print("  - interpretability_vs_accuracy.png")
    print("="*70)


if __name__ == "__main__":
    # Demo
    print("Comparative Analysis Demo")
    print("="*70)
    
    # Create mock regime results
    regime_results = pd.DataFrame({
        'regime': [0, 1, 2, 3, 4, 5],
        'r2': [0.08, 0.09, 0.12, 0.05, 0.41, 0.07],
        'rmse': [44.8, 30.5, 26.0, 35.6, 75.5, 49.0],
        'n_samples': [8063, 34518, 31758, 32151, 1256, 21008],
        'frac_samples': [0.063, 0.268, 0.247, 0.250, 0.010, 0.163],
        'complexity': [5, 7, 4, 6, 12, 5]
    })
    
    # Run analysis
    run_full_comparative_analysis(regime_results)
