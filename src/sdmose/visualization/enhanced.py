"""Enhanced Interactive Visualizations for SD-MoSE

Provides advanced visualization capabilities:
1. Interactive 3D regime maps
2. Equation complexity vs performance plots
3. Feature importance analysis
4. Residual analysis with spatial patterns
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Optional, List, Dict


def plot_3d_regime_map(
    lats: np.ndarray,
    lons: np.ndarray,
    regimes: np.ndarray,
    fco2_values: Optional[np.ndarray] = None,
    save_path: str = "figures/3d_regime_map.html"
):
    """Create interactive 3D globe showing regime distribution.
    
    Args:
        lats: Latitude values
        lons: Longitude values
        regimes: Regime assignments
        fco2_values: Optional fCO2 values for coloring
        save_path: Output HTML path
    """
    try:
        import plotly.graph_objects as go
        
        # Convert to Cartesian coordinates for 3D globe
        r = 1  # Earth radius (normalized)
        lat_rad = np.radians (lats)
        lon_rad = np.radians(lons)
        
        x = r * np.cos(lat_rad) * np.cos(lon_rad)
        y = r * np.cos(lat_rad) * np.sin(lon_rad)
        z = r * np.sin(lat_rad)
        
        # Create scatter plot
        fig = go.Figure(data=[go.Scatter3d(
            x=x, y=y, z=z,
            mode='markers',
            marker=dict(
                size=2,
                color=regimes if fco2_values is None else fco2_values,
                colorscale='Viridis' if fco2_values is None else 'RdYlBu_r',
                showscale=True,
                colorbar=dict(
                    title="Regime ID" if fco2_values is None else "fCO2 (μatm)",
                    thickness=15
                ),
                line=dict(width=0)
            ),
            text=[f"Lat: {lat:.1f}°, Lon: {lon:.1f}°<br>Regime: {r}"
                  for lat, lon, r in zip(lats, lons, regimes)],
            hoverinfo='text'
        )])
        
        # Layout
        fig.update_layout(
            title="SD-MoSE: 3D Regime Distribution",
            scene=dict(
                xaxis=dict(showbackground=False, visible=False),
                yaxis=dict(showbackground=False, visible=False),
                zaxis=dict(showbackground=False, visible=False),
                aspectmode='data'
            ),
            width=900,
            height=700
        )
        
        # Save
        Path(save_path).parent.mkdir(exist_ok=True, parents=True)
        fig.write_html(save_path)
        print(f"✓ 3D regime map saved: {save_path}")
        
    except ImportError:
        print("WARNING: plotly not installed. Install with: pip install plotly")


def plot_equation_complexity_vs_performance(
    regime_results: pd.DataFrame,
    save_path: str = "figures/complexity_vs_performance.png"
):
    """Plot equation complexity vs prediction performance.
    
    Args:
        regime_results: DataFrame with columns: regime, r2, rmse, complexity
        save_path: Output path
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # R² vs Complexity
    ax = axes[0]
    scatter = ax.scatter(
        regime_results['complexity'],
        regime_results['r2'],
        s=regime_results['n_samples'] / 50,
        c=regime_results['regime'],
        cmap='tab10',
        alpha=0.7,
        edgecolors='black',
        linewidth=1
    )
    
    # Annotate points
    for _, row in regime_results.iterrows():
        ax.annotate(
            f"R{int(row['regime'])}",
            (row['complexity'], row['r2']),
            xytext=(5, 5),
            textcoords='offset points',
            fontsize=9
        )
    
    ax.set_xlabel('Equation Complexity (nodes)', fontsize=12)
    ax.set_ylabel('R² Score', fontsize=12)
    ax.set_title('Prediction Quality vs. Equation Complexity', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0.5, color='red', linestyle='--', alpha=0.5, label='R²=0.5')
    ax.legend()
    plt.colorbar(scatter, ax=ax, label='Regime ID')
    
    # RMSE vs Complexity
    ax = axes[1]
    scatter = ax.scatter(
        regime_results['complexity'],
        regime_results['rmse'],
        s=regime_results['n_samples'] / 50,
        c=regime_results['regime'],
        cmap='tab10',
        alpha=0.7,
        edgecolors='black',
        linewidth=1
    )
    
    # Annotate
    for _, row in regime_results.iterrows():
        ax.annotate(
            f"R{int(row['regime'])}",
            (row['complexity'], row['rmse']),
            xytext=(5, 5),
            textcoords='offset points',
            fontsize=9
        )
    
    ax.set_xlabel('Equation Complexity (nodes)', fontsize=12)
    ax.set_ylabel('RMSE (μatm)', fontsize=12)
    ax.set_title('Prediction Error vs. Equation Complexity', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=ax, label='Regime ID')
    
    plt.suptitle('Interpretability-Accuracy Trade-off Analysis', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    # Save
    Path(save_path).parent.mkdir(exist_ok=True, parents=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Complexity analysis saved: {save_path}")


def plot_residual_spatial_analysis(
    lats: np.ndarray,
    lons: np.ndarray,
    residuals: np.ndarray,
    regimes: np.ndarray,
    save_path: str = "figures/residual_spatial_analysis.png"
):
    """Analyze residual patterns by regime and geography.
    
    Args:
        lats: Latitude values
        lons: Longitude values
        residuals: Prediction residuals (y_true - y_pred)
        regimes: Regime assignments
        save_path: Output path
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Spatial map of residuals
    ax = axes[0, 0]
    scatter = ax.scatter(
        lons, lats,
        c=residuals,
        cmap='RdBu_r',
        s=5,
        alpha=0.6,
        vmin=np.percentile(residuals, 5),
        vmax=np.percentile(residuals, 95)
    )
    ax.set_xlabel('Longitude', fontsize=11)
    ax.set_ylabel('Latitude', fontsize=11)
    ax.set_title('(a) Spatial Distribution of Residuals', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=ax, label='Residual (μatm)')
    
    # 2. Residuals by regime (boxplot)
    ax = axes[0, 1]
    regime_ids = np.unique(regimes)
    residual_by_regime = [residuals[regimes == r] for r in regime_ids]
    
    bp = ax.boxplot(
        residual_by_regime,
        labels=[f"R{int(r)}" for r in regime_ids],
        patch_artist=True,
        showmeans=True
    )
    
    # Color boxes
    colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(regime_ids)))
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    ax.axhline(y=0, color='red', linestyle='--', linewidth=2)
    ax.set_xlabel('Regime', fontsize=11)
    ax.set_ylabel('Residual (μatm)', fontsize=11)
    ax.set_title('(b) Residuals by Regime', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    # 3. Residual vs latitude
    ax = axes[1, 0]
    ax.hexbin(lats, residuals, gridsize=30, cmap='YlOrRd', mincnt=1)
    ax.axhline(y=0, color='blue', linestyle='--', linewidth=2)
    ax.set_xlabel('Latitude', fontsize=11)
    ax.set_ylabel('Residual (μatm)', fontsize=11)
    ax.set_title('(c) Residuals vs. Latitude', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # 4. Residual histogram by regime
    ax = axes[1, 1]
    for r in regime_ids:
        regime_resid = residuals[regimes == r]
        ax.hist(
            regime_resid,
            bins=30,
            alpha=0.5,
            label=f"R{int(r)} (n={len(regime_resid):,})",
            density=True
        )
    
    ax.axvline(x=0, color='red', linestyle='--', linewidth=2)
    ax.set_xlabel('Residual (μatm)', fontsize=11)
    ax.set_ylabel('Density', fontsize=11)
    ax.set_title('(d) Residual Distribution by Regime', fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    
    plt.suptitle('Residual Analysis: Spatial Patterns and Regime Bias', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    # Save
    Path(save_path).parent.mkdir(exist_ok=True, parents=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Residual analysis saved: {save_path}")


def plot_feature_importance_heatmap(
    feature_names: List[str],
    regime_importance: Dict[int, np.ndarray],
    save_path: str = "figures/feature_importance_heatmap.png"
):
    """Plot feature importance across regimes as heatmap.
    
    Args:
        feature_names: List of feature names
        regime_importance: Dict mapping regime_id -> importance array
        save_path: Output path
    """
    # Create matrix
    regime_ids = sorted(regime_importance.keys())
    importance_matrix = np.array([regime_importance[r] for r in regime_ids])
    
    # Normalize by row (each regime sums to 1)
    importance_matrix = importance_matrix / importance_matrix.sum(axis=1, keepdims=True)
    
    # Plot
    fig, ax = plt.subplots(figsize=(12, 6))
    
    im = ax.imshow(importance_matrix, cmap='YlOrRd', aspect='auto')
    
    # Set ticks
    ax.set_xticks(np.arange(len(feature_names)))
    ax.set_yticks(np.arange(len(regime_ids)))
    ax.set_xticklabels(feature_names, rotation=45, ha='right')
    ax.set_yticklabels([f"Regime {r}" for r in regime_ids])
    
    # Add values
    for i in range(len(regime_ids)):
        for j in range(len(feature_names)):
            text = ax.text(
                j, i, f"{importance_matrix[i, j]:.2f}",
                ha="center", va="center",
                color="white" if importance_matrix[i, j] > 0.5 else "black",
                fontsize=9
            )
    
    ax.set_title('Feature Importance by Regime', fontsize=14, fontweight='bold')
    ax.set_xlabel('Features', fontsize=12)
    ax.set_ylabel('Regimes', fontsize=12)
    
    plt.colorbar(im, ax=ax, label='Normalized Importance')
    plt.tight_layout()
    
    # Save
    Path(save_path).parent.mkdir(exist_ok=True, parents=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Feature importance heatmap saved: {save_path}")


def create_dashboard_summary(
    regime_results: pd.DataFrame,
    output_dir: str = "figures/dashboard"
):
    """Create a comprehensive visualization dashboard.
    
    Args:
        regime_results: DataFrame with regime performance metrics
        output_dir: Output directory for figures
    """
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True, parents=True)
    
    print("\n" + "="*70)
    print("GENERATING VISUALIZATION DASHBOARD")
    print("="*70)
    
    # Generate all plots
    if 'complexity' in regime_results.columns:
        plot_equation_complexity_vs_performance(
            regime_results,
            save_path=str(output_path / "complexity_vs_performance.png")
        )
    
    print("\n✓ Dashboard generation complete")
    print(f"  Location: {output_path}/")
    print("="*70)


if __name__ == "__main__":
    # Demo
    print("Enhanced Visualization Demo")
    print("="*70)
    
    # Create mock data
    np.random.seed(42)
    n_regimes = 6
    
    regime_results = pd.DataFrame({
        'regime': range(n_regimes),
        'r2': [0.08, 0.09, 0.12, 0.05, 0.41, 0.07],
        'rmse': [44.8, 30.5, 26.0, 35.6, 75.5, 49.0],
        'complexity': [5, 7, 4, 6, 12, 5],
        'n_samples': [8063, 34518, 31758, 32151, 1256, 21008]
    })
    
    # Generate visualizations
    create_dashboard_summary(regime_results)
    
    print("\n✓ Demo complete")
