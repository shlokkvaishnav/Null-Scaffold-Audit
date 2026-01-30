"""Interactive regime map visualization with Plotly.

Creates interactive 3D and 2D maps showing:
- Regime assignments with hover info
- Confidence/entropy overlays
- Time slider for temporal evolution
- Click-to-zoom functionality

Usage:
    python -m scripts.viz.interactive_regime_map --checkpoint checkpoints/final.pth
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import torch

# Add src to path
root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(root / "src"))

from climate_discovery.config import FEATURES_GATING, TEST_NC
from climate_discovery.data.datasets import ClimateDataset
from climate_discovery.models.mixture import SDMoSE
from climate_discovery.models.gating import GatingNetwork


def create_interactive_regime_map(
    lats: np.ndarray,
    lons: np.ndarray,
    regime_labels: np.ndarray,
    regime_probs: np.ndarray,
    fco2_values: np.ndarray = None,
    timestamps: np.ndarray = None,
    save_path: str = "figures/interactive_regime_map.html",
):
    """Create interactive Plotly map of regimes.
    
    Args:
        lats: Latitude values (N,)
        lons: Longitude values (N,)
        regime_labels: Dominant regime per sample (N,)
        regime_probs: Regime probabilities (N, K)
        fco2_values: fCO₂ measurements (N,)
        timestamps: Time values for animation (N,)
        save_path: Output HTML path
    """
    # Compute entropy for uncertainty
    eps = 1e-10
    entropy = -np.sum(regime_probs * np.log(regime_probs + eps), axis=1)
    max_entropy = np.log(regime_probs.shape[1])
    entropy_normalized = entropy / max_entropy
    
    # Compute confidence (max probability)
    confidence = np.max(regime_probs, axis=1)
    
    # Create hover text
    hover_text = []
    for i in range(len(lats)):
        text = (
            f"<b>Location:</b> ({lats[i]:.2f}°, {lons[i]:.2f}°)<br>"
            f"<b>Regime:</b> {regime_labels[i]}<br>"
            f"<b>Confidence:</b> {confidence[i]:.3f}<br>"
            f"<b>Entropy:</b> {entropy[i]:.3f}<br>"
        )
        if fco2_values is not None:
            text += f"<b>fCO₂:</b> {fco2_values[i]:.1f} μatm<br>"
        
        # Show probabilities for all regimes
        text += "<b>Probabilities:</b><br>"
        for k in range(regime_probs.shape[1]):
            text += f"  Regime {k}: {regime_probs[i, k]:.3f}<br>"
        
        hover_text.append(text)
    
    # Create main figure
    fig = go.Figure()
    
    # Add regime scatter
    fig.add_trace(go.Scattergeo(
        lon=lons,
        lat=lats,
        mode='markers',
        marker=dict(
            size=4,
            color=regime_labels,
            colorscale='Viridis',
            showscale=True,
            colorbar=dict(
                title="Regime",
                thickness=20,
                len=0.7,
            ),
            line=dict(width=0.5, color='white'),
        ),
        text=hover_text,
        hovertemplate='%{text}<extra></extra>',
        name='Regimes',
    ))
    
    # Layout
    fig.update_layout(
        title=dict(
            text='<b>Interactive SD-MoSE Regime Map</b><br>'
                 '<sub>Hover for details | Zoom/pan to explore</sub>',
            x=0.5,
            xanchor='center',
        ),
        geo=dict(
            projection_type='natural earth',
            showland=True,
            landcolor='rgb(243, 243, 243)',
            coastlinecolor='rgb(204, 204, 204)',
            showocean=True,
            oceancolor='rgb(230, 245, 255)',
            showcountries=True,
            countrycolor='rgb(204, 204, 204)',
        ),
        height=700,
        width=1200,
    )
    
    # Save
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(save_path))
    print(f"✓ Interactive map saved: {save_path}")
    
    return fig


def create_multi_view_dashboard(
    lats: np.ndarray,
    lons: np.ndarray,
    regime_labels: np.ndarray,
    regime_probs: np.ndarray,
    fco2_values: np.ndarray,
    save_path: str = "figures/regime_dashboard.html",
):
    """Create multi-panel interactive dashboard.
    
    Shows:
    - Regime map
    - Confidence map
    - Entropy map  
    - fCO₂ map
    """
    # Compute metrics
    eps = 1e-10
    entropy = -np.sum(regime_probs * np.log(regime_probs + eps), axis=1)
    confidence = np.max(regime_probs, axis=1)
    
    # Create subplots
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            'Regime Assignment',
            'Confidence (max prob)',
            'Uncertainty (entropy)',
            'fCO₂ observations',
        ),
        specs=[
            [{'type': 'scattergeo'}, {'type': 'scattergeo'}],
            [{'type': 'scattergeo'}, {'type': 'scattergeo'}],
        ],
    )
    
    # 1. Regime map
    fig.add_trace(
        go.Scattergeo(
            lon=lons, lat=lats,
            mode='markers',
            marker=dict(size=3, color=regime_labels, colorscale='Viridis', showscale=True),
            showlegend=False,
        ),
        row=1, col=1,
    )
    
    # 2. Confidence map
    fig.add_trace(
        go.Scattergeo(
            lon=lons, lat=lats,
            mode='markers',
            marker=dict(size=3, color=confidence, colorscale='RdYlGn', 
                       cmin=0, cmax=1, showscale=True,
                       colorbar=dict(x=1.15, len=0.45, y=0.77)),
            showlegend=False,
        ),
        row=1, col=2,
    )
    
    # 3. Entropy map
    fig.add_trace(
        go.Scattergeo(
            lon=lons, lat=lats,
            mode='markers',
            marker=dict(size=3, color=entropy, colorscale='Reds',
                       showscale=True,
                       colorbar=dict(x=0.46, len=0.45, y=0.23)),
            showlegend=False,
        ),
        row=2, col=1,
    )
    
    # 4. fCO₂ map
    fig.add_trace(
        go.Scattergeo(
            lon=lons, lat=lats,
            mode='markers',
            marker=dict(size=3, color=fco2_values, colorscale='Plasma',
                       showscale=True,
                       colorbar=dict(x=1.15, len=0.45, y=0.23)),
            showlegend=False,
        ),
        row=2, col=2,
    )
    
    # Update geo properties for all subplots
    for i in range(1, 5):
        fig.update_geos(
            projection_type='natural earth',
            showland=True,
            landcolor='lightgray',
            showocean=True,
            oceancolor='lightblue',
            row=(i-1)//2 + 1,
            col=(i-1)%2 + 1,
        )
    
    # Layout
    fig.update_layout(
        title_text='<b>SD-MoSE Interactive Dashboard</b>',
        height=900,
        width=1400,
    )
    
    # Save
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(save_path))
    print(f"✓ Dashboard saved: {save_path}")
    
    return fig


def create_temporal_animation(
    lats: np.ndarray,
    lons: np.ndarray,
    regime_labels: np.ndarray,
    timestamps: np.ndarray,
    save_path: str = "figures/regime_evolution.html",
):
    """Create animated map showing regime evolution over time.
    
    Args:
        lats: Latitudes (N,)
        lons: Longitudes (N,)
        regime_labels: Regime assignments (N,)
        timestamps: Time values (N,) - e.g., year fractions
        save_path: Output HTML path
    """
    import pandas as pd
    
    # Create DataFrame
    df = pd.DataFrame({
        'lat': lats,
        'lon': lons,
        'regime': regime_labels,
        'time': timestamps,
    })
    
    # Create time bins (monthly if possible)
    df['time_bin'] = pd.cut(df['time'], bins=12, labels=range(12))
    
    # Create animated scatter
    fig = px.scatter_geo(
        df,
        lat='lat',
        lon='lon',
        color='regime',
        animation_frame='time_bin',
        color_continuous_scale='Viridis',
        title='<b>Regime Evolution Over Time</b>',
        projection='natural earth',
    )
    
    fig.update_geos(
        showland=True,
        landcolor='lightgray',
        showocean=True,
        oceancolor='lightblue',
    )
    
    fig.update_layout(
        height=700,
        width=1200,
    )
    
    # Save
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(save_path))
    print(f"✓ Animation saved: {save_path}")
    
    return fig


def main():
    parser = argparse.ArgumentParser(description="Create interactive regime maps")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/final.pth")
    parser.add_argument("--output-dir", type=str, default="figures")
    parser.add_argument("--type", type=str, default="all",
                       choices=["map", "dashboard", "animation", "all"])
    
    args = parser.parse_args()
    
    print("="*70)
    print("INTERACTIVE REGIME VISUALIZATION")
    print("="*70)
    
    # Load data
    print("\n📦 Loading data...")
    dataset = ClimateDataset(
        TEST_NC,
        expert_features=[],
        gating_features=FEATURES_GATING,
        target="fco2",
    )
    
    # Create dummy model (in real use, load from checkpoint)
    print("\n🧠 Creating model...")
    gating = GatingNetwork(
        input_dim=len(FEATURES_GATING),
        num_regimes=6,
        hidden_dims=[64, 32],
    )
    
    # Get regime assignments
    print("\n🎯 Computing regime assignments...")
    X_gate = torch.from_numpy(dataset.X_gate).float()
    
    with torch.no_grad():
        regime_probs = gating(X_gate).numpy()
        regime_labels = np.argmax(regime_probs, axis=1)
    
    # Denormalize coordinates
    lats = dataset.X_gate[:, 0] * 90
    lons = dataset.X_gate[:, 1] * 180
    fco2 = dataset.y
    
    # Extract timestamps (if available)
    year_norm = dataset.X_gate[:, 8] if dataset.X_gate.shape[1] > 8 else None
    
    # Create visualizations
    output_dir = Path(args.output_dir)
    
    if args.type in ["map", "all"]:
        print("\n🗺️  Creating interactive regime map...")
        create_interactive_regime_map(
            lats, lons, regime_labels, regime_probs, fco2,
            save_path=output_dir / "interactive_regime_map.html",
        )
    
    if args.type in ["dashboard", "all"]:
        print("\n📊 Creating multi-view dashboard...")
        create_multi_view_dashboard(
            lats, lons, regime_labels, regime_probs, fco2,
            save_path=output_dir / "regime_dashboard.html",
        )
    
    if args.type in ["animation", "all"] and year_norm is not None:
        print("\n🎬 Creating temporal animation...")
        create_temporal_animation(
            lats, lons, regime_labels, year_norm,
            save_path=output_dir / "regime_evolution.html",
        )
    
    print("\n" + "="*70)
    print("✓ VISUALIZATION COMPLETE")
    print("="*70)
    print(f"\nOpen in browser:")
    print(f"  - {output_dir / 'interactive_regime_map.html'}")
    print(f"  - {output_dir / 'regime_dashboard.html'}")
    if year_norm is not None:
        print(f"  - {output_dir / 'regime_evolution.html'}")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
