"""Interactive Plotly Regime Map

Generates an interactive 3D globe visualization showing ocean regime assignments.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

# Add src to path
root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(root / "src"))

from sdmose.config import TRAIN_NC, FEATURES_GATING, TARGET
from sdmose.data.datasets import ClimateDataset


def create_interactive_regime_map(data_path=None, output_path="figures/interactive_regime_map.html"):
    """Create interactive Plotly map of ocean regimes.
    
    Args:
        data_path: Path to results CSV with regime assignments
        output_path: Where to save HTML file
    """
    print("Creating interactive regime map...")
    
   # Load data
    if data_path is None:
        # Load from dataset
        dataset = ClimateDataset(
            TRAIN_NC,
            expert_features=[],  # Not needed
            gating_features=FEATURES_GATING,
            target=TARGET,
            drop_nan=True
        )
        df = dataset.get_dataframe()
    else:
        df = pd.DataFrame(data_path)
    
    # Create 3D scatter on globe
    fig = go.Figure()
    
    # Color palette for regimes
    colors = px.colors.qualitative.Plotly
    
    # Get unique regimes
    if 'regime' not in df.columns:
        print("Warning: No 'regime' column found. Run pipeline first.")
        return
    
    regimes = sorted(df['regime'].unique())
    
    for regime_id in regimes:
        regime_data = df[df['regime'] == regime_id]
        
        fig.add_trace(go.Scattergeo(
            lon=regime_data['lon'],
            lat=regime_data['lat'],
            mode='markers',
            marker=dict(
                size=3,
                color=colors[regime_id % len(colors)],
                opacity=0.6
            ),
            name=f'Regime {regime_id}',
            text=[f'Regime {regime_id}<br>Lat: {lat:.2f}<br>Lon: {lon:.2f}' 
                  for lat, lon in zip(regime_data['lat'], regime_data['lon'])],
            hovertemplate='%{text}<extra></extra>'
        ))
    
    fig.update_layout(
        title='Ocean Carbon Regimes - Global Distribution',
        geo=dict(
            projection_type='orthographic',
            showland=True,
            landcolor='rgb(243, 243, 243)',
            coastlinecolor='rgb(204, 204, 204)',
            showocean=True,
            oceancolor='rgb(230, 245, 255)',
            showcountries=True,
        ),
        height=800,
        width=1200
    )
    
    # Save
    output = Path(output_path)
    output.parent.mkdir(exist_ok=True, parents=True)
    fig.write_html(str(output))
    
    print(f"✓ Saved interactive map: {output}")
    print(f"  Open in browser to explore {len(regimes)} regimes across {len(df):,} ocean points")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, help='Path to data with regime assignments')
    parser.add_argument('--output', type=str, default='figures/interactive_regime_map.html')
    args = parser.parse_args()
    
    create_interactive_regime_map(args.data, args.output)
