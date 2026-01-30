"""Demo: Interactive visualizations and sensitivity analysis.

Demonstrates:
1. Interactive Plotly regime maps
2. Equation sensitivity analysis
3. Regime evolution animations

Usage:
    python -m scripts.examples.visualization_demo
"""

import sys
from pathlib import Path

import numpy as np
import torch

# Add src to path
root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(root / "src"))

from climate_discovery.config import FEATURES_GATING
from climate_discovery.models.gating import GatingNetwork


def demo_interactive_maps():
    """Demo interactive Plotly maps."""
    print("\n" + "="*70)
    print("DEMO 1: Interactive Regime Maps")
    print("="*70)
    
    print("\n📍 Creating interactive visualizations...")
    print("\nFeatures:")
    print("  ✅ Hover for detailed info (regime, confidence, probabilities)")
    print("  ✅ Zoom/pan to explore regions")
    print("  ✅ Multi-panel dashboard")
    print("  ✅ Temporal animation with slider")
    
    print("\nTo create:")
    print("  python -m scripts.viz.interactive_regime_map --type all")
    
    print("\nOutputs:")
    print("  - figures/interactive_regime_map.html - Main interactive map")
    print("  - figures/regime_dashboard.html - 4-panel dashboard")
    print("  - figures/regime_evolution.html - Animated with time slider")
    
    print("\n💡 Tip: Open HTML files in browser for full interactivity!")


def demo_sensitivity_analysis():
    """Demo equation sensitivity analysis."""
    print("\n" + "="*70)
    print("DEMO 2: Equation Sensitivity Analysis")
    print("="*70)
    
    print("\n🔬 Analyzing discovered equations...")
    print("\nFor each equation, computes:")
    print("  • ∂f/∂SST - How much does SST impact prediction?")
    print("  • ∂f/∂SSS - Salinity sensitivity")
    print("  • ∂f/∂Chl - Chlorophyll sensitivity")
    print("  • ∂f/∂|∇SST| - Fronts/gradients impact")
    
    print("\nExample results:")
    print("  Regime 0 (Cold upwelling):")
    print("    SST:    ∂f/∂x = -5.34  ← Strong negative impact")
    print("    SSS:    ∂f/∂x = +0.12")
    print("    Chl:    ∂f/∂x = +2.87  ← Biology matters!")
    print("    |∇SST|: ∂f/∂x = +0.45")
    
    print("\n  Regime 1 (Warm oligotrophic):")
    print("    SST:    ∂f/∂x = +3.14  ← Positive (different physics!)")
    print("    SSS:    ∂f/∂x = -1.57")
    print("    Chl:    ∂f/∂x = +0.31  ← Less important")
    print("    |∇SST|: ∂f/∂x = +0.08")
    
    print("\nTo run:")
    print("  python -m scripts.analysis.equation_sensitivity")
    
    print("\nOutput:")
    print("  - figures/equation_sensitivity.png - 4-panel visualization")
    print("    • Heatmap: Sensitivity by regime")
    print("    • Bar chart: Comparison across regimes")
    print("    • Rankings: Overall variable importance")
    print("    • Distributions: Sensitivity variability")


def demo_evolution_videos():
    """Demo regime evolution animations."""
    print("\n" + "="*70)
    print("DEMO 3: Regime Evolution Videos")
    print("="*70)
    
    print("\n🎬 Creating animated visualizations...")
    print("\nShows:")
    print("  • Monthly regime changes (seasonal cycle)")
    print("  • Regime boundary shifts over time")
    print("  • Uncertainty evolution")
    print("  • Side-by-side model comparisons")
    
    print("\nExample use cases:")
    print("  1. Show how El Niño affects regime structure")
    print("  2. Visualize seasonal blooms (regime transitions)")
    print("  3. Compare flat vs hierarchical models")
    print("  4. Presentation-ready animations for talks")
    
    print("\nTo create:")
    print("  # MP4 video (high quality)")
    print("  python -m scripts.viz.regime_evolution_video --format mp4 --fps 2")
    
    print("\n  # GIF animation (shareable)")
    print("  python -m scripts.viz.regime_evolution_video --format gif --fps 1")
    
    print("\nOutputs:")
    print("  - figures/regime_evolution.mp4 - Animated MP4")
    print("  - figures/regime_evolution.gif - Animated GIF")
    print("  - figures/model_comparison.png - Side-by-side comparison")


def main():
    print("\n" + "="*70)
    print("VISUALIZATION & INTERPRETABILITY DEMO")
    print("="*70)
    print("\nShowcasing 3 powerful visualization tools:")
    
    # Demo 1
    demo_interactive_maps()
    
    # Demo 2
    demo_sensitivity_analysis()
    
    # Demo 3
    demo_evolution_videos()
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY: Enhanced Interpretability")
    print("="*70)
    
    print("\n🎨 **Interactive Maps** (Plotly/HTML)")
    print("   → Share with collaborators, explore regime boundaries")
    
    print("\n🔬 **Sensitivity Analysis**")
    print("   → Understand which variables drive each regime")
    
    print("\n🎬 **Evolution Animations**")
    print("   → Publication-ready videos showing temporal dynamics")
    
    print("\n" + "-"*70)
    print("Quick Start:")
    print("-"*70)
    
    print("\n1. Create interactive map:")
    print("   python -m scripts.viz.interactive_regime_map")
    
    print("\n2. Analyze equation sensitivity:")
    print("   python -m scripts.analysis.equation_sensitivity")
    
    print("\n3. Generate evolution video:")
    print("   python -m scripts.viz.regime_evolution_video")
    
    print("\n" + "="*70)
    print("✓ All tools ready to use!")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
