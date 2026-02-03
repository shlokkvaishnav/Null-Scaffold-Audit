"""Consolidated Visualization Module

This module consolidates scattered visualization scripts:
- scripts/viz/generate_publication_figures.py → visualization/publication.py  
- scripts/viz/plot_interactive_regime_map.py → visualization/regimes.py
- scripts/analysis/equation_sensitivity.py → visualization/equations.py

Usage:
    from climate_discovery.visualization import (
        plot_performance, plot_regime_map, plot_sensitivity,
        generate_all_figures
    )
"""

from .performance import plot_performance_summary, plot_tradeoff
from .regimes import plot_regime_distribution, plot_regime_map
from .equations import plot_sensitivity_heatmap, compute_sensitivity  
from .publication import generate_all_figures

__all__ = [
    # Performance plots
    "plot_performance_summary",
    "plot_tradeoff",
    
    # Regime plots
    "plot_regime_distribution",
    "plot_regime_map",
    
    # Equation plots
    "plot_sensitivity_heatmap",
    "compute_sensitivity",
    
    # Publication generator
    "generate_all_figures",
]
