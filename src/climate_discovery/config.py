"""Central configuration for SD-MoSE pipeline.

Paper: Soft-Dynamic Mixture of Symbolic Experts for Interpretable Air-Sea CO₂ Laws
Reference: Vaishnav (2026)

Scientific Background:
- Air-sea CO₂ flux governed by: solubility (SST), carbonate chemistry (SSS),
  biological activity (Chl-a), circulation (lat/lon), and seasonality
- Soft regimes represent overlapping ocean provinces with fuzzy boundaries
- Ensemble validation ensures discovered regimes are robust, not artifacts
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np


# =============================================================================
# PATH CONFIGURATION
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
FIGURE_DIR = PROJECT_ROOT / "figures"
RESULTS_DIR = PROJECT_ROOT / "results"

# Ensure directories exist
for directory in [RAW_DIR, PROCESSED_DIR, CHECKPOINT_DIR, FIGURE_DIR, RESULTS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)


# =============================================================================
# DATA FILES (with fallback logic)
# =============================================================================

# SOCAT v2025 gridded monthly product (NOAA NCEI dataset 0304549)
# Fallback to v2024 if v2025 not available
SOCAT_VERSIONS = ["SOCATv2025_tracks_gridded_monthly.nc", 
                  "SOCATv2024_tracks_gridded_monthly.nc"]
SOCAT_PATH = None
for version in SOCAT_VERSIONS:
    candidate = RAW_DIR / version
    if candidate.exists():
        SOCAT_PATH = candidate
        break
if SOCAT_PATH is None:
    SOCAT_PATH = RAW_DIR / SOCAT_VERSIONS[0]  # Default to latest

# Copernicus Marine Service Chlorophyll-a (0.25° monthly)
CHL_FILENAME = "cmems_mod_glo_bgc_my_0.25deg_P1M-m_chl.nc"
CHL_PATH = RAW_DIR / CHL_FILENAME

# Processed datasets
TRAIN_NC = PROCESSED_DIR / "train_dataset.nc"
TEST_NC = PROCESSED_DIR / "test_dataset.nc"
FUSED_NC = PROCESSED_DIR / "climate_fused_dataset.nc"
SCALERS_NC = PROCESSED_DIR / "scalers.nc"


# =============================================================================
# TEMPORAL CONFIGURATION
# =============================================================================

# Training period: 2015-2021 (7 years, ~84 months)
# Test period: 2022-2024 (3 years, ~36 months)
# Rationale: Test on recent years to evaluate generalization to novel climate states
START_YEAR = 2015
END_YEAR = 2024
SPLIT_YEAR = 2022

TRAIN_START = f"{START_YEAR}-01-01"
TRAIN_END = f"{SPLIT_YEAR - 1}-12-31"
TEST_START = f"{SPLIT_YEAR}-01-01"
TEST_END = f"{END_YEAR}-12-31"


# =============================================================================
# FEATURE DEFINITIONS
# =============================================================================

# Physical drivers (thermodynamic + circulation)
FEATURES_PHYS = ["sst", "sss"]

# Biological drivers
FEATURES_BIO = ["log_chl"]  # log-transform handles skewness, chlorophyll ~ lognormal

# Temporal features (encode seasonality + long-term trend)
FEATURES_TIME = ["sin_month", "cos_month", "year_norm"]

# Features for gating network (regime assignment)
# Includes spatial coordinates for basin-scale structure
FEATURES_GATING = ["lat_norm", "lon_norm", "sst", "sss", "log_chl"] + FEATURES_TIME

# Features for symbolic experts (within-regime laws)
# Excludes lat/lon to enforce spatial structure via gating, not experts
FEATURES_EXPERT = ["sst", "sss", "log_chl"] + FEATURES_TIME

# Complete feature set (for baselines)
FEATURES_ALL = FEATURES_PHYS + FEATURES_BIO + FEATURES_TIME

# Target variable
TARGET = "fco2"  # Fugacity of CO₂ in seawater (μatm)


# =============================================================================
# MODEL ARCHITECTURE
# =============================================================================

@dataclass
class ModelConfig:
    """SD-MoSE model hyperparameters."""
    
    # Regime structure
    n_regimes: int = 6  # K=6 regimes (ablation study: 3, 6, 9)
    
    # Gating network architecture
    gating_hidden_dims: List[int] = field(default_factory=lambda: [64, 32])
    gating_dropout: float = 0.1
    gating_activation: str = "relu"  # or "gelu", "tanh"
    
    # Training schedule
    sdmose_iterations: int = 5  # Alternating gating ↔ expert optimization
    gating_epochs: int = 100
    gating_lr: float = 1e-3
    gating_weight_decay: float = 1e-4
    gating_batch_size: int = 2048
    
    # Symbolic discovery (PySR)
    pysr_niterations: int = 40
    pysr_populations: int = 31
    pysr_binary_operators: List[str] = field(
        default_factory=lambda: ["+", "-", "*", "/"]
    )
    pysr_unary_operators: List[str] = field(
        default_factory=lambda: ["exp", "log", "sqrt", "square"]
    )
    pysr_complexity_penalty: float = 0.01  # Favor simpler equations
    
    # Ensemble training
    n_ensemble: int = 5  # For regime robustness validation
    ensemble_seeds: List[int] = field(default_factory=lambda: list(range(5)))
    
    # Regularization
    entropy_weight: float = 0.01  # Encourage confident regime assignments
    spatial_smoothness_weight: float = 0.05  # Penalize rapid regime transitions
    
    # Device
    device: str = "cuda" if os.getenv("CUDA_VISIBLE_DEVICES") else "cpu"


# Default configuration instance
DEFAULT_CONFIG = ModelConfig()


# =============================================================================
# EVALUATION ZONES
# =============================================================================

# Latitude bands for out-of-distribution testing
# Tests whether discovered regimes generalize across climate zones
LAT_BANDS = {
    "tropics": (-20, 20),           # High stratification, weak seasonality
    "mid_lat_n": (20, 50),          # Strong fronts, high variability
    "mid_lat_s": (-50, -20),        # Southern Ocean influence
    "high_lat_n": (50, 90),         # Arctic, seasonal sea ice
    "high_lat_s": (-90, -50),       # Antarctic, permanent sea ice margin
}

# Ocean basins for regional analysis
OCEAN_BASINS = {
    "atlantic": {"lon": (-80, 20), "lat": (-60, 70)},
    "pacific": {"lon": (120, -80), "lat": (-60, 65)},
    "indian": {"lon": (20, 120), "lat": (-60, 30)},
    "southern": {"lon": (-180, 180), "lat": (-90, -50)},
    "arctic": {"lon": (-180, 180), "lat": (65, 90)},
}


# =============================================================================
# PHYSICAL CONSTANTS & VALIDATION RANGES
# =============================================================================

# Plausible fCO₂ range (μatm) - values outside this suggest data errors
FCO2_MIN_PLAUSIBLE = 200.0
FCO2_MAX_PLAUSIBLE = 600.0

# SST range (°C) - post-QC expected range
SST_MIN = -2.0   # Freezing point of seawater
SST_MAX = 35.0   # Max observed in warm pools

# SSS range (PSU) - practical salinity
SSS_MIN = 0.0    # Freshwater limit
SSS_MAX = 42.0   # Hypersaline regions

# Chlorophyll-a range (mg/m³) - before log transform
CHL_MIN = 0.001  # Oligotrophic gyres
CHL_MAX = 100.0  # Coastal blooms


# =============================================================================
# VISUALIZATION SETTINGS
# =============================================================================

@dataclass
class VizConfig:
    """Publication-quality figure settings."""
    
    # Figure aesthetics
    dpi: int = 300
    figure_format: str = "png"  # or "pdf" for vector graphics
    cmap_regimes: str = "tab10"  # Discrete regime colors
    cmap_uncertainty: str = "viridis"  # Continuous entropy/confidence
    cmap_flux: str = "RdBu_r"  # Diverging colormap for fCO₂ anomalies
    
    # Map projection
    projection: str = "PlateCarree"  # or "Robinson" for global views
    
    # Font sizes
    font_title: int = 14
    font_label: int = 12
    font_tick: int = 10
    
    # Regime visualization
    regime_alpha: float = 0.7  # Transparency for overlapping regimes
    confidence_threshold: float = 0.5  # Min probability to show regime


DEFAULT_VIZ = VizConfig()


# =============================================================================
# RANDOM SEEDS (for reproducibility)
# =============================================================================

RANDOM_SEED = 42
TRAIN_VAL_SPLIT = 0.85  # 85% train, 15% validation within training period


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_checkpoint_path(experiment_name: str, seed: Optional[int] = None) -> Path:
    """Generate checkpoint path with optional ensemble seed."""
    if seed is not None:
        return CHECKPOINT_DIR / "ensemble" / f"seed_{seed}" / f"{experiment_name}.pth"
    return CHECKPOINT_DIR / f"{experiment_name}.pth"


def get_figure_path(figure_name: str, subdir: Optional[str] = None) -> Path:
    """Generate figure path with optional subdirectory."""
    base = FIGURE_DIR / subdir if subdir else FIGURE_DIR
    base.mkdir(parents=True, exist_ok=True)
    return base / f"{figure_name}.{DEFAULT_VIZ.figure_format}"


def validate_feature_columns(df_columns: List[str], required: List[str]) -> None:
    """Raise error if required features missing from DataFrame."""
    missing = set(required) - set(df_columns)
    if missing:
        raise ValueError(f"Missing required feature columns: {missing}")


def get_temporal_mask(
    dates: np.ndarray, 
    start: str, 
    end: str
) -> np.ndarray:
    """Create boolean mask for date range."""
    import pandas as pd
    dates = pd.to_datetime(dates)
    return (dates >= start) & (dates <= end)