"""Central configuration for SD-MoSE pipeline.
Paper: Soft Regime Mixture of Symbolic Experts for Discovering Interpretable Air-Sea CO₂ Laws.
"""

from pathlib import Path

# --- Paths ---
PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
FIGURE_DIR = PROJECT_ROOT / "figures"

# SOCAT (NOAA NCEI 0304549). Try v2025 first; fallback v2024.
SOCAT_FILENAME = "SOCATv2025_tracks_gridded_monthly.nc"
SOCAT_PATH = RAW_DIR / SOCAT_FILENAME
CHL_FILENAME = "cmems_mod_glo_bgc_my_0.25deg_P1M-m_chl.nc"
CHL_PATH = RAW_DIR / CHL_FILENAME

TRAIN_NC = PROCESSED_DIR / "train_dataset.nc"
TEST_NC = PROCESSED_DIR / "test_dataset.nc"
FUSED_NC = PROCESSED_DIR / "climate_fused_dataset.nc"
SCALERS_NC = PROCESSED_DIR / "scalers.nc"

# --- Time split (generalization: train years 1–7, test 8–10) ---
START_YEAR = 2015
END_YEAR = 2024
SPLIT_YEAR = 2022  # Train: 2015..2021, Test: 2022..2024
TRAIN_END = f"{SPLIT_YEAR - 1}-12-31"
TEST_START = f"{SPLIT_YEAR}-01-01"

# --- Features ---
FEATURES_PHYS = ["sst", "sss"]
FEATURES_BIO = ["log_chl"]
FEATURES_TIME = ["sin_month", "cos_month", "year_feature"]
FEATURES_GATING = ["lat_norm", "lon_norm", "sin_month", "cos_month", "year_feature"]
FEATURES_EXPERT = ["sst", "sss", "log_chl"]
FEATURES_ALL = FEATURES_EXPERT + FEATURES_TIME  # order for flat X

TARGET = "fco2"

# --- Model ---
N_REGIMES = 6
SDMOSE_ITERATIONS = 5  # Training loop: gating <-> experts refit

# --- OOD latitude bands (for evaluation) ---
LAT_BANDS = {
    "tropics": (-20, 20),
    "mid_lat_n": (20, 50),
    "mid_lat_s": (-50, -20),
    "high_lat_n": (50, 90),
    "high_lat_s": (-90, -50),
}

# --- fCO2 plausible range (µatm) after standardization we use raw; for plausibility checks ---
FCO2_MIN_PLAUSIBLE = 200
FCO2_MAX_PLAUSIBLE = 550
