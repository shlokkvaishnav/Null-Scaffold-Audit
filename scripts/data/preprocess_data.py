"""Preprocess raw data: align, merge, quality control, feature engineering.

Scientific Pipeline:
1. Load SOCAT (fCO₂, SST, SSS) and Copernicus (Chl-a)
2. Harmonize coordinates and temporal alignment
3. Quality control: Remove outliers, invalid values
4. Feature engineering: Log-chlorophyll, cyclic time, normalized coordinates
5. Train/test split (temporal: 2015-2021 / 2022-2024)
6. Standardization (z-score) of physical variables
7. Save processed datasets + scalers

Outputs:
- train_dataset.nc: Training data (standardized)
- test_dataset.nc: Test data (standardized)
- climate_fused_dataset.nc: Full dataset (for visualization)
- scalers.nc: Mean/std for inverse transform
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import xarray as xr

# Add src to path
root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(root / "src"))

from climate_discovery.config import (
    CHL_PATH,
    END_YEAR,
    FCO2_MAX_PLAUSIBLE,
    FCO2_MIN_PLAUSIBLE,
    FUSED_NC,
    PROCESSED_DIR,
    SCALERS_NC,
    SOCAT_PATH,
    SPLIT_YEAR,
    START_YEAR,
    TEST_NC,
    TRAIN_NC,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


# =============================================================================
# QUALITY CONTROL
# =============================================================================

def quality_control_fco2(ds: xr.Dataset) -> Tuple[xr.Dataset, Dict]:
    """Remove outliers and invalid fCO₂ values.
    
    Args:
        ds: Dataset with 'fco2' variable
        
    Returns:
        Cleaned dataset, QC statistics dict
    """
    fco2 = ds["fco2"].values
    n_total = np.sum(~np.isnan(fco2))
    
    # Flag outliers
    invalid = (fco2 < FCO2_MIN_PLAUSIBLE) | (fco2 > FCO2_MAX_PLAUSIBLE)
    n_outliers = np.sum(invalid)
    
    # Set outliers to NaN
    ds["fco2"] = ds["fco2"].where(~invalid, np.nan)
    
    stats = {
        "n_total": n_total,
        "n_outliers": n_outliers,
        "frac_removed": n_outliers / n_total if n_total > 0 else 0,
        "min": float(np.nanmin(fco2)),
        "max": float(np.nanmax(fco2)),
    }
    
    logger.info(
        f"fCO₂ QC: {n_outliers}/{n_total} ({stats['frac_removed']*100:.2f}%) "
        f"outside [{FCO2_MIN_PLAUSIBLE}, {FCO2_MAX_PLAUSIBLE}] μatm removed"
    )
    
    return ds, stats


# =============================================================================
# FEATURE ENGINEERING
# =============================================================================

def add_temporal_features(ds: xr.Dataset) -> xr.Dataset:
    """Add cyclic time encodings and normalized year.
    
    Scientific rationale:
    - sin_month, cos_month: Capture seasonality (Jan ≈ Dec)
    - year_norm: Linear trend normalized to [-1, 1] range
    
    Args:
        ds: Dataset with 'time' coordinate
        
    Returns:
        Dataset with added features
    """
    # Cyclic month encoding
    month = ds.time.dt.month  # 1-12
    angle = 2 * np.pi * (month - 1) / 12.0  # 0 to 2π
    
    ds["sin_month"] = np.sin(angle)
    ds["cos_month"] = np.cos(angle)
    
    # Normalized year (centered on mid-point of training period)
    reference_year = (START_YEAR + END_YEAR) / 2.0
    ds["year_norm"] = (ds.time.dt.year - reference_year) / 10.0
    
    return ds


def add_spatial_features(ds: xr.Dataset) -> xr.Dataset:
    """Add normalized spatial coordinates.
    
    For gating network input:
    - lat_norm: [-90, 90] → [-1, 1]
    - lon_norm: [-180, 180] → [-1, 1]
    
    Args:
        ds: Dataset with 'lat', 'lon' coordinates
        
    Returns:
        Dataset with added features
    """
    ds["lat_norm"] = ds["lat"] / 90.0
    ds["lon_norm"] = ds["lon"] / 180.0
    
    return ds


def add_log_chlorophyll(ds: xr.Dataset, chl_var: str = "chl") -> xr.Dataset:
    """Transform chlorophyll to log-scale.
    
    Chlorophyll-a follows log-normal distribution.
    Log transform → Gaussian-like, better for regression.
    
    Args:
        ds: Dataset with chlorophyll variable
        chl_var: Name of chlorophyll variable
        
    Returns:
        Dataset with 'log_chl' added
    """
    epsilon = 1e-6  # Floor to prevent log(0)
    chl_safe = ds[chl_var].clip(min=epsilon)
    ds["log_chl"] = np.log10(chl_safe)
    
    return ds


# =============================================================================
# COORDINATE HARMONIZATION
# =============================================================================

def harmonize_socat(ds: xr.Dataset) -> xr.Dataset:
    """Rename SOCAT variables to standard names.
    
    SOCAT uses inconsistent naming across versions.
    Standardize to: fco2, sst, sss
    """
    # Variable name mapping
    var_map = {
        "fco2_ave_unwtd": "fco2",
        "fco2_ave_weighted": "fco2",
        "sst_ave_unwtd": "sst",
        "sst_ave_weighted": "sst",
        "salinity_ave_unwtd": "sss",
        "salinity_ave_weighted": "sss",
    }
    
    # Dimension name mapping
    dim_map = {
        "tmnth": "time",
        "ylat": "lat",
        "xlon": "lon",
    }
    
    # Build rename dict for existing vars/dims
    rename_dict = {}
    for old, new in {**var_map, **dim_map}.items():
        if old in ds.dims or old in ds.data_vars or old in ds.coords:
            rename_dict[old] = new
    
    ds = ds.rename(rename_dict)
    
    # Select only needed variables
    available_vars = [v for v in ["fco2", "sst", "sss"] if v in ds.data_vars]
    ds = ds[available_vars]
    
    return ds


def harmonize_chlorophyll(ds: xr.Dataset) -> xr.Dataset:
    """Standardize Copernicus chlorophyll dataset.
    
    - Rename latitude/longitude
    - Select surface layer (depth=0)
    - Extract chlorophyll variable
    """
    # Rename coordinates
    coord_map = {
        "latitude": "lat",
        "longitude": "lon",
    }
    rename_dict = {k: v for k, v in coord_map.items() if k in ds.dims or k in ds.coords}
    ds = ds.rename(rename_dict)
    
    # Select surface layer
    if "depth" in ds.dims:
        ds = ds.isel(depth=0, drop=True)
    elif "elevation" in ds.dims:
        ds = ds.isel(elevation=0, drop=True)
    
    return ds


def fix_longitude_range(ds: xr.Dataset) -> xr.Dataset:
    """Convert longitude from [0, 360] to [-180, 180] if needed.
    
    Args:
        ds: Dataset with 'lon' coordinate
        
    Returns:
        Dataset with lon in [-180, 180]
    """
    if "lon" not in ds.coords:
        return ds
    
    lon = ds.coords["lon"]
    
    # Check if longitude is in [0, 360]
    if float(lon.max()) > 180:
        logger.info("Converting longitude from [0,360] to [-180,180]")
        ds = ds.assign_coords(lon=(lon + 180) % 360 - 180)
        ds = ds.sortby("lon")
    
    return ds


# =============================================================================
# STANDARDIZATION
# =============================================================================

def compute_standardization_params(
    ds_train: xr.Dataset,
    variables: list,
) -> Dict[str, Tuple[float, float]]:
    """Compute mean and std for z-score normalization.
    
    Args:
        ds_train: Training dataset
        variables: Variables to standardize
        
    Returns:
        Dict mapping variable to (mean, std)
    """
    params = {}
    for var in variables:
        if var not in ds_train:
            logger.warning(f"Variable {var} not in dataset, skipping")
            continue
        
        mean = float(ds_train[var].mean(dim=("time", "lat", "lon")).values)
        std = float(ds_train[var].std(dim=("time", "lat", "lon")).values)
        params[var] = (mean, std)
        
        logger.info(f"{var}: mean={mean:.3f}, std={std:.3f}")
    
    return params


def apply_standardization(
    ds: xr.Dataset,
    params: Dict[str, Tuple[float, float]],
    epsilon: float = 1e-6,
) -> xr.Dataset:
    """Apply z-score normalization.
    
    Args:
        ds: Dataset to standardize
        params: Dict from compute_standardization_params()
        epsilon: Prevent division by zero
        
    Returns:
        Standardized dataset
    """
    ds = ds.copy()
    for var, (mean, std) in params.items():
        if var in ds:
            ds[var] = (ds[var] - mean) / (std + epsilon)
    return ds


# =============================================================================
# MAIN PREPROCESSING PIPELINE
# =============================================================================

def preprocess(
    socat_path: Path = SOCAT_PATH,
    chl_path: Path = CHL_PATH,
    output_dir: Path = PROCESSED_DIR,
) -> None:
    """Execute full preprocessing pipeline.
    
    Args:
        socat_path: Path to SOCAT NetCDF
        chl_path: Path to Copernicus chlorophyll NetCDF
        output_dir: Directory for processed outputs
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("=" * 60)
    logger.info("LOADING RAW DATASETS")
    logger.info("=" * 60)
    
    # Load datasets
    try:
        ds_socat = xr.open_dataset(socat_path, decode_times=True)
        logger.info(f"✓ Loaded SOCAT: {socat_path}")
    except FileNotFoundError:
        logger.error(f"❌ SOCAT file not found: {socat_path}")
        logger.info("Run: python -m scripts.data.download_data")
        sys.exit(1)
    
    try:
        ds_chl = xr.open_dataset(chl_path, decode_times=True)
        logger.info(f"✓ Loaded Chlorophyll: {chl_path}")
    except FileNotFoundError:
        logger.error(f"❌ Chlorophyll file not found: {chl_path}")
        logger.info("Run: python -m scripts.data.download_data")
        sys.exit(1)
    
    # =========================================================================
    # COORDINATE HARMONIZATION
    # =========================================================================
    logger.info("=" * 60)
    logger.info("HARMONIZING COORDINATES")
    logger.info("=" * 60)
    
    ds_socat = harmonize_socat(ds_socat)
    ds_chl = harmonize_chlorophyll(ds_chl)
    
    # Fix longitude range
    ds_socat = fix_longitude_range(ds_socat)
    ds_chl = fix_longitude_range(ds_chl)
    
    # =========================================================================
    # TEMPORAL SLICING
    # =========================================================================
    logger.info("=" * 60)
    logger.info("TEMPORAL SLICING")
    logger.info("=" * 60)
    
    start_time = f"{START_YEAR}-01-01"
    end_time = f"{END_YEAR}-12-31"
    
    ds_socat = ds_socat.sel(time=slice(start_time, end_time))
    ds_chl = ds_chl.sel(time=slice(start_time, end_time))
    
    logger.info(f"Time range: {start_time} to {end_time}")
    logger.info(f"SOCAT timesteps: {len(ds_socat.time)}")
    logger.info(f"Chlorophyll timesteps: {len(ds_chl.time)}")
    
    # =========================================================================
    # MERGE DATASETS
    # =========================================================================
    logger.info("=" * 60)
    logger.info("MERGING DATASETS")
    logger.info("=" * 60)
    
    # Find chlorophyll variable name
    chl_candidates = [v for v in ds_chl.data_vars if "chl" in v.lower()]
    if not chl_candidates:
        logger.error("❌ No chlorophyll variable found in dataset")
        sys.exit(1)
    chl_var = chl_candidates[0]
    logger.info(f"Using chlorophyll variable: {chl_var}")
    
    # Interpolate chlorophyll to SOCAT grid
    chl_interp = ds_chl[chl_var].interp_like(ds_socat, method="linear")
    chl_interp.name = "chl"
    
    # Merge
    ds_merged = xr.merge([ds_socat, chl_interp])
    logger.info(f"✓ Merged dataset shape: {dict(ds_merged.dims)}")
    
    # =========================================================================
    # QUALITY CONTROL
    # =========================================================================
    logger.info("=" * 60)
    logger.info("QUALITY CONTROL")
    logger.info("=" * 60)
    
    ds_merged, qc_stats = quality_control_fco2(ds_merged)
    
    # =========================================================================
    # FEATURE ENGINEERING
    # =========================================================================
    logger.info("=" * 60)
    logger.info("FEATURE ENGINEERING")
    logger.info("=" * 60)
    
    ds_merged = add_log_chlorophyll(ds_merged, chl_var="chl")
    ds_merged = add_temporal_features(ds_merged)
    ds_merged = add_spatial_features(ds_merged)
    
    logger.info("✓ Added features:")
    logger.info("  - log_chl (log10 transform)")
    logger.info("  - sin_month, cos_month (cyclic time)")
    logger.info("  - year_norm (normalized year)")
    logger.info("  - lat_norm, lon_norm (normalized coordinates)")
    
    # =========================================================================
    # TRAIN/TEST SPLIT
    # =========================================================================
    logger.info("=" * 60)
    logger.info("TRAIN/TEST SPLIT")
    logger.info("=" * 60)
    
    split_date = f"{SPLIT_YEAR}-01-01"
    
    ds_train = ds_merged.sel(time=slice(None, f"{SPLIT_YEAR - 1}-12-31"))
    ds_test = ds_merged.sel(time=slice(split_date, None))
    
    logger.info(f"Train: {START_YEAR}–{SPLIT_YEAR-1} ({len(ds_train.time)} timesteps)")
    logger.info(f"Test:  {SPLIT_YEAR}–{END_YEAR} ({len(ds_test.time)} timesteps)")
    
    # =========================================================================
    # STANDARDIZATION
    # =========================================================================
    logger.info("=" * 60)
    logger.info("STANDARDIZATION")
    logger.info("=" * 60)
    
    # Compute parameters from training data only
    variables_to_standardize = ["sst", "sss", "log_chl"]
    std_params = compute_standardization_params(ds_train, variables_to_standardize)
    
    # Apply to both train and test
    ds_train = apply_standardization(ds_train, std_params)
    ds_test = apply_standardization(ds_test, std_params)
    
    # Save scalers for inverse transform
    scalers = xr.Dataset()
    for var, (mean, std) in std_params.items():
        scalers[f"{var}_mean"] = mean
        scalers[f"{var}_std"] = std
    
    # =========================================================================
    # SAVE OUTPUTS
    # =========================================================================
    logger.info("=" * 60)
    logger.info("SAVING OUTPUTS")
    logger.info("=" * 60)
    
    # Compression settings
    comp = dict(zlib=True, complevel=5)
    encoding = {v: comp for v in ds_train.data_vars}
    
    # Save train/test
    ds_train.to_netcdf(TRAIN_NC, encoding=encoding)
    logger.info(f"✓ Saved: {TRAIN_NC}")
    
    ds_test.to_netcdf(TEST_NC, encoding=encoding)
    logger.info(f"✓ Saved: {TEST_NC}")
    
    # Save scalers
    scalers.to_netcdf(SCALERS_NC)
    logger.info(f"✓ Saved: {SCALERS_NC}")
    
    # Save fused dataset (for visualization)
    ds_fused = xr.concat([ds_train, ds_test], dim="time").sortby("time")
    ds_fused.to_netcdf(FUSED_NC, encoding={v: comp for v in ds_fused.data_vars})
    logger.info(f"✓ Saved: {FUSED_NC}")
    
    # =========================================================================
    # SUMMARY
    # =========================================================================
    logger.info("=" * 60)
    logger.info("PREPROCESSING COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"\nNext step: python -m scripts.train.train_gating")


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Preprocess SOCAT and Copernicus data"
    )
    parser.add_argument(
        "--socat",
        type=str,
        default=None,
        help=f"Path to SOCAT file (default: {SOCAT_PATH})"
    )
    parser.add_argument(
        "--chlorophyll",
        type=str,
        default=None,
        help=f"Path to chlorophyll file (default: {CHL_PATH})"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help=f"Output directory (default: {PROCESSED_DIR})"
    )
    
    args = parser.parse_args()
    
    socat = Path(args.socat) if args.socat else SOCAT_PATH
    chl = Path(args.chlorophyll) if args.chlorophyll else CHL_PATH
    output = Path(args.output) if args.output else PROCESSED_DIR
    
    preprocess(socat, chl, output)


if __name__ == "__main__":
    main()