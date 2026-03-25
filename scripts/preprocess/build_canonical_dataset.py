"""
Unified Preprocessing Pipeline

Regrids and merges all raw climate data into canonical observation tensors.

CRITICAL: This is the ONLY preprocessing script. All features processed together.
"""

import sys
from pathlib import Path
from typing import Dict, Tuple
import numpy as np
import xarray as xr
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from data.schema import DataContract


def load_sst(raw_dir: Path) -> xr.DataArray:
    """
    Load SST from NOAA OISST files.
    
    Returns:
        xr.DataArray: SST with dims (time, lat, lon)
    """
    print("Loading SST data...")
    
    sst_files = sorted((raw_dir / "sst").glob("*.nc"))
    
    if not sst_files:
        raise FileNotFoundError("No SST files found in data/raw/sst/")
    
    # Load and concatenate monthly files
    datasets = []
    for f in sst_files:
        ds = xr.open_dataset(f)
        # Extract SST variable (check actual variable name in data)
        sst = ds["sst"] if "sst" in ds else ds["analysed_sst"]
        datasets.append(sst)
    
    sst_full = xr.concat(datasets, dim="time")
    
    # Compute monthly means if daily data
    if len(sst_full.time) > DataContract.N_TIME:
        sst_monthly = sst_full.resample(time="1MS").mean()
    else:
        sst_monthly = sst_full
    
    print(f"  Loaded {len(sst_monthly.time)} time steps")
    return sst_monthly


def load_sss(raw_dir: Path) -> xr.DataArray:
    """
    Load SSS from EN4 files.
    
    Returns:
        xr.DataArray: SSS with dims (time, lat, lon)
    """
    print("Loading SSS data...")
    
    sss_files = sorted((raw_dir / "sss").glob("*.nc"))
    
    if not sss_files:
        raise FileNotFoundError("No SSS files found in data/raw/sss/")
    
    datasets = []
    for f in sss_files:
        ds = xr.open_dataset(f)
        # Extract surface salinity (first depth level)
        sss = ds["salinity"].isel(depth=0) if "depth" in ds.dims else ds["salinity"]
        datasets.append(sss)
    
    sss_full = xr.concat(datasets, dim="time")
    
    print(f"  Loaded {len(sss_full.time)} time steps")
    return sss_full


def load_chl(raw_dir: Path) -> xr.DataArray:
    """
    Load Chlorophyll-a from MODIS files.
    
    Returns:
        xr.DataArray: Chl-a with dims (time, lat, lon)
    """
    print("Loading Chlorophyll-a data...")
    
    chl_files = sorted((raw_dir / "chl").glob("*.nc"))
    
    if not chl_files:
        raise FileNotFoundError("No Chl-a files found in data/raw/chl/")
    
    # Validate files first (check they're not HTML error pages)
    corrupted_count = 0
    for f in chl_files:
        with open(f, 'rb') as fh:
            header = fh.read(20)
            if b'<!DOCTYPE' in header or b'<html' in header.lower():
                corrupted_count += 1
                if corrupted_count == 1:  # Only print detailed error once
                    print(f"\n  ✗ ERROR: Chlorophyll files are corrupted HTML error pages!")
                    print(f"  Example: {f.name}")
                    print(f"  First bytes: {header}")
                    print(f"\n  This indicates NASA OceanColor download failed.")
                    print(f"  Solution: Run download script with NASA Earthdata credentials:")
                    print(f"    python scripts/download/download_chl.py")
                    print(f"\n  Setup instructions:")
                    print(f"  1. Register at https://urs.earthdata.nasa.gov/users/new")
                    print(f"  2. Create ~/.netrc file with:")
                    print(f"     machine urs.earthdata.nasa.gov")
                    print(f"     login YOUR_USERNAME")
                    print(f"     password YOUR_PASSWORD")
                    print(f"  3. Run: chmod 600 ~/.netrc (on Unix/Mac)")
                raise ValueError(
                    f"Found {corrupted_count} corrupted chlorophyll files. "
                    f"Please re-run download script with proper authentication."
                )
    
    datasets = []
    for f in chl_files:
        try:
            # Try opening with netcdf4 engine first
            ds = xr.open_dataset(f, engine='netcdf4')
        except Exception:
            try:
                # Fall back to h5netcdf for HDF5 files
                ds = xr.open_dataset(f, engine='h5netcdf')
            except Exception:
                print(f"  Warning: Could not open {f.name}, skipping...")
                continue
        
        # Extract chlorophyll variable (MODIS uses 'chlor_a')
        if 'chlor_a' in ds.variables:
            chl = ds['chlor_a']
        elif 'chlorophyll' in ds.variables:
            chl = ds['chlorophyll']
        else:
            # Find the main data variable
            data_vars = [v for v in ds.data_vars if 'chl' in v.lower()]
            if data_vars:
                chl = ds[data_vars[0]]
            else:
                print(f"  Warning: No chlorophyll variable found in {f.name}, skipping...")
                continue
        
        datasets.append(chl)
    
    if not datasets:
        raise ValueError("No valid chlorophyll data could be loaded")
    
    chl_full = xr.concat(datasets, dim="time")
    
    # Log transform (often better for modeling)
    chl_log = np.log10(chl_full + 1e-6)  # Add epsilon to avoid log(0)
    
    print(f"  Loaded {len(chl_log.time)} time steps")
    return chl_log


def load_fco2(raw_dir: Path) -> xr.DataArray:
    """
    Load fCO2 from SOCAT gridded product.
    
    Returns:
        xr.DataArray: fCO2 with dims (time, lat, lon)
    """
    print("Loading fCO2 data...")
    
    fco2_file = raw_dir / "fco2" / "SOCATv2023_tracks_gridded_monthly.nc"
    
    if not fco2_file.exists():
        raise FileNotFoundError(f"SOCAT file not found: {fco2_file}")
    
    ds = xr.open_dataset(fco2_file)
    fco2 = ds["fco2_sw_mean"]
    
    # Select time range matching contract
    start_date = DataContract.TEMPORAL["start_date"]
    end_date = DataContract.TEMPORAL["end_date"]
    fco2_subset = fco2.sel(time=slice(start_date, end_date))
    
    print(f"  Loaded {len(fco2_subset.time)} time steps")
    return fco2_subset


def regrid_to_target(da: xr.DataArray, target_grid: Dict) -> xr.DataArray:
    """
    Regrid data array to 2° × 2° target grid.
    
    Args:
        da: Input DataArray
        target_grid: Target lat/lon coordinates
        
    Returns:
        Regridded DataArray
    """
    print(f"  Regridding from {da.sizes} to 2° grid...")
    
    # Simple linear interpolation (for production, use xesmf conservative)
    da_regridded = da.interp(
        lat=target_grid["lat"],
        lon=target_grid["lon"],
        method="linear"
    )
    
    return da_regridded


def create_land_mask(sst: xr.DataArray) -> xr.DataArray:
    """
    Create land/ice mask from SST NaNs.
    
    Args:
        sst: Regridded SST
        
    Returns:
        Boolean mask (True = valid ocean)
    """
    print("Creating land/ocean mask...")
    
    # Valid ocean points are where SST is not NaN in most time steps
    valid_count = (~sst.isnull()).sum(dim="time")
    threshold = len(sst.time) * 0.5  # At least 50% valid
    
    mask = valid_count > threshold
    
    print(f"  Valid ocean points: {mask.sum().values} / {mask.size}")
    return mask


def build_canonical_dataset(raw_dir: Path) -> xr.Dataset:
    """
    Build canonical dataset from all raw sources.
    
    Returns:
        xr.Dataset: Canonical dataset conforming to DataContract
    """
    print("\n" + "="*60)
    print("BUILDING CANONICAL DATASET")
    print("="*60 + "\n")
    
    # Get target grid
    grid = DataContract.get_grid_spec()
    
    # Load all variables
    sst = load_sst(raw_dir)
    sss = load_sss(raw_dir)
    chl = load_chl(raw_dir)
    fco2 = load_fco2(raw_dir)
    
    # Regrid to common grid
    print("\nRegridding variables...")
    sst_reg = regrid_to_target(sst, grid)
    sss_reg = regrid_to_target(sss, grid)
    chl_reg = regrid_to_target(chl, grid)
    fco2_reg = regrid_to_target(fco2, grid)
    
    # Create land mask
    mask = create_land_mask(sst_reg)
    
    # Stack features
    print("\nStacking features...")
    features = xr.concat(
        [sst_reg, sss_reg, chl_reg],
        dim=xr.DataArray(
            DataContract.FEATURES,
            dims=["feature"],
            name="feature_names"
        )
    )
    
    # Create dataset
    ds = xr.Dataset({
        "features": features,
        "target": fco2_reg,
        "mask": mask,
    })
    
    # Add metadata
    ds.attrs.update(DataContract.export_metadata())
    
    # Quality checks
    print("\nRunning quality checks...")
    DataContract.validate_dataset(ds)
    print("  ✓ All quality checks passed")
    
    return ds


def split_train_test(ds: xr.Dataset) -> Tuple[xr.Dataset, xr.Dataset]:
    """
    Split dataset into train and test sets (time-based).
    
    Args:
        ds: Full dataset
        
    Returns:
        (train_ds, test_ds)
    """
    train_start, train_end = DataContract.TRAIN_PERIOD
    test_start, test_end = DataContract.TEST_PERIOD
    
    train_ds = ds.sel(time=slice(train_start, train_end))
    test_ds = ds.sel(time=slice(test_start, test_end))
    
    print(f"\nTrain set: {len(train_ds.time)} months")
    print(f"Test set: {len(test_ds.time)} months")
    
    return train_ds, test_ds


def main():
    """Build canonical dataset from raw data."""
    raw_dir = Path("data/raw")
    output_dir = Path("data/processed")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Build dataset
    ds = build_canonical_dataset(raw_dir)
    
    # Split train/test
    train_ds, test_ds = split_train_test(ds)
    
    # Save
    print("\nSaving datasets...")
    
    output_file = output_dir / "canonical_dataset.nc"
    ds.to_netcdf(output_file)
    print(f"  ✓ Full dataset: {output_file}")
    
    train_file = output_dir / "train.nc"
    train_ds.to_netcdf(train_file)
    print(f"  ✓ Train set: {train_file}")
    
    test_file = output_dir / "test.nc"
    test_ds.to_netcdf(test_file)
    print(f"  ✓ Test set: {test_file}")
    
    # Summary
    print("\n" + "="*60)
    print("PREPROCESSING COMPLETE")
    print("="*60)
    print(f"Total size: {output_file.stat().st_size / 1e6:.1f} MB")
    print(f"Train: {len(train_ds.time)} months ({train_ds.time.min().values} to {train_ds.time.max().values})")
    print(f"Test: {len(test_ds.time)} months ({test_ds.time.min().values} to {test_ds.time.max().values})")
    print(f"Features: {DataContract.FEATURES}")
    print(f"Grid: {len(ds.lat)} lat × {len(ds.lon)} lon")


if __name__ == "__main__":
    main()
