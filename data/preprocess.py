"""Preprocess and fuse SOCAT + CMEMS data for SD-MoSE pipeline.

This script:
1. Loads SOCAT gridded fCO₂ data
2. Loads CMEMS chlorophyll data
3. Spatially/temporally aligns the datasets
4. Adds derived features (log_chl, cyclic time, normalized coords)
5. Outputs climate_fused_dataset.nc

Requirements:
- SOCAT data in data/raw/ (run download_socat.py)
- CMEMS data in data/raw/ (run download_copernicus.py)
- Python packages: xarray, pandas, numpy, netCDF4
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr


def add_derived_features(ds):
    """Add derived features for SD-MoSE model.
    
    Features added:
    - log_chl: log(chlorophyll) for handling skewness
    - lat_norm, lon_norm: normalized coordinates [-1, 1]
    - sin_month, cos_month: cyclic month encoding
    - year_norm: normalized year for trend detection
    """
    print("  Adding derived features...")
    
    # Log-transformed chlorophyll
    if 'chl' in ds:
        ds['log_chl'] = np.log(np.clip(ds['chl'], 1e-3, None))
        print("    ✓ log_chl")
    
    # Normalized coordinates
    ds['lat_norm'] = ds['lat'] / 90.0
    ds['lon_norm'] = ds['lon'] / 180.0
    print("    ✓ lat_norm, lon_norm")
    
    # Cyclic time features
    if 'time' in ds.dims:
        time_pd = pd.to_datetime(ds['time'].values)
        month_numeric = time_pd.month - 1  # 0-11
        angle = 2 * np.pi * month_numeric / 12.0
        
        ds['sin_month'] = ('time', np.sin(angle))
        ds['cos_month'] = ('time', np.cos(angle))
        print("    ✓ sin_month, cos_month")
        
        # Normalized year (centered at 2015)
        year = time_pd.year
        ds['year_norm'] = ('time', (year - 2015) / 10.0)
        print("    ✓ year_norm")
    
    # SST gradient (spatial derivative)
    if 'sst' in ds and 'lat' in ds.dims and 'lon' in ds.dims:
        # Simple gradient magnitude
        ds['sst_gradient'] = np.sqrt(
            ds['sst'].differentiate('lat')**2 + 
            ds['sst'].differentiate('lon')**2
        )
        print("    ✓ sst_gradient")
    
    return ds


def main():
    """Main preprocessing pipeline."""
    
    # Setup paths
    project_root = Path(__file__).parent.parent
    raw_dir = project_root / "data" / "raw"
    processed_dir = project_root / "data" / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)
    
    socat_file = raw_dir / "SOCATv2024_tracks_gridded_monthly.nc"
    cmems_file = raw_dir / "cmems_mod_glo_bgc_my_0.25deg_P1M-m_chl.nc"
    output_file = processed_dir / "climate_fused_dataset.nc"
    
    print("=" * 70)
    print("SOCAT + CMEMS PREPROCESSING")
    print("=" * 70)
    print()
    
    # Check inputs
    if not socat_file.exists():
        print(f"✗ ERROR: SOCAT file not found: {socat_file}")
        print("  Run: python data/download_socat.py")
        sys.exit(1)
    
    if not cmems_file.exists():
        print(f"✗ ERROR: CMEMS file not found: {cmems_file}")
        print("  Run: python data/download_copernicus.py")
        sys.exit(1)
    
    print("Input files:")
    print(f"  - SOCAT: {socat_file.name}")
    print(f"  - CMEMS: {cmems_file.name}")
    print(f"Output: {output_file}")
    print()
    
    # Load SOCAT
    print("1. Loading SOCAT data...")
    try:
        socat = xr.open_dataset(socat_file)
        print(f"   Dimensions: {dict(socat.dims)}")
        print(f"   Variables: {list(socat.data_vars)[:5]}...")  # Show first 5
    except Exception as e:
        print(f"   ✗ Failed to load SOCAT: {e}")
        sys.exit(1)
    
    # Load CMEMS
    print()
    print("2. Loading CMEMS chlorophyll...")
    try:
        cmems = xr.open_dataset(cmems_file)
        print(f"   Dimensions: {dict(cmems.dims)}")
        print(f"   Variables: {list(cmems.data_vars)}")
    except Exception as e:
        print(f"   ✗ Failed to load CMEMS: {e}")
        sys.exit(1)
    
    # Regrid and align
    print()
    print("3. Aligning datasets...")
    print("   Regridding CMEMS to SOCAT grid (1° resolution)...")
    
    try:
        # Interpolate CMEMS to SOCAT grid
        cmems_regrid = cmems.interp(
            lat=socat['lat'],
            lon=socat['lon'],
            time=socat['time'],
            method='linear'
        )
        print("   ✓ Spatial/temporal alignment complete")
    except Exception as e:
        print(f"   ✗ Alignment failed: {e}")
        sys.exit(1)
    
    # Merge datasets
    print()
    print("4. Merging datasets...")
    try:
        # Select key variables from SOCAT
        socat_vars = {}
        # Try different possible variable names
        for var_name in ['fco2_ave_weighted', 'fco2', 'fCO2']:
            if var_name in socat:
                socat_vars['fco2'] = socat[var_name]
                break
        
        for var_name in ['sst_ave_weighted', 'sst', 'SST']:
            if var_name in socat:
                socat_vars['sst'] = socat[var_name]
                break
        
        for var_name in ['sss_ave_weighted', 'sss', 'SSS', 'sal']:
            if var_name in socat:
                socat_vars['sss'] = socat[var_name]
                break
        
        if not socat_vars:
            print("   ✗ Could not find fco2/sst/sss in SOCAT file")
            print(f"   Available variables: {list(socat.data_vars)}")
            sys.exit(1)
        
        # Create merged dataset
        merged = xr.Dataset(socat_vars)
        merged['chl'] = cmems_regrid['chl']
        merged['lat'] = socat['lat']
        merged['lon'] = socat['lon']
        merged['time'] = socat['time']
        
        print(f"   ✓ Merged {len(merged.data_vars)} variables")
        
    except Exception as e:
        print(f"   ✗ Merging failed: {e}")
        sys.exit(1)
    
    # Add derived features
    print()
    print("5. Feature engineering...")
    merged = add_derived_features(merged)
    
    # Filter to 2000-2023
    print()
    print("6. Filtering to 2000-2023...")
    merged = merged.sel(time=slice('2000-01-01', '2023-12-31'))
    print(f"   ✓ Time range: {merged.time.values[0]} to {merged.time.values[-1]}")
    
    # Save
    print()
    print("7. Saving preprocessed dataset...")
    try:
        merged.to_netcdf(
            output_file,
            engine='netcdf4',
            encoding={
                var: {'zlib': True, 'complevel': 4}
                for var in merged.data_vars
            }
        )
        size_mb = output_file.stat().st_size / 1e6
        print(f"   ✓ Saved: {output_file}")
        print(f"   Size: {size_mb:.1f} MB")
    except Exception as e:
        print(f"   ✗ Save failed: {e}")
        sys.exit(1)
    
    # Summary
    print()
    print("=" * 70)
    print("PREPROCESSING COMPLETE")
    print("=" * 70)
    print()
    print("Dataset summary:")
    print(f"  Dimensions: {dict(merged.dims)}")
    print(f"  Variables: {list(merged.data_vars)}")
    print(f"  Time range: {merged.time.values[0]} to {merged.time.values[-1]}")
    print()
    print("Next step: Run the pipeline!")
    print("  python scripts/pipeline.py --n-regimes 6 --pysr_iterations 40")
    print()
    
    # Close datasets
    socat.close()
    cmems.close()
    merged.close()


if __name__ == "__main__":
    main()
