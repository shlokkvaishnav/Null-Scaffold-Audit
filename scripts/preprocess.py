import xarray as xr
import numpy as np
import logging
from pathlib import Path

# Setup Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger(__name__)

# --- PATHS ---
RAW_DIR = Path("data/01_raw")
PROCESSED_DIR = Path("data/03_processed")

# Ensure these match your actual filenames
SOCAT_PATH = RAW_DIR / "SOCATv2025_tracks_gridded_monthly.nc"
CHL_PATH = RAW_DIR / "cmems_mod_glo_bgc_my_0.25deg_P1M-m_chl.nc"
OUTPUT_PATH = PROCESSED_DIR / "climate_fused_dataset.nc"

# Roadmap Requirement: Start with 2 degree resolution for fast prototyping
TARGET_RES = 2.0  

def create_target_grid(res):
    """Creates a global grid at specified resolution (-180 to 180)."""
    lats = np.arange(-90 + res/2, 90, res)
    lons = np.arange(-180 + res/2, 180, res)
    return xr.Dataset({
        "lat": (["lat"], lats, {"units": "degrees_north"}),
        "lon": (["lon"], lons, {"units": "degrees_east"}),
    })

def add_cyclic_time(ds):
    """Adds sin/cos month features as per roadmap."""
    # Month is 1-12
    month = ds.time.dt.month
    ds['sin_month'] = np.sin(2 * np.pi * month / 12)
    ds['cos_month'] = np.cos(2 * np.pi * month / 12)
    ds['year'] = ds.time.dt.year
    return ds

def preprocess():
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    
    # 1. Load Data with Dask (for memory efficiency)
    logger.info("1. Loading Datasets...")
    try:
        ds_socat = xr.open_dataset(SOCAT_PATH, decode_times=True, chunks={"time": 12})
        ds_chl = xr.open_dataset(CHL_PATH, decode_times=True, chunks={"time": 12})
    except FileNotFoundError as e:
        logger.error(f"❌ File not found: {e}. Check data/01_raw/")
        return

    # 2. Standardize SOCAT Variables
    rename_map = {
        'fco2_ave_unwtd': 'fco2',
        'sst_ave_unwtd': 'sst',
        'salinity_ave_unwtd': 'sss',
        'tmnth': 'time',
        'ylat': 'lat',
        'xlon': 'lon'
    }
    
    # Only rename what exists
    existing_vars = {k: v for k, v in rename_map.items() if k in ds_socat}
    ds_socat = ds_socat.rename(existing_vars)
    
    # Keep only relevant variables
    ds_socat = ds_socat[['fco2', 'sst', 'sss']]

    # 3. Handle Longitude Convention (0-360 vs -180-180)
    if ds_socat.lon.max() > 180:
        logger.info("   Adjusting SOCAT longitude from 0-360 to -180/180...")
        ds_socat.coords['lon'] = (ds_socat.coords['lon'] + 180) % 360 - 180
        ds_socat = ds_socat.sortby(ds_socat.lon)

    # 4. Standardize Chlorophyll
    rename_chl = {}
    if 'latitude' in ds_chl.dims:
        rename_chl['latitude'] = 'lat'
    if 'longitude' in ds_chl.dims:
        rename_chl['longitude'] = 'lon'
    
    if rename_chl:
        ds_chl = ds_chl.rename(rename_chl)
    
    if 'depth' in ds_chl.dims:
        ds_chl = ds_chl.isel(depth=0)

    # 5. Regrid Both to Target Coarse Grid
    logger.info(f"2. Regridding both datasets to {TARGET_RES} degree resolution...")
    target_grid = create_target_grid(TARGET_RES)

    # We use 'linear' interpolation to align them to the new shared grid
    ds_socat_regrid = ds_socat.interp(
        lat=target_grid.lat, 
        lon=target_grid.lon, 
        method="linear"
    )
    
    ds_chl_regrid = ds_chl['chl'].interp(
        lat=target_grid.lat, 
        lon=target_grid.lon, 
        method="linear"
    )

    # 6. Time Intersection
    common_start = max(ds_socat_regrid.time[0].values, ds_chl_regrid.time[0].values)
    common_end = min(ds_socat_regrid.time[-1].values, ds_chl_regrid.time[-1].values)
    
    logger.info(f"   Time overlap: {str(common_start)[:10]} to {str(common_end)[:10]}")
    ds_socat_regrid = ds_socat_regrid.sel(time=slice(common_start, common_end))
    ds_chl_regrid = ds_chl_regrid.sel(time=slice(common_start, common_end))

    # 7. Merge & Feature Engineering
    logger.info("3. Merging and Engineering Features...")
    
    # Log transform Biology
    log_chl = np.log10(ds_chl_regrid + 1e-6)
    log_chl.name = 'log_chl'

    ds_merged = xr.merge([ds_socat_regrid, log_chl])
    ds_merged = add_cyclic_time(ds_merged)

    # 8. Save
    logger.info("4. Saving to NetCDF...")
    
    comp = dict(zlib=True, complevel=5)
    encoding = {var: comp for var in ds_merged.data_vars}
    
    ds_merged.to_netcdf(OUTPUT_PATH, encoding=encoding)
    
    logger.info(f"✅ Success! Saved to {OUTPUT_PATH}")
    logger.info(f"   Dimensions: {ds_merged.sizes}")
    logger.info(f"   Variables: {list(ds_merged.data_vars)}")

if __name__ == "__main__":
    preprocess()