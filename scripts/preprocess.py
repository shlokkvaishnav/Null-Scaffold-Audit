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
SOCAT_PATH = RAW_DIR / "SOCATv2025_tracks_gridded_monthly.nc"
CHL_PATH = RAW_DIR / "cmems_mod_glo_bgc_my_0.25deg_P1M-m_chl.nc"
OUTPUT_PATH = PROCESSED_DIR / "climate_fused_dataset.nc"

def add_cyclic_time(ds):
    """Adds sin/cos month features."""
    month = ds.time.dt.month
    ds['sin_month'] = np.sin(2 * np.pi * month / 12)
    ds['cos_month'] = np.cos(2 * np.pi * month / 12)
    ds['year'] = ds.time.dt.year
    return ds

def preprocess():
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    
    # 1. Load Data
    logger.info("1. Loading Datasets...")
    try:
        # Load SOCAT (Physics - Sparse)
        ds_socat = xr.open_dataset(SOCAT_PATH, decode_times=True)
        # Load Chlorophyll (Biology - Dense)
        ds_chl = xr.open_dataset(CHL_PATH, decode_times=True)
    except FileNotFoundError as e:
        logger.error(f"❌ File not found: {e}")
        return

    # 2. Rename Variables
    # SOCAT
    rename_socat = {
        'fco2_ave_unwtd': 'fco2',
        'sst_ave_unwtd': 'sst',
        'salinity_ave_unwtd': 'sss', 
        'tmnth': 'time', 'ylat': 'lat', 'xlon': 'lon'
    }
    ds_socat = ds_socat.rename({k: v for k, v in rename_socat.items() if k in ds_socat})
    ds_socat = ds_socat[['fco2', 'sst', 'sss']] # Select physics vars

    # Chlorophyll
    rename_chl = {}
    if 'latitude' in ds_chl.dims: 
        rename_chl['latitude'] = 'lat'
    if 'longitude' in ds_chl.dims: 
        rename_chl['longitude'] = 'lon'
    ds_chl = ds_chl.rename(rename_chl)
    if 'depth' in ds_chl.dims: 
        ds_chl = ds_chl.isel(depth=0)

    # 3. Handle Longitude (Align -180..180)
    # If SOCAT is 0..360, flip it.
    if ds_socat.lon.max() > 180:
        logger.info("   Adjusting SOCAT longitude to -180..180...")
        ds_socat.coords['lon'] = (ds_socat.coords['lon'] + 180) % 360 - 180
        ds_socat = ds_socat.sortby(ds_socat.lon)

    # 4. Time Intersection
    common_start = max(ds_socat.time[0].values, ds_chl.time[0].values)
    common_end = min(ds_socat.time[-1].values, ds_chl.time[-1].values)
    logger.info(f"   Time overlap: {str(common_start)[:10]} to {str(common_end)[:10]}")
    
    ds_socat = ds_socat.sel(time=slice(common_start, common_end))
    ds_chl = ds_chl.sel(time=slice(common_start, common_end))

    # 5. Regrid: Align Dense Biology to Sparse Physics
    # STRATEGY CHANGE: Do not interpolate SOCAT. Interpolate Chlorophyll TO SOCAT.
    logger.info("2. Downsampling Chlorophyll to match SOCAT grid...")
    
    # interp_like matches the grid of 'ds_socat' exactly
    ds_chl_aligned = ds_chl['chl'].interp_like(ds_socat, method='linear')

    # 6. Merge
    logger.info("3. Merging...")
    log_chl = np.log10(ds_chl_aligned + 1e-6)
    log_chl.name = 'log_chl'

    ds_merged = xr.merge([ds_socat, log_chl])
    ds_merged = add_cyclic_time(ds_merged)

    # 7. Save
    logger.info("4. Saving...")
    comp = dict(zlib=True, complevel=5)
    encoding = {var: comp for var in ds_merged.data_vars}
    ds_merged.to_netcdf(OUTPUT_PATH, encoding=encoding)
    logger.info(f"✅ Success! Saved to {OUTPUT_PATH}")

if __name__ == "__main__":
    preprocess()