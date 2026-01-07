import xarray as xr
import numpy as np
import logging
from pathlib import Path

# Setup Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger(__name__)

# --- PATHS ---
# --- PATHS ---
# Anchor paths to the project root (one level up from this script)
PROJECT_ROOT = Path(__file__).parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "01_raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "03_processed"

SOCAT_PATH = RAW_DIR / "SOCATv2025_tracks_gridded_monthly.nc"
CHL_PATH = RAW_DIR / "cmems_mod_glo_bgc_my_0.25deg_P1M-m_chl.nc"
TRAIN_OUTPUT_PATH = PROCESSED_DIR / "train_dataset.nc"
TEST_OUTPUT_PATH = PROCESSED_DIR / "test_dataset.nc"
SCALER_OUTPUT_PATH = PROCESSED_DIR / "scalers.nc"

# --- CONFIGURATION ---
# "5 - 10 years" recommendation
START_YEAR = 2015
SPLIT_YEAR = 2022 # Train: 2015-2021, Test: 2022-2024
END_YEAR = 2024

def add_cyclic_time(ds):
    """Adds sin/cos month features."""
    month = ds.time.dt.month
    ds['sin_month'] = np.sin(2 * np.pi * month / 12)
    ds['cos_month'] = np.cos(2 * np.pi * month / 12)
    ds['year_feature'] = (ds.time.dt.year - START_YEAR) / (END_YEAR - START_YEAR) # Normalized Year
    return ds

def preprocess():
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    
    # 1. Load Data
    logger.info("1. Loading Datasets...")
    try:
        ds_socat = xr.open_dataset(SOCAT_PATH, decode_times=True)
        ds_chl = xr.open_dataset(CHL_PATH, decode_times=True)
    except FileNotFoundError as e:
        logger.error(f"❌ File not found: {e}")
        return

    # 2. Rename Variables
    rename_socat = {
        'fco2_ave_unwtd': 'fco2',
        'sst_ave_unwtd': 'sst',
        'salinity_ave_unwtd': 'sss', 
        'tmnth': 'time', 'ylat': 'lat', 'xlon': 'lon'
    }
    ds_socat = ds_socat.rename({k: v for k, v in rename_socat.items() if k in ds_socat})
    ds_socat = ds_socat[['fco2', 'sst', 'sss']] 

    rename_chl = {'latitude': 'lat', 'longitude': 'lon'}
    ds_chl = ds_chl.rename({k: v for k, v in rename_chl.items() if k in ds_chl.dims})
    
    if 'depth' in ds_chl.dims: 
        ds_chl = ds_chl.isel(depth=0)

    # 3. Handle Longitude (Align -180..180)
    if ds_socat.lon.max() > 180:
        logger.info("   Adjusting SOCAT longitude to -180..180...")
        ds_socat.coords['lon'] = (ds_socat.coords['lon'] + 180) % 360 - 180
        ds_socat = ds_socat.sortby(ds_socat.lon)

    # 4. Time Slicing (The "Manageable Data" Trick)
    start_time = f"{START_YEAR}-01-01"
    end_time = f"{END_YEAR}-12-31"
    logger.info(f"   Slicing data to {start_time} to {end_time}...")
    
    ds_socat = ds_socat.sel(time=slice(start_time, end_time))
    ds_chl = ds_chl.sel(time=slice(start_time, end_time))

    # 5. Regrid: Align Dense Biology to Sparse Physics
    logger.info("2. Downsampling Chlorophyll to match SOCAT grid...")
    ds_chl_aligned = ds_chl['chl'].interp_like(ds_socat, method='linear')

    # 6. Merge & Log Transform
    logger.info("3. Merging...")
    log_chl = np.log10(ds_chl_aligned + 1e-6)
    log_chl.name = 'log_chl'

    ds_merged = xr.merge([ds_socat, log_chl])
    ds_merged = add_cyclic_time(ds_merged)

    # 7. Train/Test Split
    logger.info(f"4. Splitting Data (Split Year: {SPLIT_YEAR})...")
    ds_train = ds_merged.sel(time=slice(None, f"{SPLIT_YEAR-1}-12-31"))
    ds_test = ds_merged.sel(time=slice(f"{SPLIT_YEAR}-01-01", None))
    
    logger.info(f"   Train samples (time steps): {len(ds_train.time)}")
    logger.info(f"   Test samples (time steps): {len(ds_test.time)}")

    # 8. Normalization (Fit on Train, Apply to Both)
    logger.info("5. Normalizing Features...")
    features = ['sst', 'sss', 'log_chl']
    scalers = xr.Dataset()

    for var in features:
        mean = ds_train[var].mean(dim=['time', 'lat', 'lon'])
        std = ds_train[var].std(dim=['time', 'lat', 'lon'])
        
        # Store scalers
        scalers[f"{var}_mean"] = mean
        scalers[f"{var}_std"] = std
        
        # Apply standard scaling
        ds_train[var] = (ds_train[var] - mean) / (std + 1e-6)
        ds_test[var] = (ds_test[var] - mean) / (std + 1e-6)

    # 9. Save
    logger.info("6. Saving datasets...")
    comp = dict(zlib=True, complevel=5)
    
    # Save Train
    encoding_train = {var: comp for var in ds_train.data_vars}
    ds_train.to_netcdf(TRAIN_OUTPUT_PATH, encoding=encoding_train)
    
    # Save Test
    encoding_test = {var: comp for var in ds_test.data_vars}
    ds_test.to_netcdf(TEST_OUTPUT_PATH, encoding=encoding_test)
    
    # Save Scalers
    scalers.to_netcdf(SCALER_OUTPUT_PATH)

    logger.info(f"✅ Success!")
    logger.info(f"   Train: {TRAIN_OUTPUT_PATH}")
    logger.info(f"   Test:  {TEST_OUTPUT_PATH}")
    logger.info(f"   Scalers: {SCALER_OUTPUT_PATH}")

if __name__ == "__main__":
    preprocess()