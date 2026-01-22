"""Preprocess raw data: align, merge, split train/test, normalize. Writes to data/processed/."""

import sys
from pathlib import Path

root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(root / "src"))

import logging

import numpy as np
import xarray as xr

from climate_discovery.config import (
    CHL_PATH,
    END_YEAR,
    FUSED_NC,
    PROCESSED_DIR,
    SCALERS_NC,
    SOCAT_PATH,
    SPLIT_YEAR,
    START_YEAR,
    TEST_NC,
    TRAIN_NC,
)

TRAIN_OUTPUT_PATH = TRAIN_NC
TEST_OUTPUT_PATH = TEST_NC
SCALER_OUTPUT_PATH = SCALERS_NC

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger(__name__)


def add_cyclic_time(ds):
    month = ds.time.dt.month
    ds["sin_month"] = np.sin(2 * np.pi * month / 12)
    ds["cos_month"] = np.cos(2 * np.pi * month / 12)
    ds["year_feature"] = (ds.time.dt.year - START_YEAR) / (END_YEAR - START_YEAR)
    return ds


def preprocess():
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    logger.info("Loading datasets...")
    try:
        ds_socat = xr.open_dataset(SOCAT_PATH, decode_times=True)
        ds_chl = xr.open_dataset(CHL_PATH, decode_times=True)
    except FileNotFoundError as e:
        logger.error("File not found: %s", e)
        sys.exit(1)

    var_map = {
        "fco2_ave_unwtd": "fco2",
        "sst_ave_unwtd": "sst",
        "salinity_ave_unwtd": "sss",
    }
    dim_map = {"tmnth": "time", "ylat": "lat", "xlon": "lon"}
    rename_all = {
        k: v
        for k, v in {**var_map, **dim_map}.items()
        if k in ds_socat.dims or k in ds_socat
    }
    ds_socat = ds_socat.rename(rename_all)[["fco2", "sst", "sss"]]

    ds_chl = ds_chl.rename(
        {
            k: v
            for k, v in {"latitude": "lat", "longitude": "lon"}.items()
            if k in ds_chl.dims
        }
    )
    if "depth" in ds_chl.dims:
        ds_chl = ds_chl.isel(depth=0)

    lon_name = "lon" if "lon" in ds_socat.coords else "xlon"
    lon = ds_socat.coords[lon_name]
    if float(lon.max()) > 180:
        ds_socat = ds_socat.assign_coords(**{lon_name: (lon + 180) % 360 - 180}).sortby(
            lon_name
        )

    tdim = "time" if "time" in ds_socat.dims else "tmnth"
    start_time, end_time = f"{START_YEAR}-01-01", f"{END_YEAR}-12-31"
    ds_socat = ds_socat.sel({tdim: slice(start_time, end_time)})
    ds_chl = ds_chl.sel(time=slice(start_time, end_time))

    chl_var = (
        "chl"
        if "chl" in ds_chl
        else [v for v in ds_chl.data_vars if "chl" in v.lower()][0]
    )
    chl = np.log10(ds_chl[chl_var].interp_like(ds_socat, method="linear") + 1e-6)
    chl.name = "log_chl"
    ds_merged = xr.merge([ds_socat, chl])
    ds_merged = add_cyclic_time(ds_merged)

    ds_train = ds_merged.sel({tdim: slice(None, f"{SPLIT_YEAR - 1}-12-31")})
    ds_test = ds_merged.sel({tdim: slice(f"{SPLIT_YEAR}-01-01", None)})
    logger.info(
        "Train steps: %d | Test steps: %d", len(ds_train[tdim]), len(ds_test[tdim])
    )

    scalers = xr.Dataset()
    for var in ["sst", "sss", "log_chl"]:
        m = ds_train[var].mean(dim=[d for d in ds_train[var].dims])
        s = ds_train[var].std(dim=[d for d in ds_train[var].dims])
        scalers[f"{var}_mean"] = m
        scalers[f"{var}_std"] = s
        ds_train[var] = (ds_train[var] - m) / (s + 1e-6)
        ds_test[var] = (ds_test[var] - m) / (s + 1e-6)

    comp = dict(zlib=True, complevel=5)
    enc = {v: comp for v in ds_train.data_vars}
    ds_train.to_netcdf(TRAIN_OUTPUT_PATH, encoding=enc)
    ds_test.to_netcdf(TEST_OUTPUT_PATH, encoding={v: comp for v in ds_test.data_vars})
    scalers.to_netcdf(SCALER_OUTPUT_PATH)

    ds_fused = xr.concat([ds_train, ds_test], dim=tdim).sortby(tdim)
    ds_fused.to_netcdf(FUSED_NC, encoding={v: comp for v in ds_fused.data_vars})

    logger.info(
        "Saved: %s, %s, %s, %s",
        TRAIN_OUTPUT_PATH,
        TEST_OUTPUT_PATH,
        FUSED_NC,
        SCALER_OUTPUT_PATH,
    )


if __name__ == "__main__":
    preprocess()
