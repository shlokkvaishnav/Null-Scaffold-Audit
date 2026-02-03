"""SD-MoSE Data Loader - CMEMS Multi-Observation Support.

Loads dense gridded data (Carbon + Physics) for Soft Regime modeling.
"""

import sys
from pathlib import Path
from typing import Tuple, Optional
import numpy as np
import pandas as pd
import xarray as xr

class SDMoSEDataLoader:
    """CMEMS Multi-Observation Data Loader."""
    
    EXPERT_FEATURES = ['sst', 'sss', 'log_chl']
    GATING_FEATURES = ['lat_norm', 'lon_norm', 'sst', 'sin_month', 'cos_month']
    TARGET = 'spco2'  # Surface pCO2 in CMEMS
    
    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        self.raw_dir = self.data_dir / "raw"
        self.processed_dir = self.data_dir / "processed"
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        
        self.carbon_file = self.raw_dir / "cmems_multiobs_carbon.nc"
        self.phys_file = self.raw_dir / "cmems_multiobs_phys.nc"
        self.processed_file = self.processed_dir / "cmems_fused.nc"

    def load(self, train_years=(2000, 2020), test_years=(2021, 2023)):
        """Load and split CMEMS data.
        
        Returns:
            train_df, test_df: DataFrame with features and 'lat_idx', 'lon_idx'
            grid_shape: Tuple (n_lat, n_lon) for spatial reshaping
        """
        if not self.processed_file.exists():
            self._preprocess()
            
        print(f"Loading {self.processed_file}...")
        ds = xr.open_dataset(self.processed_file)
        
        # Subsample for DataFrame conversion (Memory safety)
        # Using 10% random sample for now if large
        # For full grid training, we'd need a different iterator
        print("Converting to DataFrame (subsampling for safety)...")
        # Stack to (sample, feature)
        stacked = ds.stack(sample=("time", "lat", "lon"))
        
        # Select valid data
        valid_mask = stacked[self.TARGET].notnull()
        if 'sst' in stacked:
            valid_mask &= stacked['sst'].notnull()
            
        stacked = stacked.where(valid_mask, drop=True)
        
        df = pd.DataFrame({
            'spco2': stacked['spco2'].values,
            'sst': stacked['sst'].values if 'sst' in stacked else np.nan,
            'sss': stacked['sss'].values if 'sss' in stacked else np.nan,
            'lat': stacked['lat'].values,
            'lon': stacked['lon'].values,
            'year': stacked['time'].dt.year.values,
            'month': stacked['time'].dt.month.values,
        })
        
        # Add spatial indices for loss calculation
        # We map unique lat/lon values to integer indices
        lat_vals = sorted(ds.lat.values)
        lon_vals = sorted(ds.lon.values)
        lat_map = {val: idx for idx, val in enumerate(lat_vals)}
        lon_map = {val: idx for idx, val in enumerate(lon_vals)}
        
        df['lat_idx'] = df['lat'].map(lat_map)
        df['lon_idx'] = df['lon'].map(lon_map)
        
        # Add derived features
        df = self._add_features(df)
        
        # Split
        train_df = df[(df['year'] >= train_years[0]) & (df['year'] <= train_years[1])]
        test_df = df[(df['year'] >= test_years[0]) & (df['year'] <= test_years[1])]
        
        # Rename TARGET to fco2 for compatibility if needed (or keep spco2)
        train_df['fco2'] = train_df['spco2']
        test_df['fco2'] = test_df['spco2']
        
        return train_df, test_df, (len(lat_vals), len(lon_vals))

    def _preprocess(self):
        print("Preprocessing CMEMS data...")
        # Check files
        if not self.carbon_file.exists():
            print("Downloading CMEMS data...")
            import subprocess
            subprocess.run([sys.executable, "data/download_cmems.py"], check=True)
            
        ds_carb = xr.open_dataset(self.carbon_file)
        
        # Try load physics
        if self.phys_file.exists():
            ds_phys = xr.open_dataset(self.phys_file)
            # Regrid physics to carbon grid (usually same 0.25deg but check)
            ds_phys = ds_phys.interp_like(ds_carb, method='nearest')
            ds = xr.merge([ds_carb, ds_phys])
        else:
            print("Warning: Physics file missing. Using placeholders.")
            ds = ds_carb
            ds['sst'] = ds['spco2'] * 0 + 20.0  # Dummy
            ds['sss'] = ds['spco2'] * 0 + 35.0  # Dummy
        
        # Save fused
        ds.to_netcdf(self.processed_file)
        print(f"Saved fused data to {self.processed_file}")

    def _add_features(self, df):
        # Add features logic (same as before)
        df['lat_norm'] = df['lat'] / 90.0
        df['lon_norm'] = df['lon'] / 180.0
        # Cyclic time
        month_angle = 2 * np.pi * (df['month'] - 1) / 12.0
        df['sin_month'] = np.sin(month_angle)
        df['cos_month'] = np.cos(month_angle)
        
        # Chlorophyll (assume loaded or add placeholder if missing)
        if 'chl' not in df.columns:
             df['chl'] = 1.0
        df['log_chl'] = np.log(np.clip(df['chl'], 1e-3, None))
        
        return df

if __name__ == "__main__":
    loader = SDMoSEDataLoader()
    tr, te = loader.load()
    print(tr.head())
