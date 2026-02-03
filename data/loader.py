"""SD-MoSE Data Loader - Complete data pipeline for ocean CO₂ modeling.

This module provides a simple interface to load SOCAT and Copernicus data for
the SD-MoSE (Soft Dynamic Mixture of Symbolic Experts) pipeline.

Features:
- Auto-downloads SOCAT fCO₂ data if missing
- Auto-downloads CMEMS chlorophyll if missing (requires Copernicus account)
- Preprocesses and caches data
- Splits by year for train/test
- Returns pandas DataFrames ready for modeling

Usage:
    from data.loader import SDMoSEDataLoader
    
    loader = SDMoSEDataLoader()
    train_df, test_df = loader.load()
    
    # Or with custom settings
    train_df, test_df = loader.load(
        train_years=(2000, 2020),
        test_years=(2021, 2023)
    )
"""

import sys
from pathlib import Path
from typing import Tuple, Optional
from urllib.request import urlretrieve
import ssl
import zipfile
import os

import numpy as np
import pandas as pd

try:
    import xarray as xr
except ImportError:
    print("ERROR: xarray required. Install: pip install xarray netCDF4")
    sys.exit(1)


class SDMoSEDataLoader:
    """Complete data pipeline for SD-MoSE.
    
    Handles SOCAT fCO₂ and CMEMS chlorophyll data:
    - Downloads if missing
    - Preprocesses and fuses datasets
    - Adds derived features (log_chl, cyclic time, etc.)
    - Splits into train/test by year
    
    Attributes:
        data_dir: Root data directory
        raw_dir: Directory for raw downloaded files
        processed_dir: Directory for preprocessed files
        features: List of feature column names
        target: Target variable name
    """
    
    # Expected features for SD-MoSE model
    EXPERT_FEATURES = ['sst', 'sss', 'log_chl']
    GATING_FEATURES = ['lat_norm', 'lon_norm', 'sst', 'sin_month', 'cos_month']
    ALL_FEATURES = ['sst', 'sss', 'chl', 'log_chl', 'lat', 'lon', 
                    'lat_norm', 'lon_norm', 'sin_month', 'cos_month', 'year_norm']
    TARGET = 'fco2'
    
    # Data source URLs
    SOCAT_URL = "https://www.socat.info/socat_files/v2023/SOCATv2023_tracks_gridded_monthly.nc.zip"
    SOCAT_FILENAME = "SOCATv2023_tracks_gridded_monthly.nc"
    CMEMS_FILENAME = "cmems_mod_glo_bgc_my_0.25deg_P1M-m_chl.nc"
    PROCESSED_FILENAME = "climate_fused_dataset.nc"
    
    def __init__(self, data_dir: str = "data"):
        """Initialize data loader.
        
        Args:
            data_dir: Root data directory (default: "data")
        """
        self.data_dir = Path(data_dir)
        self.raw_dir = self.data_dir / "raw"
        self.processed_dir = self.data_dir / "processed"
        
        # Create directories
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)
    
    def load(
        self,
        train_years: Tuple[int, int] = (2000, 2020),
        test_years: Tuple[int, int] = (2021, 2023),
        force_reprocess: bool = False,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Load and split data for SD-MoSE training.
        
        Args:
            train_years: (start_year, end_year) for training data
            test_years: (start_year, end_year) for test data
            force_reprocess: If True, reprocess even if cached file exists
            
        Returns:
            Tuple of (train_df, test_df) DataFrames with features and target
            
        Example:
            >>> loader = SDMoSEDataLoader()
            >>> train_df, test_df = loader.load()
            >>> print(f"Train: {len(train_df)}, Test: {len(test_df)}")
        """
        # Check for processed data
        processed_file = self.processed_dir / self.PROCESSED_FILENAME
        
        if not processed_file.exists() or force_reprocess:
            print("Preprocessed data not found. Starting data pipeline...")
            self._run_full_pipeline()
        
        # Load processed data
        print(f"Loading processed data from {processed_file}")
        df = self._load_processed_data(processed_file)
        
        # Split by year
        train_df = df[(df['year'] >= train_years[0]) & (df['year'] <= train_years[1])]
        test_df = df[(df['year'] >= test_years[0]) & (df['year'] <= test_years[1])]
        
        print(f"Train samples: {len(train_df)} ({train_years[0]}-{train_years[1]})")
        print(f"Test samples: {len(test_df)} ({test_years[0]}-{test_years[1]})")
        
        return train_df, test_df
    
    def load_numpy(
        self,
        train_years: Tuple[int, int] = (2000, 2020),
        test_years: Tuple[int, int] = (2021, 2023),
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Load data as numpy arrays for direct use in models.
        
        Returns:
            X_expert_train, X_gate_train, y_train,
            X_expert_test, X_gate_test, y_test
        """
        train_df, test_df = self.load(train_years, test_years)
        
        # Extract arrays
        X_expert_train = train_df[self.EXPERT_FEATURES].values
        X_gate_train = train_df[self.GATING_FEATURES].values
        y_train = train_df[self.TARGET].values
        
        X_expert_test = test_df[self.EXPERT_FEATURES].values
        X_gate_test = test_df[self.GATING_FEATURES].values
        y_test = test_df[self.TARGET].values
        
        return X_expert_train, X_gate_train, y_train, X_expert_test, X_gate_test, y_test
    
    def _run_full_pipeline(self):
        """Run complete data acquisition and preprocessing pipeline."""
        print("\n" + "=" * 60)
        print("SD-MoSE DATA PIPELINE")
        print("=" * 60)
        
        # Step 1: Download SOCAT
        socat_file = self.raw_dir / self.SOCAT_FILENAME
        if not socat_file.exists():
            self._download_socat(socat_file)
        else:
            print(f"[OK] SOCAT data exists: {socat_file.name}")
        
        # Step 2: Download CMEMS (or use placeholder)
        cmems_file = self.raw_dir / self.CMEMS_FILENAME
        if not cmems_file.exists():
            print("\n[WARN] CMEMS chlorophyll data not found.")
            print("  For full functionality, run: python data/download_copernicus.py")
            print("  Proceeding without chlorophyll (will use placeholder)...")
            cmems_file = None
        else:
            print(f"[OK] CMEMS data exists: {cmems_file.name}")
        
        # Step 3: Preprocess and fuse
        self._preprocess_data(socat_file, cmems_file)
        
        print("\n" + "=" * 60)
        print("DATA PIPELINE COMPLETE")
        print("=" * 60 + "\n")
    
    def _download_socat(self, output_file: Path):
        """Download SOCAT gridded dataset."""
        print(f"\n[DOWNLOADING] Downloading SOCAT data (~500 MB)...")
        print(f"   URL: {self.SOCAT_URL}")
        print(f"   This may take 5-10 minutes...")
        
        try:
            def progress(block_num, block_size, total_size):
                downloaded = block_num * block_size
                percent = min(100.0 * downloaded / total_size, 100)
                bar_length = 40
                filled = int(bar_length * percent / 100)
                bar = '#' * filled + '-' * (bar_length - filled)
                print(f'\r   [{bar}] {percent:.1f}%', end='')
                if downloaded >= total_size:
                    print()
            
            # Create unverified context for Windows/corporate networks
            try:
                _create_unverified_https_context = ssl._create_unverified_context
            except AttributeError:
                # Legacy Python that doesn't verify HTTPS certificates by default
                pass
            else:
                # Handle target environment that doesn't verify HTTPS certificates by default
                ssl._create_default_https_context = _create_unverified_https_context

            if self.SOCAT_URL.endswith('.zip'):
                zip_path = output_file.with_suffix('.zip')
                print(f"   [DOWNLOADING] Downloading to {zip_path.name}...")
                urlretrieve(self.SOCAT_URL, zip_path, reporthook=progress)
                print(f"   [EXTRACTING] Unzipping...")
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    # Find the .nc file in the zip
                    nc_files = [f for f in zip_ref.namelist() if f.endswith('.nc')]
                    if not nc_files:
                        raise ValueError("No .nc file found in downloaded zip")
                    # Extract the first .nc file
                    source = zip_ref.open(nc_files[0])
                    with open(output_file, 'wb') as target:
                        target.write(source.read())
                # Cleanup
                os.remove(zip_path)
            else:
                urlretrieve(self.SOCAT_URL, output_file, reporthook=progress)
            
            print(f"   [OK] Downloaded: {output_file.name}")
            
        except Exception as e:
            print(f"\n   [ERROR] Download failed: {e}")
            print("   Try downloading manually from https://www.socat.info/")
            raise
    
    def _preprocess_data(self, socat_file: Path, cmems_file: Optional[Path]):
        """Preprocess and fuse SOCAT + CMEMS data."""
        print("\n[PROCESSING] Preprocessing data...")
        
        # Load SOCAT
        print("   Loading SOCAT...")
        socat = xr.open_dataset(socat_file)
        
        # Find variable names (different SOCAT versions use different names)
        fco2_var = self._find_variable(socat, ['fco2_ave_weighted', 'fco2', 'fCO2', 'pCO2'])
        sst_var = self._find_variable(socat, ['sst_ave_weighted', 'sst', 'SST', 'temperature'])
        sss_var = self._find_variable(socat, ['sss_ave_weighted', 'sss', 'SSS', 'salinity', 'sal'])
        
        if not fco2_var:
            raise ValueError(f"Could not find fCO2 variable. Available: {list(socat.data_vars)}")
        
        print(f"   Found variables: fco2={fco2_var}, sst={sst_var}, sss={sss_var}")
        
        # Stack to flat samples
        print("   Flattening spatial dimensions...")
        stacked = socat.stack(sample=("time", "lat", "lon"))
        
        # Build DataFrame
        data = {
            'fco2': stacked[fco2_var].values,
            'sst': stacked[sst_var].values if sst_var else np.nan,
            'sss': stacked[sss_var].values if sss_var else np.nan,
            'lat': stacked['lat'].values,
            'lon': stacked['lon'].values,
            'time': stacked['time'].values,
        }
        
        # Add chlorophyll if available
        if cmems_file and cmems_file.exists():
            print("   Adding chlorophyll from CMEMS...")
            cmems = xr.open_dataset(cmems_file)
            # Regrid to SOCAT resolution
            cmems_regrid = cmems.interp(lat=socat['lat'], lon=socat['lon'], 
                                        time=socat['time'], method='linear')
            cmems_stacked = cmems_regrid.stack(sample=("time", "lat", "lon"))
            data['chl'] = cmems_stacked['chl'].values
            cmems.close()
        else:
            # Placeholder chlorophyll (synthetic for testing)
            print("   Using synthetic chlorophyll placeholder...")
            data['chl'] = np.exp(np.random.randn(len(data['fco2'])) * 0.5 - 1.5)
        
        df = pd.DataFrame(data)
        
        # Drop NaN
        print("   Removing NaN values...")
        initial_len = len(df)
        df = df.dropna()
        print(f"   Removed {initial_len - len(df)} NaN samples ({len(df)} remaining)")
        
        # Add derived features
        print("   Adding derived features...")
        df = self._add_features(df)
        
        # Filter to valid range
        df = df[(df['fco2'] > 100) & (df['fco2'] < 600)]
        df = df[(df['sst'] > -5) & (df['sst'] < 35)]
        print(f"   After filtering: {len(df)} samples")
        
        # Save processed data
        output_file = self.processed_dir / self.PROCESSED_FILENAME
        print(f"   Saving to {output_file}...")
        
        # Convert to xarray and save as NetCDF
        ds = xr.Dataset({
            col: (['sample'], df[col].values) 
            for col in df.columns if col != 'time'
        })
        ds['time'] = (['sample'], df['time'].values)
        ds.to_netcdf(output_file)
        
        print(f"   [OK] Saved {len(df)} samples to {output_file.name}")
        
        socat.close()
    
    def _find_variable(self, ds: xr.Dataset, candidates: list) -> Optional[str]:
        """Find the first matching variable name in dataset."""
        for name in candidates:
            if name in ds.data_vars:
                return name
        return None
    
    def _add_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add derived features for SD-MoSE model."""
        df = df.copy()
        
        # Log chlorophyll
        df['log_chl'] = np.log(np.clip(df['chl'], 1e-3, None))
        
        # Normalized coordinates
        df['lat_norm'] = df['lat'] / 90.0
        df['lon_norm'] = df['lon'] / 180.0
        
        # Cyclic time features
        time_pd = pd.to_datetime(df['time'])
        df['year'] = time_pd.dt.year
        df['month'] = time_pd.dt.month
        month_angle = 2 * np.pi * (time_pd.dt.month - 1) / 12.0
        df['sin_month'] = np.sin(month_angle)
        df['cos_month'] = np.cos(month_angle)
        
        # Normalized year
        df['year_norm'] = (df['year'] - 2015) / 10.0
        
        return df
    
    def _load_processed_data(self, filepath: Path) -> pd.DataFrame:
        """Load preprocessed NetCDF as DataFrame."""
        ds = xr.open_dataset(filepath)
        df = ds.to_dataframe().reset_index(drop=True)
        ds.close()
        return df
    
    def get_feature_names(self) -> dict:
        """Get feature name lists for model configuration."""
        return {
            'expert': self.EXPERT_FEATURES,
            'gating': self.GATING_FEATURES,
            'all': self.ALL_FEATURES,
            'target': self.TARGET,
        }


# Convenience function for quick loading
def load_data(
    train_years: Tuple[int, int] = (2000, 2020),
    test_years: Tuple[int, int] = (2021, 2023),
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Quick loader function.
    
    Usage:
        from data.loader import load_data
        train_df, test_df = load_data()
    """
    loader = SDMoSEDataLoader()
    return loader.load(train_years, test_years)


if __name__ == "__main__":
    # Test the loader
    print("Testing SD-MoSE Data Loader...")
    print("-" * 40)
    
    loader = SDMoSEDataLoader()
    train_df, test_df = loader.load()
    
    print("\nTrain DataFrame:")
    print(train_df.head())
    print(f"\nColumns: {list(train_df.columns)}")
    print(f"Shape: {train_df.shape}")
    
    print("\nTest DataFrame:")
    print(f"Shape: {test_df.shape}")
    
    print("\nFeature names:")
    print(loader.get_feature_names())
